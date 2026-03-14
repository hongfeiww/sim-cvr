import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingTable(nn.Module):
    '''
    All features share the same embedding dimension for simplicity;
    in production, different dims per feature.
    '''
 
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int,
                 padding_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(int(size), embed_dim, padding_idx=padding_idx)
            for name, size in vocab_sizes.items()
        })
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            nn.init.zeros_(emb.weight[0])
 
    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            nn.init.zeros_(emb.weight[0])  # keep pad vector at zero
 
    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: self.embeddings[name](idx)
                for name, idx in feature_dict.items()
                if name in self.embeddings}
 
    def lookup_sequence(self, seq_ids: torch.Tensor, feature_name: str) -> torch.Tensor:
        '''Look up embeddings for a [B, L] sequence of ids → [B, L, D].'''
        return self.embeddings[feature_name](seq_ids)
class GSU(nn.Module):
    '''
    fast top-K retrieval from long user click sequence.
    pure DIN: attention over full L items → O(L) attention cost, noisy for L>200
    GSU reduce to K → O(K) attention cost, higher signal-to-noise ratio
    In production at Kuaishou, L can be 5000+; K=50~100 is the sweet spot.
    '''

    def __init__(self, mode: str = 'hard', embed_dim: int = 32):
        super().__init__()
        assert mode in ('hard', 'soft') # hard: category match; soft: inner-product relevance
        self.mode = mode
        if mode == 'soft':
            self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        target_emb: torch.Tensor,
        target_category: torch.Tensor,
        seq_emb: torch.Tensor,
        seq_category: torch.Tensor, 
        seq_mask: torch.Tensor,
        top_k: int = 20,
    ):
        B, L, D = seq_emb.shape

        if self.mode == 'hard':
            # Category match: 1.0 for match, -1e9 for mismatch or padding
            match = (seq_category == target_category.unsqueeze(1)) & seq_mask
            scores = match.float().masked_fill(~seq_mask, -1e9)
        else:
            q = self.proj(target_emb).unsqueeze(2)
            scores = torch.bmm(seq_emb, q).squeeze(2) / (D ** 0.5)
            scores = scores.masked_fill(~seq_mask, -1e9)

        k = min(top_k, L)
        topk_scores, topk_idx = torch.topk(scores, k, dim=1)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        retrieved_emb  = torch.gather(seq_emb, 1, idx_exp)
        retrieved_mask = topk_scores > -1e8

        return retrieved_emb, retrieved_mask


class TargetAttention(nn.Module):
    '''
    DIN-style interaction between target item and retrieved behavior sequence.
    Activation unit input: [target ‖ behavior ‖ target−behavior ‖ target⊙behavior]
    '''

    def __init__(self, embed_dim: int, use_softmax: bool = True):
        super().__init__()
        self.scale = math.sqrt(embed_dim)
        self.use_softmax = use_softmax
        self.activation_unit = MLP(
            input_dim=embed_dim * 4,
            hidden_dims=[64],
            output_dim=1,
            dropout=0.0,
            use_bn=False,
        )

    def forward(self, target: torch.Tensor, sequence: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:        
        B, L, D = sequence.shape
        t_exp = target.unsqueeze(1).expand(-1, L, -1) 
        feat  = torch.cat([t_exp, sequence, t_exp - sequence,
                           t_exp * sequence], dim=-1) # [B, L, 4D]
        scores = self.activation_unit(feat.view(B * L, -1)).view(B, L)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = (F.softmax(scores / self.scale, dim=-1)
                   if self.use_softmax else torch.sigmoid(scores))
        weights = torch.nan_to_num(weights, nan=0.0)

        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1) # [B, D]
