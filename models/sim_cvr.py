'''
SIM-lite CVR Tower.
'''

from typing import Dict, List

import torch
import torch.nn as nn

from .layers import EmbeddingTable, GSU, MLP, TargetAttention

# Feature groups for Ali-CCP
USER_FIELDS = ['user_id', 'age_level', 'gender', 'shopping_level', 'city_level']
ITEM_FIELDS = ['item_id', 'item_category', 'item_price_level', 'item_sales_level']
AD_FIELDS   = ['ad_id', 'campaign_id', 'customer_id', 'brand_id']
CTX_FIELDS  = ['pid', 'hour']
ALL_FIELDS  = USER_FIELDS + ITEM_FIELDS + AD_FIELDS + CTX_FIELDS  # 15 fields


class SIMCVRTower(nn.Module):
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        embed_dim: int = 32,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        top_k: int = 20,
        gsu_mode: str = 'hard',
        emb_table: EmbeddingTable = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k     = top_k

        # Shared embedding table in ESMM
        self.emb_table = emb_table or EmbeddingTable(vocab_sizes, embed_dim)

        self.gsu = GSU(mode=gsu_mode, embed_dim=embed_dim)
        self.esu = TargetAttention(embed_dim=embed_dim, use_softmax=True)

        # DNN input: 15 scalar fields + 1 interest vector = 16 × embed_dim
        dnn_input_dim = (len(ALL_FIELDS) + 1) * embed_dim
        self.dnn = MLP(dnn_input_dim, hidden_dims, output_dim=1, dropout=dropout)

    def forward(
        self,
        # Scalar features
        user_id: torch.Tensor,
        age_level: torch.Tensor,
        gender: torch.Tensor,
        shopping_level: torch.Tensor,
        city_level: torch.Tensor,
        item_id: torch.Tensor,
        item_category: torch.Tensor,
        item_price_level: torch.Tensor,
        item_sales_level: torch.Tensor,
        ad_id: torch.Tensor,
        campaign_id: torch.Tensor,
        customer_id: torch.Tensor,
        brand_id: torch.Tensor,
        pid: torch.Tensor,
        hour: torch.Tensor,
        # Sequence
        seq_item_ids: torch.Tensor, # [B, L]
        seq_categories: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:

        # Embedding lookup 
        feat = {
            'user_id': user_id, 'age_level': age_level, 'gender': gender,
            'shopping_level': shopping_level, 'city_level': city_level,
            'item_id': item_id, 'item_category': item_category,
            'item_price_level': item_price_level, 'item_sales_level': item_sales_level,
            'ad_id': ad_id, 'campaign_id': campaign_id,
            'customer_id': customer_id, 'brand_id': brand_id,
            'pid': pid, 'hour': hour,
        }
        embs    = self.emb_table(feat) # {name: [B, D]}
        seq_emb = self.emb_table.lookup_sequence(seq_item_ids, 'item_id') # [B, L, D]

        # GSU 
        retrieved_emb, retrieved_mask = self.gsu(
            target_emb=embs['item_id'],
            target_category=item_category,
            seq_emb=seq_emb,
            seq_category=seq_categories,
            seq_mask=seq_mask,
            top_k=self.top_k,
        )

        # ESU [B, D]
        interest = self.esu(
            target=embs['item_id'],
            sequence=retrieved_emb,
            mask=retrieved_mask,
        )

        # Concat all embeddings
        all_embs = [embs[f] for f in ALL_FIELDS] + [interest] # 16 × [B, D]
        dnn_input = torch.cat(all_embs, dim=-1) # [B, 16D]

        return self.dnn(dnn_input) # [B, 1]
