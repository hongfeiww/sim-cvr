'''
ESMM: joint ctr and ctcvr training.

Paper: Ma et al., 'Entire Space Multi-Task Model: An Effective Approach
       for Estimating Post-Click Conversion Rate', SIGIR 2018.
Core idea of it:
  Naive CVR training only uses clicked samples (purchase given click).
  Training space (clicks) ≠ Inference space (all impressions).
  Sample selection bias: model never sees non-clicked impressions.
  ESMM fixes this by decomposing pCTCVR: pCTCVR = pCTR × pCVR

Training obj(both on FULL impression):
  L = w_ctr * BCE(pCTR, click) + BCE(pCTCVR, ctcvr)

Architecture of this impl:
  Shared Embedding → CTR Tower (optional) + CVR Tower (SIM-lite)
'''

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .layers import EmbeddingTable, MLP
from .sim_cvr import SIMCVRTower, ALL_FIELDS


class CTRTower(nn.Module):
    # simple MLP without sequence modeling. Main part is CVR tower.
    def __init__(self, num_fields: int, embed_dim: int,
                 hidden_dims: List[int], dropout: float):
        super().__init__()
        self.dnn = MLP(num_fields * embed_dim, hidden_dims, output_dim=1,
                       dropout=dropout)

    def forward(self, emb_list: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(emb_list, dim=-1) # [B, num_fields * D]
        return self.dnn(x).squeeze(1) # [B]


class ESMM(nn.Module):
    def __init__(
        self,
        vocab_sizes:     Dict[str, int],
        embed_dim:       int   = 18,
        hidden_dims:     List[int] = [360, 200, 80],
        dropout:         float = 0.3,
        top_k:           int   = 20,
        gsu_mode:        str   = 'hard',
        use_ctr_tower:   bool  = True,
        ctr_weight:      float = 1.0,
        ctcvr_weight:    float = 1.0,
    ):
        super().__init__()
        self.use_ctr_tower = use_ctr_tower
        self.ctr_weight    = ctr_weight
        self.ctcvr_weight  = ctcvr_weight

        # Shared embeddings for CTR and CVR
        self.shared_emb = EmbeddingTable(vocab_sizes, embed_dim)

        # CVR tower with SIM-lite
        self.cvr_tower = SIMCVRTower(
            vocab_sizes=vocab_sizes,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            top_k=top_k,
            gsu_mode=gsu_mode,
            emb_table=self.shared_emb, # shared weights
        )

        # CTR tower (optional)
        num_fields = len(ALL_FIELDS)
        if use_ctr_tower:
            self.ctr_tower = CTRTower(num_fields, embed_dim, hidden_dims, dropout)
        else:
            self.ctr_tower = nn.Linear(num_fields * embed_dim, 1)

    def _scalar_embs(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        feat = {f: batch[f] for f in ALL_FIELDS if f in batch}
        emb_dict = self.shared_emb(feat)
        return [emb_dict[f] for f in ALL_FIELDS if f in emb_dict]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # CTR
        scalar_embs = self._scalar_embs(batch)
        if self.use_ctr_tower:
            ctr_logit = self.ctr_tower(scalar_embs) # [B]
        else:
            ctr_logit = self.ctr_tower(
                torch.cat(scalar_embs, dim=-1)
            ).squeeze(1)
        p_ctr = torch.sigmoid(ctr_logit) # [B]
        # CVR
        cvr_logit = self.cvr_tower(
            user_id=batch['user_id'],
            age_level=batch['age_level'],
            gender=batch['gender'],
            shopping_level=batch['shopping_level'],
            city_level=batch['city_level'],
            item_id=batch['item_id'],
            item_category=batch['item_category'],
            item_price_level=batch['item_price_level'],
            item_sales_level=batch['item_sales_level'],
            ad_id=batch['ad_id'],
            campaign_id=batch['campaign_id'],
            customer_id=batch['customer_id'],
            brand_id=batch['brand_id'],
            pid=batch['pid'],
            hour=batch['hour'],
            seq_item_ids=batch['seq_item_ids'],
            seq_categories=batch['seq_categories'],
            seq_mask=batch['seq_mask'],
        ).squeeze(1) # [B]
        p_cvr = torch.sigmoid(cvr_logit)

        p_ctcvr = p_ctr * p_cvr # [B]

        return p_ctr, p_cvr, p_ctcvr

    def predict_ecpm(
        self,
        batch: Dict[str, torch.Tensor],
        bid:   torch.Tensor,
    ) -> torch.Tensor:       
        #  eCPM for ad ranking. eCPM = bid × pCTCVR × 1000     
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = self.forward(batch)
        return bid * p_ctcvr * 1000.0
