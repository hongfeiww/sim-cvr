'''
AUC scores:
  AUC_CTR : AUC of pCTR   vs click on imp
  AUC_CVR : AUC of pCVR   vs purchase on click
  AUC_CTCVR : AUC of pCTCVR vs purchase on imp
  gAUC: impression-weighted average of per-user AUC. 

Cost monitoring:
  cost_ratio = realized_CVR / mean_pCVR
        ideal = 1.0, healthy = [0.8, 1.2] in production.
        cost_ratio < 0.8 → over‑cost (advertisers pay too much for too few conversions)
        cost_ratio > 1.2 → under‑cost (platform misses revenue; advertisers get extra value for free)
        Deviations outside the healthy range trigger alerts for model calibration or bid adjustment.
'''

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, Optional


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray, eps=1e-7) -> float:
    y_score = np.clip(y_score, eps, 1 - eps)
    return float(log_loss(y_true, y_score))


def compute_esmm_metrics(
    p_ctr:    np.ndarray,
    p_cvr:    np.ndarray,
    p_ctcvr:  np.ndarray,
    click:    np.ndarray,
    purchase: np.ndarray,
    bid:      Optional[np.ndarray] = None,
) -> Dict[str, float]:
    click_mask = click == 1

    metrics = {
        # CTR task: all imp
        'auc_ctr':    _safe_auc(click,             p_ctr),
        'logloss_ctr':_safe_logloss(click,         p_ctr),

        # CVR task: clicked samples only
        'auc_cvr':    _safe_auc(purchase[click_mask], p_cvr[click_mask])
                      if click_mask.sum() > 1 else float('nan'),
        'logloss_cvr':_safe_logloss(purchase[click_mask], p_cvr[click_mask])
                      if click_mask.sum() > 1 else float('nan'),

        # CTCVR task: all imp
        'auc_ctcvr':     _safe_auc(purchase,      p_ctcvr),
        'logloss_ctcvr': _safe_logloss(purchase,  p_ctcvr),

        # Distribution stats
        'mean_p_ctr':    float(p_ctr.mean()),
        'mean_p_cvr':    float(p_cvr.mean()),
        'mean_p_ctcvr':  float(p_ctcvr.mean()),
        'realized_ctr':  float(click.mean()),
        'realized_cvr':  float(purchase[click_mask].mean()) if click_mask.sum() > 0 else 0.0,
        'realized_ctcvr':float(purchase.mean()),
    }

    # Cost ratio: realized_CVR / predicted_CVR
    # Using uniform bid=1 for ratio (bid cancels out in ratio)
    if bid is not None:
        expected_spend = (p_ctcvr * bid).sum()
        actual_spend   = (purchase * bid).sum()
    else:
        expected_spend = p_ctcvr.sum()
        actual_spend   = purchase.sum()

    cost_ratio = float(actual_spend / (expected_spend + 1e-12))
    metrics['cost_ratio'] = cost_ratio
    metrics['cost_status'] = (
        'overdelivery'  if cost_ratio > 1.2 else
        'underdelivery' if cost_ratio < 0.8 else
        'healthy'
    )

    return metrics


def compute_gauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    user_ids: np.ndarray,
    min_samples: int = 2,
) -> float:     
    auc_sum = weight_sum = 0.0
    for uid in np.unique(user_ids):
        m = user_ids == uid
        if m.sum() < min_samples or len(np.unique(y_true[m])) < 2:
            continue
        auc_sum    += roc_auc_score(y_true[m], y_score[m]) * m.sum()
        weight_sum += m.sum()
    return float(auc_sum / weight_sum) if weight_sum > 0 else float("nan")