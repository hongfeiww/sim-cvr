'''
Ad Cost Calibration for ad Bidding.
Advertisers set a target CPA (cost-per-acquisition = cost per conversion).
The platform ranks ads by eCPM and charges per impression:

    eCPM      = bid × pCTR × pCVR × 1000
    expected  = bid × pCTCVR          (what advertiser expects to pay per impression)
    actual    = bid × realized_CTCVR  (what platform actually charges)

    cost_ratio = actual / expected = realized_CTCVR / mean_pCTCVR

Ideal state  : cost_ratio ≈ 1.0  (1:1)
Over-delivery: cost_ratio > 1.2  → pCVR under-estimated → underspend
               (advertiser budget not fully used, platform loses revenue)
Under-delivery: cost_ratio < 0.8 → pCVR over-estimated → overspend 
               (advertiser pays more per conversion than their CPA target, ROI is impacted)

Fix: isotonic regression calibration maps raw pCVR → calibrated probabilities
     Applied post-training; does not change model weights or AUC.
'''

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class CostCalibrator:
    def __init__(self):
        self.iso_reg   = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        self.fit_stats: Dict = {}

    def fit(
        self,
        raw_scores: np.ndarray,           # pCVR from model, shape [N]
        labels: np.ndarray,               # purchase labels, shape [N]
        click_mask: Optional[np.ndarray] = None,
    ) -> 'CostCalibrator':
        if click_mask is not None:
            raw_scores = raw_scores[click_mask]
            labels     = labels[click_mask]
            logger.info(f'Calibration fit on clicked samples: n={click_mask.sum():,}')
        else:
            logger.info(f'Calibration fit on all impressions: n={len(raw_scores):,}')

        self.iso_reg.fit(raw_scores, labels)
        self.is_fitted = True

        cal = self.iso_reg.predict(raw_scores)
        self.fit_stats = {
            'n':               int(len(labels)),
            'mean_raw':        float(raw_scores.mean()),
            'mean_calibrated': float(cal.mean()),
            'mean_actual':     float(labels.mean()),
            'raw_ece':         self._ece(raw_scores, labels),
            'calibrated_ece':  self._ece(cal, labels),
        }
        logger.info(
            f"pCVR calibration: raw={self.fit_stats['mean_raw']:.5f} → "
            f"calibrated={self.fit_stats['mean_calibrated']:.5f} | "
            f"actual={self.fit_stats['mean_actual']:.5f} | "
            f"ECE: {self.fit_stats['raw_ece']:.5f} → {self.fit_stats['calibrated_ece']:.5f}"
        )
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('Must call fit() before transform()')
        return self.iso_reg.predict(raw_scores)

    @staticmethod
    def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        # Expected Calibration Error — measures how well probabilities match reality.
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(scores)) * abs(labels[mask].mean() - scores[mask].mean())
        return float(ece)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'iso_reg': self.iso_reg, 'fit_stats': self.fit_stats}, f)
        logger.info(f'Calibrator saved → {path}')

    @classmethod
    def load(cls, path: str) -> 'CostCalibrator':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls()
        obj.iso_reg   = state['iso_reg']
        obj.fit_stats = state['fit_stats']
        obj.is_fitted = True
        return obj


class CostMonitor:
    '''
    Monitors expected vs actual ad spend ratio per evaluation batch.
    cost_ratio = Σ(actual_conversions × bid) / Σ(predicted_conversions × bid)
    With uniform bid this simplifies to:
        cost_ratio = realized_CVR / mean_pCVR
    Also computes per-advertiser (customer_id) breakdowns so ops team can
    identify which campaigns are miscalibrated.
    '''

    HEALTHY_LO = 0.80
    HEALTHY_HI = 1.20

    def __init__(self):
        self.history = []

    def compute(
        self,
        p_ctcvr: np.ndarray, # [N] predicted CVR
        purchase: np.ndarray, # [N] actual purchase label
        bid: Optional[np.ndarray] = None, # [N] per-sample CPA bid
        group_ids: Optional[np.ndarray] = None, # [N] customer_id
    ) -> Dict:
        if bid is None:
            bid = np.ones(len(p_ctcvr), dtype=np.float32)

        expected_spend = (p_ctcvr * bid).sum()
        actual_spend   = (purchase * bid).sum()
        ratio = float(actual_spend / (expected_spend + 1e-12))

        status = (
            'overdelivery'   if ratio > self.HEALTHY_HI else
            'underdelivery'  if ratio < self.HEALTHY_LO else
            'healthy'
        )

        result: Dict = {
            'global_cost_ratio':   ratio,
            'expected_spend':      float(expected_spend),
            'actual_spend':        float(actual_spend),
            'mean_predicted_cvr':  float(p_ctcvr.mean()),
            'mean_realized_cvr':   float(purchase.mean()),
            'status':              status,
        }

        if group_ids is not None:
            per_group = {}
            for gid in np.unique(group_ids):
                m = group_ids == gid
                if m.sum() < 10:
                    continue
                g_exp = (p_ctcvr[m] * bid[m]).sum()
                g_act = (purchase[m] * bid[m]).sum()
                g_ratio = float(g_act / (g_exp + 1e-12))
                per_group[int(gid)] = {
                    'cost_ratio': g_ratio,
                    'n':          int(m.sum()),
                    'status': (
                        'overdelivery'  if g_ratio > self.HEALTHY_HI else
                        'underdelivery' if g_ratio < self.HEALTHY_LO else
                        'healthy'
                    ),
                }
            result['per_group']              = per_group
            result['n_overdelivery_groups']  = sum(
                1 for g in per_group.values() if g['status'] == 'overdelivery')
            result['n_underdelivery_groups'] = sum(
                1 for g in per_group.values() if g['status'] == 'underdelivery')

        self.history.append(result)
        self._log(result)
        return result

    def _log(self, r: Dict) -> None:
        icon = {'healthy': '✓', 'overdelivery': '↑', 'underdelivery': '⚠'}[r['status']]
        logger.info(
            f"[CostMonitor] {icon} ratio={r['global_cost_ratio']:.3f} | "
            f"pCVR={r['mean_predicted_cvr']:.5f} | "
            f"realCVR={r['mean_realized_cvr']:.5f} | "
            f"status={r['status']}"
        )
        if r['status'] == 'underdelivery':
            logger.warning(
                'UNDERDELIVERY: pCVR overestimated. Advertisers overspending. '
                'Apply isotonic calibration or reduce bid multiplier.'
            )