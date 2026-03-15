'''
Unit tests for ESMM, SIM-lite CVR tower, layers, metrics, and calibration.
All tests run without GPU or real data.
'''

import numpy as np
import pytest
import torch
import torch.nn as nn

# Shared fixtures

VOCAB = {
    'user_id': 101, 'age_level': 8, 'gender': 4, 'shopping_level': 4,
    'city_level': 6, 'item_id': 501, 'item_category': 51,
    'item_price_level': 6, 'item_sales_level': 6,
    'ad_id': 201, 'campaign_id': 51, 'customer_id': 31, 'brand_id': 81,
    'pid': 4, 'hour': 25,
}
B, L, D = 8, 20, 32


def make_batch(B=B, L=L):
    seq_mask = torch.zeros(B, L, dtype=torch.bool)
    for i in range(B):
        sl = torch.randint(1, L, (1,)).item()
        seq_mask[i, -sl:] = True
    return {
        'user_id':          torch.randint(1, 100, (B,)),
        'age_level':        torch.randint(1, 7,   (B,)),
        'gender':           torch.randint(1, 3,   (B,)),
        'shopping_level':   torch.randint(1, 3,   (B,)),
        'city_level':       torch.randint(1, 5,   (B,)),
        'item_id':          torch.randint(1, 500, (B,)),
        'item_category':    torch.randint(1, 50,  (B,)),
        'item_price_level': torch.randint(1, 5,   (B,)),
        'item_sales_level': torch.randint(1, 5,   (B,)),
        'ad_id':            torch.randint(1, 200, (B,)),
        'campaign_id':      torch.randint(1, 50,  (B,)),
        'customer_id':      torch.randint(1, 30,  (B,)),
        'brand_id':         torch.randint(1, 80,  (B,)),
        'pid':              torch.randint(1, 3,   (B,)),
        'hour':             torch.randint(0, 24,  (B,)),
        'seq_item_ids':     torch.randint(0, 500, (B, L)),
        'seq_categories':   torch.randint(0, 50,  (B, L)),
        'seq_mask':         seq_mask,
    }


# ESMM tests

class TestESMM:

    def test_output_shapes(self):
        from models.esmm import ESMM
        model = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        batch = make_batch()
        p_ctr, p_cvr, p_ctcvr = model(batch)
        assert p_ctr.shape   == (B,), f'p_ctr shape: {p_ctr.shape}'
        assert p_cvr.shape   == (B,)
        assert p_ctcvr.shape == (B,)

    def test_probabilities_in_range(self):
        from models.esmm import ESMM
        model = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        model.eval()
        batch = make_batch()
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = model(batch)
        for name, t in [('p_ctr', p_ctr), ('p_cvr', p_cvr), ('p_ctcvr', p_ctcvr)]:
            assert (t > 0).all() and (t < 1).all(), \
                f'{name} has values outside (0,1): min={t.min():.4f} max={t.max():.4f}'

    def test_ctcvr_equals_ctr_times_cvr(self):
        # pCTCVR must exactly equal pCTR × pCVR (ESMM constraint).
        from models.esmm import ESMM
        model = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        model.eval()
        batch = make_batch()
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = model(batch)
        expected = p_ctr * p_cvr
        assert torch.allclose(p_ctcvr, expected, atol=1e-6), \
            'pCTCVR != pCTR × pCVR — ESMM decomposition violated'

    def test_no_nan_forward(self):
        from models.esmm import ESMM
        model = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        batch = make_batch()
        p_ctr, p_cvr, p_ctcvr = model(batch)
        for name, t in [('p_ctr', p_ctr), ('p_cvr', p_cvr), ('p_ctcvr', p_ctcvr)]:
            assert not torch.isnan(t).any(), f'NaN in {name}'

    def test_loss_computation(self):
        from models.esmm import ESMM
        model  = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        batch  = make_batch()
        click  = torch.randint(0, 2, (B,)).float()
        purchase = (click * torch.randint(0, 2, (B,)).float())
        p_ctr, p_cvr, p_ctcvr = model(batch)
        loss, l_ctr, l_ctcvr  = model.compute_loss(p_ctr, p_ctcvr, click, purchase)
        assert loss.item() > 0, 'Loss must be positive'
        assert not torch.isnan(loss), 'Loss is NaN'
        assert abs(loss.item() - (l_ctr.item() + l_ctcvr.item())) < 1e-5, \
            'Total loss != L_CTR + L_CTCVR'

    def test_no_nan_gradients(self):
        from models.esmm import ESMM
        model    = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        batch    = make_batch()
        click    = torch.randint(0, 2, (B,)).float()
        purchase = click * torch.randint(0, 2, (B,)).float()
        p_ctr, p_cvr, p_ctcvr = model(batch)
        loss, *_ = model.compute_loss(p_ctr, p_ctcvr, click, purchase)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f'NaN gradient in {name}'

    def test_all_padding_sequence_no_nan(self):
        #Model must not produce NaN when the entire sequence is padding.
        from models.esmm import ESMM
        model = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32])
        model.eval()
        batch = make_batch()
        batch['seq_mask']     = torch.zeros(B, L, dtype=torch.bool)
        batch['seq_item_ids'] = torch.zeros(B, L, dtype=torch.long)
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = model(batch)
        assert not torch.isnan(p_cvr).any(),   'NaN in p_cvr with all-padding seq'
        assert not torch.isnan(p_ctcvr).any(), 'NaN in p_ctcvr with all-padding seq'

    def test_no_ctr_tower_ablation(self):
        from models.esmm import ESMM
        model = ESMM(VOCAB, embed_dim=D, hidden_dims=[64, 32], use_ctr_tower=False)
        model.eval()
        batch = make_batch()
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = model(batch)
        assert p_ctr.shape == (B,) and not torch.isnan(p_ctr).any()


# Layer tests

class TestGSU:

    def test_hard_retrieves_category_matches(self):
        from models.layers import GSU
        gsu = GSU(mode='hard', embed_dim=D)
        Bt, Lt = 4, 30
        target_emb = torch.randn(Bt, D)
        target_cat = torch.ones(Bt, dtype=torch.long)   # all category 1
        seq_emb    = torch.randn(Bt, Lt, D)
        seq_cat    = torch.zeros(Bt, Lt, dtype=torch.long)
        seq_cat[:, :8] = 1                              # first 8 match
        seq_mask   = torch.ones(Bt, Lt, dtype=torch.bool)

        retrieved, mask = gsu(target_emb, target_cat, seq_emb, seq_cat, seq_mask, top_k=5)
        assert retrieved.shape == (Bt, 5, D)
        # All retrieved items should have score > -1e8 (valid category match)
        assert mask.sum() > 0

    def test_output_k_equals_min_top_k_L(self):
        from models.layers import GSU
        gsu = GSU(mode='soft', embed_dim=D)
        Bt, Lt = 4, 10
        batch = {
            'target_emb':     torch.randn(Bt, D),
            'target_category':torch.zeros(Bt, dtype=torch.long),
            'seq_emb':        torch.randn(Bt, Lt, D),
            'seq_category':   torch.zeros(Bt, Lt, dtype=torch.long),
            'seq_mask':       torch.ones(Bt, Lt, dtype=torch.bool),
        }
        retrieved, mask = gsu(**batch, top_k=20)  # top_k > L → returns L
        assert retrieved.shape == (Bt, Lt, D)


class TestTargetAttention:

    def test_output_shape(self):
        from models.layers import TargetAttention
        attn   = TargetAttention(embed_dim=D)
        target = torch.randn(B, D)
        seq    = torch.randn(B, L, D)
        mask   = torch.ones(B, L, dtype=torch.bool)
        out    = attn(target, seq, mask)
        assert out.shape == (B, D)

    def test_all_padding_returns_no_nan(self):
        from models.layers import TargetAttention
        attn   = TargetAttention(embed_dim=D)
        target = torch.randn(B, D)
        seq    = torch.randn(B, L, D)
        mask   = torch.zeros(B, L, dtype=torch.bool)   # all padding
        out    = attn(target, seq, mask)
        assert not torch.isnan(out).any(), 'NaN with all-padding mask'


# Metrics tests

class TestMetrics:

    def test_auc_ctr_perfect(self):
        from utils.metrics import compute_esmm_metrics
        N = 100
        click    = np.array([0]*50 + [1]*50)
        purchase = click * np.array([0]*50 + [1]*50)
        p_ctr    = np.array([0.1]*50 + [0.9]*50)
        p_cvr    = np.array([0.1]*50 + [0.9]*50)
        p_ctcvr  = p_ctr * p_cvr
        m = compute_esmm_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
        assert m['auc_ctr'] == 1.0, f"Expected AUC_CTR=1.0, got {m['auc_ctr']}"

    def test_cost_ratio_overdelivery(self):
        from utils.metrics import compute_esmm_metrics
        N = 1000
        purchase = np.zeros(N); purchase[:10] = 1    # 1% actual CVR
        p_cvr    = np.full(N, 0.02)                  # model says 2% → overpredicts
        p_ctr    = np.full(N, 0.5)
        p_ctcvr  = p_ctr * p_cvr
        click    = np.zeros(N); click[:100] = 1
        m = compute_esmm_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
        # cost_ratio = realized / predicted = 0.01 / 0.02 = 0.5 → underdelivery
        assert m['cost_status'] == 'underdelivery', \
            f"Expected underdelivery, got {m['cost_status']} (ratio={m['cost_ratio']:.3f})"

    def test_healthy_cost_ratio(self):
        from utils.metrics import compute_esmm_metrics
        N = 1000
        purchase = np.zeros(N); purchase[:10] = 1    # 1% actual CVR
        p_cvr    = np.full(N, 0.01)                  # model exactly matches
        p_ctr    = np.full(N, 0.5)
        p_ctcvr  = p_ctr * p_cvr
        click    = np.zeros(N); click[:50] = 1
        m = compute_esmm_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase)
        assert m['cost_status'] == 'healthy', \
            f"Expected healthy, got {m['cost_status']} (ratio={m['cost_ratio']:.3f})"


# Calibration tests 

class TestCostCalibrator:

    def test_fit_transform_reduces_ece(self):
        from utils.calibration.cost_calibrator import CostCalibrator
        rng = np.random.default_rng(42)
        N   = 5000
        # Miscalibrated model: predicts 2× the true probability
        true_cvr   = 0.05
        raw_scores = np.clip(rng.uniform(0, 0.2, N), 0, 1)
        labels     = (rng.uniform(size=N) < true_cvr).astype(float)

        cal = CostCalibrator()
        cal.fit(raw_scores, labels)
        calibrated = cal.transform(raw_scores)

        assert cal.fit_stats['calibrated_ece'] <= cal.fit_stats['raw_ece'], \
            'Calibration should reduce ECE'

    def test_transform_preserves_rank_order(self):
        # Isotonic regression must be monotone: AUC unchanged after calibration.
        from utils.calibration.cost_calibrator import CostCalibrator
        from sklearn.metrics import roc_auc_score
        rng    = np.random.default_rng(0)
        N      = 2000
        scores = rng.uniform(size=N)
        labels = (rng.uniform(size=N) < scores * 0.3).astype(float)

        cal = CostCalibrator()
        cal.fit(scores, labels)
        calibrated = cal.transform(scores)

        auc_raw = roc_auc_score(labels, scores)
        auc_cal = roc_auc_score(labels, calibrated)
        assert abs(auc_raw - auc_cal) < 1e-6, \
            f"AUC changed after calibration: {auc_raw:.4f} → {auc_cal:.4f}"


class TestCostMonitor:

    def test_healthy_detection(self):
        from utils.calibration.cost_calibrator import CostMonitor
        m       = CostMonitor()
        p_cvr   = np.full(1000, 0.05)
        purchase = np.zeros(1000); purchase[:50] = 1   # realized CVR = 5%
        r = m.compute(p_cvr, purchase)
        assert r['status'] == 'healthy', f"Expected healthy, got {r['status']}"
        assert abs(r["global_cost_ratio"] - 1.0) < 0.05

    def test_underdelivery_detection(self):
        from utils.calibration.cost_calibrator import CostMonitor
        m       = CostMonitor()
        p_cvr   = np.full(1000, 0.15)   # model predicts 15%
        purchase = np.zeros(1000); purchase[:10] = 1   # actual 1%
        r = m.compute(p_cvr, purchase)
        assert r['status'] == 'underdelivery', \
            f"Expected underdelivery, got {r['status']} (ratio={r['global_cost_ratio']:.3f})"
