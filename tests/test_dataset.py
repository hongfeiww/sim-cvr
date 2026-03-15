'''
Unit tests for CVR dataset — shape, NaN, label consistency, padding.
All tests use synthetic in-memory data (no files required).
'''

import numpy as np
import pandas as pd
import pytest
import torch


def make_parquet(tmp_path, n=200, seq_len=10) -> str:
    rng = np.random.default_rng(42)
    records = []
    for i in range(n):
        sl = int(rng.integers(0, seq_len))
        click    = int(rng.integers(0, 2))
        purchase = int(click * rng.integers(0, 2))   # purchase only if click
        records.append({
            'user_id':          int(rng.integers(1, 100)),
            'age_level':        int(rng.integers(1, 7)),
            'gender':           int(rng.integers(1, 3)),
            'shopping_level':   int(rng.integers(1, 3)),
            'city_level':       int(rng.integers(1, 5)),
            'item_id':          int(rng.integers(1, 500)),
            'item_category':    int(rng.integers(1, 50)),
            'item_price_level': int(rng.integers(1, 5)),
            'item_sales_level': int(rng.integers(1, 5)),
            'ad_id':            int(rng.integers(1, 200)),
            'campaign_id':      int(rng.integers(1, 50)),
            'customer_id':      int(rng.integers(1, 30)),
            'brand_id':         int(rng.integers(1, 80)),
            'pid':              int(rng.integers(1, 3)),
            'hour':             int(rng.integers(0, 24)),
            'click':            click,
            'purchase':         purchase,
            'seq_items':        rng.integers(1, 500, size=sl).tolist(),
            'seq_cats':         rng.integers(1, 50,  size=sl).tolist(),
        })
    import json, os
    df   = pd.DataFrame(records)
    path = str(tmp_path / 'train.parquet')
    df.to_parquet(path, index=False)
    # Write vocab_sizes.json
    vs = {c: 1001 for c in ['user_id','age_level','gender','shopping_level',
                              'city_level','item_id','item_category',
                              'item_price_level','item_sales_level','ad_id',
                              'campaign_id','customer_id','brand_id','pid','hour']}
    with open(str(tmp_path / 'vocab_sizes.json'), 'w') as f:
        json.dump(vs, f)
    return path


class TestCVRDataset:

    def test_output_shapes(self, tmp_path):
        from data.dataset import CVRDataset
        L   = 10
        path = make_parquet(tmp_path, n=50, seq_len=L)
        ds   = CVRDataset(path, max_seq_len=L)
        s    = ds[0]
        assert s['seq_item_ids'].shape   == (L,), f"seq shape wrong: {s['seq_item_ids'].shape}"
        assert s['seq_categories'].shape == (L,)
        assert s['seq_mask'].shape       == (L,)
        assert s['click'].ndim           == 0, 'click must be scalar'
        assert s['purchase'].ndim        == 0, 'purchase must be scalar'

    def test_labels_are_binary(self, tmp_path):
        from data.dataset import CVRDataset
        path = make_parquet(tmp_path, n=100)
        ds   = CVRDataset(path, max_seq_len=10)
        for i in range(len(ds)):
            s = ds[i]
            assert s['click'].item()    in (0.0, 1.0), 'click must be 0 or 1'
            assert s['purchase'].item() in (0.0, 1.0), 'purchase must be 0 or 1'

    def test_purchase_implies_click(self, tmp_path):
        '''purchase=1 should always imply click=1 (data invariant).'''
        from data.dataset import CVRDataset
        path = make_parquet(tmp_path, n=200)
        ds   = CVRDataset(path, max_seq_len=10)
        for i in range(len(ds)):
            s = ds[i]
            if s['purchase'].item() == 1.0:
                assert s['click'].item() == 1.0, \
                    f'Sample {i}: purchase=1 but click=0 (impossible)'

    def test_padding_positions_are_zero(self, tmp_path):
        from data.dataset import CVRDataset
        path = make_parquet(tmp_path, n=100, seq_len=5)
        ds   = CVRDataset(path, max_seq_len=10)
        for i in range(min(20, len(ds))):
            s    = ds[i]
            mask = s['seq_mask']            # True = valid
            seq  = s['seq_item_ids']
            assert (seq[~mask] == 0).all(), \
                f'Non-zero values in padding positions at sample {i}'

    def test_dataset_length(self, tmp_path):
        from data.dataset import CVRDataset
        path = make_parquet(tmp_path, n=77)
        ds   = CVRDataset(path, max_seq_len=10)
        assert len(ds) == 77

    def test_no_nan_in_scalar_features(self, tmp_path):
        from data.dataset import CVRDataset
        path = make_parquet(tmp_path, n=50)
        ds   = CVRDataset(path, max_seq_len=10)
        scalar_keys = ['user_id','item_id','item_category','ad_id','hour']
        for key in scalar_keys:
            vals = torch.stack([ds[i][key] for i in range(len(ds))]).float()
            assert not torch.isnan(vals).any(), f'NaN in {key}'

    def test_all_scalar_features_present(self, tmp_path):
        from data.dataset import CVRDataset
        path    = make_parquet(tmp_path, n=10)
        ds      = CVRDataset(path, max_seq_len=10)
        sample  = ds[0]
        required = [
            'user_id','age_level','gender','shopping_level','city_level',
            'item_id','item_category','item_price_level','item_sales_level',
            'ad_id','campaign_id','customer_id','brand_id','pid','hour',
            'seq_item_ids','seq_categories','seq_mask','click','purchase',
        ]
        for key in required:
            assert key in sample, f'Missing key: {key}'
