'''
PyTorch Dataset for ESMM on Ali-CCP.

Each sample contains:
  click    — CTR label, ALL impressions
  purchase — purchase label, ALL impressions
  ctcvr    — CTCVR label, click * purchase

ESMM loss: L = w_ctr * BCE(pCTR, click) + BCE(pCTR * pCVR, ctcvr)

'''

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CVRDataset(Dataset):
    SCALAR_USER = ['user_id', 'age_level', 'gender', 'shopping_level', 'city_level']
    SCALAR_ITEM = ['item_id', 'item_category', 'item_price_level', 'item_sales_level']
    SCALAR_AD   = ['ad_id', 'campaign_id', 'customer_id', 'brand_id']
    SCALAR_CTX  = ['pid', 'hour']
    ALL_SCALARS = SCALAR_USER + SCALAR_ITEM + SCALAR_AD + SCALAR_CTX

    def __init__(self, parquet_path: str, max_seq_len: int = 50,
                 subset: Optional[float] = None):
        self.max_seq_len = max_seq_len
        self.df = pd.read_parquet(parquet_path)

        if subset is not None:
            n = int(len(self.df) * subset)
            self.df = self.df.sample(n=n, random_state=38).reset_index(drop=True)

        for col in self.ALL_SCALARS:
            if col not in self.df.columns:
                self.df[col] = 1
        if 'click'    not in self.df.columns: self.df['click']    = 0
        if 'purchase' not in self.df.columns: self.df['purchase'] = 0

        # ctcvr = click & purchase  
        self.df['ctcvr'] = (self.df['click'].astype(int) *
                            self.df['purchase'].astype(int)).astype(float)

        seq_items_raw = (self.df['seq_items'].tolist()
                         if 'seq_items' in self.df.columns
                         else [[] for _ in range(len(self.df))])
        seq_cats_raw  = (self.df['seq_cats'].tolist()
                         if 'seq_cats' in self.df.columns
                         else [[] for _ in range(len(self.df))])

        self.seq_items = self._pad(seq_items_raw)
        self.seq_cats  = self._pad(seq_cats_raw)
        self.seq_mask  = (self.seq_items != 0)

    def _pad(self, seqs: List[List[int]]) -> np.ndarray:
        '''Right-align: most recent item at position [-1]; pad with 0 on left.'''
        out = np.zeros((len(seqs), self.max_seq_len), dtype=np.int64)
        for i, seq in enumerate(seqs):
            seq = [int(x) for x in seq][-self.max_seq_len:]
            if seq:
                out[i, -len(seq):] = seq
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample = {col: torch.tensor(int(row[col]), dtype=torch.long)
                  for col in self.ALL_SCALARS}
        sample['seq_item_ids']   = torch.tensor(self.seq_items[idx], dtype=torch.long)
        sample['seq_categories'] = torch.tensor(self.seq_cats[idx],  dtype=torch.long)
        sample['seq_mask']       = torch.tensor(self.seq_mask[idx],  dtype=torch.bool)
        sample['click']          = torch.tensor(float(row['click']),    dtype=torch.float)
        sample['purchase']       = torch.tensor(float(row['purchase']), dtype=torch.float)
        sample['ctcvr']          = torch.tensor(float(row['ctcvr']),    dtype=torch.float)
        return sample

    def label_stats(self) -> Dict[str, float]:
        click_rate = float(self.df['click'].mean())
        ctcvr_rate = float(self.df['ctcvr'].mean())
        cvr_rate   = (float(self.df.loc[self.df['click'] == 1, 'purchase'].mean())
                      if self.df['click'].sum() > 0 else 0.0)
        return {'ctr': round(click_rate,5), 'ctcvr': round(ctcvr_rate,5),
                'cvr': round(cvr_rate,5)}


def build_dataloaders(
    data_dir: str,
    batch_size: int = 4096,
    max_seq_len: int = 50,
    num_workers: int = 4,
    subset: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    train_ds = CVRDataset(os.path.join(data_dir,'train.parquet'), max_seq_len, subset)
    val_ds   = CVRDataset(os.path.join(data_dir,'val.parquet'),   max_seq_len)
    test_ds  = CVRDataset(os.path.join(data_dir,'test.parquet'),  max_seq_len)

    kw = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size,     shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2, shuffle=False, **kw)

    with open(os.path.join(data_dir, 'vocab_sizes.json')) as f:
        vocab_sizes = json.load(f)

    stats = train_ds.label_stats()
    meta  = {'vocab_sizes': vocab_sizes, 'train_ctr': stats['ctr'],
             'train_cvr': stats['cvr'],  'train_ctcvr': stats['ctcvr']}
    return train_loader, val_loader, test_loader, meta
