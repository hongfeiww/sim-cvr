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
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


SCALAR_COLS = [
    "user_id", "age_level", "gender", "shopping_level", "city_level",
    "item_id", "item_category", "item_price_level", "item_sales_level",
    "ad_id", "campaign_id", "customer_id", "brand_id", "pid", "hour",
]
LABEL_COLS  = ["click", "purchase"]
SEQ_COLS    = ["seq_items", "seq_cats"]
LOAD_COLS   = SCALAR_COLS + LABEL_COLS + SEQ_COLS


def _pad_seq(seq, max_len: int) -> List[int]:
    """Right-align sequence: most recent item at [-1], pad left with 0."""
    if not isinstance(seq, list):
        try:
            seq = list(seq)
        except Exception:
            seq = []
    seq = [int(x) for x in seq][-max_len:]
    return [0] * (max_len - len(seq)) + seq


def _row_to_sample(
    row,
    max_seq_len: int,
    scalar_cols: List[str],
) -> Dict[str, torch.Tensor]:
    """Convert one DataFrame row to a model-ready sample dict."""
    sample = {col: torch.tensor(int(row[col]), dtype=torch.long)
              for col in scalar_cols}

    seq_items = _pad_seq(row.get("seq_items", []), max_seq_len)
    seq_cats  = _pad_seq(row.get("seq_cats",  []), max_seq_len)

    sample["seq_item_ids"]   = torch.tensor(seq_items, dtype=torch.long)
    sample["seq_categories"] = torch.tensor(seq_cats,  dtype=torch.long)
    sample["seq_mask"]       = sample["seq_item_ids"] != 0

    sample["click"]    = torch.tensor(float(row.get("click",    0)), dtype=torch.float)
    sample["purchase"] = torch.tensor(float(row.get("purchase", 0)), dtype=torch.float)
    return sample


# ---------------------------------------------------------------------------
# Option A: CVRDataset -- loads full parquet into df, lazy pad per __getitem__
# Use when dataset fits in RAM after loading (e.g. synthetic 200k samples)
# ---------------------------------------------------------------------------

class CVRDataset(Dataset):
    '''
    Map-style dataset. Loads parquet once, pads sequences lazily.
    RAM: O(N * num_cols) -- one DataFrame row per sample, no pre-padded arrays.
    '''

    def __init__(self, parquet_path: str, max_seq_len: int = 50,
                 subset: Optional[float] = None):
        self.max_seq_len = max_seq_len

        # Load only needed columns -- avoids loading unused metadata columns
        existing = pd.read_parquet(parquet_path, columns=["user_id"]).columns
        cols_to_load = [c for c in LOAD_COLS
                        if c in pd.read_parquet(
                            parquet_path, columns=LOAD_COLS[:1]).columns
                        or True]  # read all, pandas ignores missing
        try:
            self.df = pd.read_parquet(parquet_path, columns=LOAD_COLS)
        except Exception:
            # Fallback: load all and keep only what we need
            self.df = pd.read_parquet(parquet_path)
            for col in SCALAR_COLS + LABEL_COLS:
                if col not in self.df.columns:
                    self.df[col] = 1

        if subset is not None:
            n = max(1, int(len(self.df) * subset))
            self.df = self.df.sample(n=n, random_state=42).reset_index(drop=True)

        # Fill missing scalar cols with default 1
        for col in SCALAR_COLS:
            if col not in self.df.columns:
                self.df[col] = 1

        # Convert scalar cols to int16/int32 to save RAM
        for col in SCALAR_COLS:
            self.df[col] = self.df[col].astype("int32")
        for col in LABEL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("float32")
            else:
                self.df[col] = 0.0

        self._has_seq = ("seq_items" in self.df.columns and
                         "seq_cats"  in self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample = {col: torch.tensor(int(row[col]), dtype=torch.long)
                  for col in SCALAR_COLS}

        if self._has_seq:
            seq_items = _pad_seq(row["seq_items"], self.max_seq_len)
            seq_cats  = _pad_seq(row["seq_cats"],  self.max_seq_len)
        else:
            seq_items = [0] * self.max_seq_len
            seq_cats  = [0] * self.max_seq_len

        sample["seq_item_ids"]   = torch.tensor(seq_items, dtype=torch.long)
        sample["seq_categories"] = torch.tensor(seq_cats,  dtype=torch.long)
        sample["seq_mask"]       = sample["seq_item_ids"] != 0
        sample["click"]          = torch.tensor(float(row["click"]),    dtype=torch.float)
        sample["purchase"]       = torch.tensor(float(row["purchase"]), dtype=torch.float)
        return sample

    def label_stats(self) -> Dict[str, float]:
        ctr  = float(self.df["click"].mean())
        ctcvr= float((self.df["click"] * self.df["purchase"]).mean())
        mask = self.df["click"] == 1
        cvr  = float(self.df.loc[mask, "purchase"].mean()) if mask.sum() > 0 else 0.0
        return {"ctr": round(ctr,5), "ctcvr": round(ctcvr,5), "cvr": round(cvr,5)}


# ---------------------------------------------------------------------------
# Option B: IterableParquetDataset -- truly streaming, O(batch) RAM
# Use when even loading the full parquet into df causes OOM
# ---------------------------------------------------------------------------

class IterableParquetDataset(IterableDataset):

    def __init__(self, parquet_path: str, max_seq_len: int = 50,
                 shuffle_buffer: int = 10_000):
        self.parquet_path   = parquet_path
        self.max_seq_len    = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        # Read schema only to check columns -- no data loaded
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(parquet_path).names
        self._has_seq = ("seq_items" in schema_cols and
                         "seq_cats"  in schema_cols)

    def _iter_row_groups(self) -> Iterator[pd.DataFrame]:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.parquet_path)
        for rg in range(pf.metadata.num_row_groups):
            cols = LOAD_COLS if self._has_seq else SCALAR_COLS + LABEL_COLS
            yield pf.read_row_group(rg, columns=cols).to_pandas()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf: List[Dict] = []
        rng = np.random.default_rng(42)

        for chunk in self._iter_row_groups():
            # Fill missing cols
            for col in SCALAR_COLS:
                if col not in chunk.columns:
                    chunk[col] = 1
            for col in LABEL_COLS:
                if col not in chunk.columns:
                    chunk[col] = 0.0

            for _, row in chunk.iterrows():
                sample = _row_to_sample(row, self.max_seq_len, SCALAR_COLS)
                buf.append(sample)
                if len(buf) >= self.shuffle_buffer:
                    rng.shuffle(buf)
                    yield from buf
                    buf = []

            del chunk

        if buf:
            rng.shuffle(buf)
            yield from buf


# ---------------------------------------------------------------------------
# build_dataloaders: auto-selects dataset class based on available RAM
# ---------------------------------------------------------------------------

def _estimate_parquet_rows(path: str) -> int:
    """Estimate row count without reading data."""
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except Exception:
        return 10_000_000   # assume large if unknown


def build_dataloaders(
    data_dir:    str,
    batch_size:  int            = 4096,
    max_seq_len: int            = 50,
    num_workers: int            = 0,      # 0 = main process (safer for IterableDataset)
    subset:      Optional[float] = None,
    streaming:   bool           = False,  # force streaming mode
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns (train_loader, val_loader, test_loader, meta_dict).

    Set streaming=True to use IterableParquetDataset (O(batch) RAM).
    Otherwise uses CVRDataset (full parquet in RAM, faster __getitem__).
    """
    paths = {s: os.path.join(data_dir, f"{s}.parquet")
             for s in ["train", "val", "test"]}

    if streaming:
        train_ds = IterableParquetDataset(
            paths["train"], max_seq_len, shuffle_buffer=batch_size * 4)
        val_ds   = IterableParquetDataset(paths["val"],  max_seq_len)
        test_ds  = IterableParquetDataset(paths["test"], max_seq_len)
        # IterableDataset: shuffle handled internally, num_workers=0 safest
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                                  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2,
                                  num_workers=0)
    else:
        train_ds = CVRDataset(paths["train"], max_seq_len, subset)
        val_ds   = CVRDataset(paths["val"],   max_seq_len)
        test_ds  = CVRDataset(paths["test"],  max_seq_len)
        kw = dict(num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  **kw)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                                  shuffle=False, **kw)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size * 2,
                                  shuffle=False, **kw)

    with open(os.path.join(data_dir, "vocab_sizes.json")) as f:
        vocab_sizes = json.load(f)

    meta: Dict = {"vocab_sizes": vocab_sizes}
    if not streaming and hasattr(train_ds, "label_stats"):
        stats = train_ds.label_stats()
        meta.update({"train_ctr": stats["ctr"], "train_cvr": stats["cvr"],
                     "train_ctcvr": stats["ctcvr"]})

    return train_loader, val_loader, test_loader, meta