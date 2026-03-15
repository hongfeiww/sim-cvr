'''
Ali-CCP preprocessing -- zero large in-memory structures.
Root cause of OOM: common_features is 10GB, even the filtered subset
is too large to hold in RAM on a machine with limited memory.
Solution: pre-index common_features into a SQLite database (disk-based),
then do random-access lookups by cf_index during skeleton streaming
'''

import argparse
import gc
import json
import logging
import os
import sqlite3
import tempfile
from collections import defaultdict
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CHUNK_SIZE = 50_000

# field_id -> column name (scalar features we use)
FIELD_MAP = {
    "101": "user_id",
    "121": "user_class1",
    "122": "user_class2",
    "124": "gender",
    "125": "age_level",
    "126": "consumption_level",
    "129": "geo_class",
    "205": "item_id",
    "206": "item_category",
    "207": "shop_id",
    "216": "brand_id",
    "301": "pid",
}

# Multi-value sequence fields (field_id -> column name)
SEQ_FIELD_MAP = {
    "109_14": "seq_cate_raw",   # category behavior: "cate_id\x03count" pairs
}

# Final scalar columns passed to model (matches models/sim_cvr.py ALL_FIELDS)
SCALAR_COLS = [
    "user_id", "age_level", "gender", "user_class1", "user_class2",
    "item_id", "item_category", "consumption_level", "geo_class",
    "shop_id", "brand_id", "pid", "hour",
]

RENAME_MAP = {
    "user_class1":       "shopping_level",
    "user_class2":       "city_level",
    "consumption_level": "item_price_level",
    "geo_class":         "item_sales_level",
    "shop_id":           "campaign_id",
    "brand_id":          "customer_id",
    "pid":               "pid",
}

MODEL_SCALAR_COLS = [
    "user_id", "age_level", "gender", "shopping_level", "city_level",
    "item_id", "item_category", "item_price_level", "item_sales_level",
    "ad_id", "campaign_id", "customer_id", "brand_id", "pid", "hour",
]

# Feature string parser

def parse_feature_list(feat_str: str) -> Tuple[Dict[str, str], Dict[str, List]]:
    """
    Parse feature_list string into scalar and sequence features.

    Format:
      features separated by \x01
      each feature: field_id \x02 feature_id \x03 feature_value

    Returns:
      scalars: {field_id: feature_id}  (take feature_id as the value)
      seqs:    {field_id: [(feature_id, feature_value), ...]}
    """
    scalars: Dict[str, str]       = {}
    seqs:    Dict[str, List]      = {}

    if not feat_str or not isinstance(feat_str, str):
        return scalars, seqs

    for feat in feat_str.split("\x01"):
        feat = feat.strip()
        if not feat:
            continue
        parts = feat.split("\x02")
        if len(parts) < 2:
            continue
        field_id = parts[0].strip()
        rest     = parts[1]

        if "\x03" in rest:
            feature_id, feature_value = rest.split("\x03", 1)
        else:
            feature_id    = rest
            feature_value = "1"

        feature_id    = feature_id.strip()
        feature_value = feature_value.strip()

        # Sequence fields (multi-value, same field_id appears multiple times)
        if field_id in SEQ_FIELD_MAP:
            if field_id not in seqs:
                seqs[field_id] = []
            seqs[field_id].append(feature_id)   # use feature_id as category id
        else:
            # Scalar: use feature_id as the encoded value
            scalars[field_id] = feature_id

    return scalars, seqs

# Step 1: index common_features into SQLite
def build_common_db(common_path: str, db_path: str) -> None:
    logger.info(f"Indexing common features into SQLite: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("CREATE TABLE IF NOT EXISTS cf "
                 "(idx TEXT PRIMARY KEY, feat_str TEXT)")
    conn.commit()

    BATCH = 10_000
    buf   = []
    total = 0

    with open(common_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: common_feature_index, feature_num, feature_list
            parts = line.split(",", 2)
            if len(parts) < 3:
                # fallback: index, feature_list (no feature_num)
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue
                idx      = parts[0].strip()
                feat_str = parts[1]
            else:
                idx      = parts[0].strip()
                feat_str = parts[2]   # skip feature_num

            buf.append((idx, feat_str))
            if len(buf) >= BATCH:
                conn.executemany("INSERT OR REPLACE INTO cf VALUES (?,?)", buf)
                conn.commit()
                total += len(buf)
                buf = []
                if total % 1_000_000 == 0:
                    logger.info(f"  {total:,} rows indexed")

    if buf:
        conn.executemany("INSERT OR REPLACE INTO cf VALUES (?,?)", buf)
        conn.commit()
        total += len(buf)

    conn.execute("CREATE INDEX IF NOT EXISTS idx ON cf(idx)")
    conn.commit()
    conn.close()
    logger.info(f"  Done: {total:,} common feature rows indexed")

# Step 2: stream skeleton in chunks, lookup common via SQLite

def parse_skeleton_line(line: str, conn: sqlite3.Connection,
                        cache: Dict, max_seq_len: int) -> dict:
    """
    Format: sample_id, click, purchase, common_feature_index,
            feature_num, feature_list
    """
    line = line.strip()
    if not line:
        return {}

    parts = line.split(",", 5)
    if len(parts) < 3:
        return {}

    try:
        click    = int(float(parts[1]))
        purchase = int(float(parts[2]))
    except (ValueError, IndexError):
        return {}

    cf_idx   = parts[3].strip() if len(parts) > 3 else ""
    feat_str = parts[5]         if len(parts) > 5 else ""

    # Parse sample-level features
    scalars, seqs = parse_feature_list(feat_str)

    # Lookup and merge common (user-level) features
    if cf_idx:
        if cf_idx not in cache:
            row = conn.execute(
                "SELECT feat_str FROM cf WHERE idx=?", (cf_idx,)
            ).fetchone()
            if row:
                cs, cseq = parse_feature_list(row[0])
                cache[cf_idx] = (cs, cseq)
            else:
                cache[cf_idx] = ({}, {})
            if len(cache) > 30_000:
                keys = list(cache.keys())
                for k in keys[:15_000]:
                    del cache[k]

        cs, cseq = cache[cf_idx]
        for k, v in cs.items():
            if k not in scalars:
                scalars[k] = v
        for k, v in cseq.items():
            if k not in seqs:
                seqs[k] = v

    # Build output row
    row: Dict = {"click": click, "purchase": purchase, "hour": 0,
                 "ad_id": "1"}   # ad_id not directly in Ali-CCP

    for field_id, val in scalars.items():
        if field_id in FIELD_MAP:
            row[FIELD_MAP[field_id]] = val

    # Extract behavior sequence from 109_14 (category behavior)
    if "109_14" in seqs:
        cats = seqs["109_14"][:max_seq_len]
        row["seq_items"] = cats
        row["seq_cats"]  = cats
    else:
        row["seq_items"] = []
        row["seq_cats"]  = []

    return row

def stream_to_parquets(skeleton_path, db_path, tmp_dir,
                       chunk_size, subsample, seed, max_seq_len):
    logger.info(f"Streaming skeleton (chunk={chunk_size:,}, "
                f"subsample={subsample})...")
    rng   = np.random.default_rng(seed)
    conn  = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size=-32768")
    cache: Dict = {}

    buf: List      = []
    tmp_files: List = []
    total = 0

    def flush(buf, ci):
        df = pd.DataFrame(buf)
        # Fill missing scalar cols
        for col in list(FIELD_MAP.values()) + ["ad_id", "hour",
                                               "seq_items", "seq_cats"]:
            if col not in df.columns:
                df[col] = "1" if col not in ("seq_items","seq_cats") else df.get(col, [[]]*len(df))
        fpath = os.path.join(tmp_dir, f"chunk_{ci:05d}.parquet")
        df.to_parquet(fpath, index=False)
        return fpath

    with open(skeleton_path, encoding="utf-8") as f:
        for line in f:
            if subsample < 1.0 and rng.random() > subsample:
                continue
            row = parse_skeleton_line(line, conn, cache, max_seq_len)
            if not row:
                continue
            buf.append(row)
            if len(buf) >= chunk_size:
                fpath = flush(buf, len(tmp_files))
                tmp_files.append(fpath)
                total += len(buf)
                logger.info(f"  Chunk {len(tmp_files):3d} | {total:,} rows")
                buf = []
                gc.collect()

    if buf:
        fpath = flush(buf, len(tmp_files))
        tmp_files.append(fpath)
        total += len(buf)

    conn.close()
    logger.info(f"  Total: {total:,} rows in {len(tmp_files)} chunks")
    return tmp_files

# Pass 3: vocab (one column at a time)
def collect_vocab(tmp_files, model_cols):
    logger.info("Collecting vocabulary...")
    encoders    = {}
    vocab_sizes = {}
    for col in model_cols:
        values = set()
        for fpath in tmp_files:
            chunk = pd.read_parquet(fpath, columns=[col] if col in
                      pd.read_parquet(fpath, columns=[model_cols[0]]).columns
                      or True else [model_cols[0]])
            try:
                chunk = pd.read_parquet(fpath, columns=[col])
                values.update(chunk[col].astype(str).unique())
            except Exception:
                values.add("1")
            del chunk
        le = LabelEncoder()
        le.fit(sorted(values))
        encoders[col]    = le
        vocab_sizes[col] = len(le.classes_) + 1
        logger.info(f"  {col}: {len(le.classes_):,} unique values")
        gc.collect()
    return encoders, vocab_sizes

# Pass 4: cold-start counts

def cold_start_filter(tmp_files, min_count):
    logger.info(f"Cold-start counts (min={min_count})...")
    uc = defaultdict(int)
    ic = defaultdict(int)
    for fpath in tmp_files:
        chunk = pd.read_parquet(fpath, columns=["user_id","item_id"])
        for v in chunk["user_id"].astype(str): uc[v] += 1
        for v in chunk["item_id"].astype(str):  ic[v] += 1
        del chunk; gc.collect()
    vu = {u for u,c in uc.items() if c >= min_count}
    vi = {i for i,c in ic.items() if c >= min_count}
    logger.info(f"  Valid users: {len(vu):,}  Valid items: {len(vi):,}")
    return vu, vi

# Pass 5: encode + filter + write final parquets

def encode_and_split(tmp_files, encoders, valid_users, valid_items,
                     output_dir, train_r, val_r, model_cols):
    logger.info("Encoding and writing final splits...")
    paths   = {s: os.path.join(output_dir, f"{s}.parquet")
               for s in ["train","val","test"]}
    writers = {s: None for s in ["train","val","test"]}
    counts  = {s: 0    for s in ["train","val","test"]}

    for fpath in tmp_files:
        chunk = pd.read_parquet(fpath)

        # Rename columns to match model
        chunk = chunk.rename(columns=RENAME_MAP)

        # Add missing model cols
        for col in model_cols:
            if col not in chunk.columns:
                chunk[col] = "1"

        # Cold-start filter
        chunk = chunk[chunk["user_id"].astype(str).isin(valid_users)]
        chunk = chunk[chunk["item_id"].astype(str).isin(valid_items)]
        if chunk.empty:
            continue

        # Encode
        for col in model_cols:
            le    = encoders[col]
            known = set(le.classes_)
            dflt  = le.classes_[0]
            vals  = chunk[col].astype(str).apply(
                lambda x: x if x in known else dflt)
            chunk[col] = le.transform(vals) + 1

        chunk = chunk.sort_values("hour").reset_index(drop=True)
        n  = len(chunk)
        t1 = int(n * train_r)
        t2 = int(n * (train_r + val_r))
        splits = {"train": chunk.iloc[:t1],
                  "val":   chunk.iloc[t1:t2],
                  "test":  chunk.iloc[t2:]}

        for split, sdf in splits.items():
            if sdf.empty: continue
            table = pa.Table.from_pandas(sdf, preserve_index=False)
            if writers[split] is None:
                writers[split] = pq.ParquetWriter(paths[split], table.schema)
            writers[split].write_table(table)
            counts[split] += len(sdf)

        del chunk; gc.collect()

    for w in writers.values():
        if w: w.close()
    for s, n in counts.items():
        logger.info(f"  {s.capitalize()}: {n:,} rows")


# Main

def preprocess(data_dir, output_dir, max_seq_len=50, min_count=5,
               subsample=1.0, chunk_size=CHUNK_SIZE, seed=42,
               train_r=0.7, val_r=0.1, keep_db=False):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "train.parquet")):
        logger.info("Output exists. Delete output_dir to reprocess.")
        return

    skeleton = os.path.join(data_dir, "sample_skeleton_train.csv")
    common   = os.path.join(data_dir, "common_features_train.csv")

    if not os.path.exists(skeleton):
        raise FileNotFoundError(
            f"{skeleton} not found.\n"
            "Download from https://tianchi.aliyun.com/dataset/408\n"
            "Or run: python data/generate_synthetic.py")

    tmp_dir = tempfile.mkdtemp(prefix="aliccp_")
    db_path = os.path.join(tmp_dir, "cf.db")

    try:
        if os.path.exists(common):
            build_common_db(common, db_path)
        else:
            sqlite3.connect(db_path).execute(
                "CREATE TABLE cf (idx TEXT PRIMARY KEY, feat_str TEXT)"
            ).connection.commit()

        tmp_files = stream_to_parquets(
            skeleton, db_path, tmp_dir,
            chunk_size, subsample, seed, max_seq_len)

        encoders, vocab_sizes = collect_vocab(tmp_files, SCALAR_COLS)
        valid_users, valid_items = cold_start_filter(tmp_files, min_count)
        encode_and_split(tmp_files, encoders, valid_users, valid_items,
                         output_dir, train_r, val_r, MODEL_SCALAR_COLS)

        with open(os.path.join(output_dir, 'vocab_sizes.json'), 'w') as f:
            json.dump(vocab_sizes, f, indent=2)
        logger.info(f'Saved to {output_dir}/')

    finally:
        if not keep_db:
            for fname in os.listdir(tmp_dir):
                try: os.remove(os.path.join(tmp_dir, fname))
                except: pass
            try: os.rmdir(tmp_dir)
            except: pass

    logger.info('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='data/raw/aliccp')
    p.add_argument('--output_dir', default='data/processed')
    p.add_argument('--seq_len',    type=int,   default=50)
    p.add_argument('--min_count',  type=int,   default=5)
    p.add_argument('--subsample',  type=float, default=1.0)
    p.add_argument('--chunk_size', type=int,   default=CHUNK_SIZE)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--keep_db',    action='store_true',
                   help='Keep SQLite DB after processing (for debugging)')
    args = p.parse_args()
    preprocess(args.data_dir, args.output_dir, args.seq_len,
               args.min_count, args.subsample, args.chunk_size,
               args.seed, keep_db=args.keep_db)