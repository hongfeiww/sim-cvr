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

FEAT_ID_MAP: Dict[str, str] = {
    '101': 'user_id',
    '121': 'age_level',
    '122': 'gender',
    '124': 'shopping_level',
    '125': 'city_level',
    '205': 'item_id',
    '206': 'item_category',
    '207': 'campaign_id',
    '210': 'item_price_level',
    '216': 'item_sales_level',
    '508': 'ad_id',
    '853': 'customer_id',
    '301': 'pid',
    '109_14': 'seq_categories_raw',
}

SCALAR_COLS = [
    'user_id', 'age_level', 'gender', 'shopping_level', 'city_level',
    'item_id', 'item_category', 'item_price_level', 'item_sales_level',
    'ad_id', 'campaign_id', 'customer_id', 'brand_id', 'pid', 'hour',
]

# Parsing helpers

def parse_features(feat_str: str) -> Dict[str, str]:
    result = {}
    if not feat_str or not isinstance(feat_str, str):
        return result
    for pair in feat_str.strip().split('\x02'):
        if '\x01' in pair:
            k, v = pair.split('\x01', 1)
            result[k.strip()] = v.strip()
    return result

# Step 1: index common_features into SQLite (streaming, no RAM accumulation)

def build_common_db(common_path: str, db_path: str) -> None:
    # Stream common_features_train.csv line-by-line into a SQLite table.
    logger.info(f'Building SQLite index from {common_path} ...')
    logger.info(f'  DB path: {db_path}')

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA cache_size=-65536')   # 64MB page cache
    conn.execute('''
        CREATE TABLE IF NOT EXISTS common_features (
            cf_index TEXT PRIMARY KEY,
            feat_str TEXT
        )
    ''')
    conn.commit()

    BATCH = 10_000
    buf   = []
    total = 0

    with open(common_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) < 2:
                continue
            buf.append((parts[0].strip(), parts[1]))
            if len(buf) >= BATCH:
                conn.executemany(
                    'INSERT OR REPLACE INTO common_features VALUES (?,?)', buf)
                conn.commit()
                total += len(buf)
                buf = []
                if total % 1_000_000 == 0:
                    logger.info(f'  Indexed {total:,} rows...')

    if buf:
        conn.executemany(
            'INSERT OR REPLACE INTO common_features VALUES (?,?)', buf)
        conn.commit()
        total += len(buf)

    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_cf ON common_features(cf_index)')
    conn.commit()
    conn.close()
    logger.info(f'  Done. {total:,} rows indexed.')

# Step 2: stream skeleton in chunks, lookup common via SQLite

def _parse_line_with_db(
    line: str,
    conn: sqlite3.Connection,
    cache: Dict[str, Dict[str, str]],
) -> dict:
    '''
    Parse one skeleton line and merge common features via SQLite lookup.
    Uses a small in-process LRU-style cache (dict) to avoid repeated
    DB hits for the same cf_index within a chunk.
    '''
    parts = line.strip().split(',')
    if len(parts) < 3:
        return {}
    try:
        click    = int(parts[1])
        purchase = int(parts[2])
        cf_index = parts[3].strip() if len(parts) > 3 else ''
        hour_raw = parts[4].strip() if len(parts) > 4 else '0'
        feat_str = parts[5]         if len(parts) > 5 else ''
    except (ValueError, IndexError):
        return {}

    feat = parse_features(feat_str)

    # Lookup common features (cache miss -> SQLite query)
    if cf_index:
        if cf_index not in cache:
            row = conn.execute(
                'SELECT feat_str FROM common_features WHERE cf_index=?',
                (cf_index,)
            ).fetchone()
            cache[cf_index] = parse_features(row[0]) if row else {}
            # Keep cache bounded to avoid unbounded growth
            if len(cache) > 50_000:
                # Evict oldest half
                keys = list(cache.keys())
                for k in keys[:25_000]:
                    del cache[k]

        for k, v in cache[cf_index].items():
            if k not in feat:
                feat[k] = v

    row: Dict = {'click': click, 'purchase': purchase}
    for fid, value in feat.items():
        if fid in FEAT_ID_MAP:
            row[FEAT_ID_MAP[fid]] = value
    try:
        row['hour'] = int(hour_raw) % 24
    except ValueError:
        row['hour'] = 0
    return row


def _fill_and_extract(df: pd.DataFrame, max_seq_len: int) -> pd.DataFrame:
    for col in SCALAR_COLS:
        if col not in df.columns:
            df[col] = 1
        else:
            df[col] = pd.to_numeric(
                df[col], errors='coerce').fillna(1).astype(int)
    if 'brand_id' not in df.columns or (df['brand_id'] == 1).all():
        df['brand_id'] = df.get(
            'customer_id', pd.Series([1] * len(df), index=df.index))

    seq_items_out, seq_cats_out = [], []
    raw_col = df.get(
        'seq_categories_raw',
        pd.Series([''] * len(df), index=df.index))
    for raw in raw_col:
        if raw and isinstance(raw, str):
            cats = [int(x) for x in raw.strip().split() if x.isdigit()]
        else:
            cats = []
        cats = cats[-max_seq_len:]
        seq_cats_out.append(cats)
        seq_items_out.append(cats)
    df['seq_items'] = seq_items_out
    df['seq_cats']  = seq_cats_out
    if 'seq_categories_raw' in df.columns:
        df = df.drop(columns=['seq_categories_raw'])
    return df


def stream_to_temp_parquets(
    skeleton_path: str,
    db_path:       str,
    tmp_dir:       str,
    chunk_size:    int,
    max_seq_len:   int,
    subsample:     float,
    seed:          int,
) -> List[str]:
    logger.info(f'Pass 2: streaming skeleton -> temp parquets '
                f'(chunk={chunk_size:,})...')
    rng       = np.random.default_rng(seed)
    conn      = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute('PRAGMA cache_size=-32768')   # 32MB page cache
    cf_cache: Dict[str, Dict[str, str]] = {}

    buf:       List = []
    tmp_files: List[str] = []
    total = 0

    def flush(buf, ci):
        df    = pd.DataFrame(buf)
        df    = _fill_and_extract(df, max_seq_len)
        fpath = os.path.join(tmp_dir, f'chunk_{ci:05d}.parquet')
        df.to_parquet(fpath, index=False)
        return fpath

    with open(skeleton_path, encoding='utf-8') as f:
        for line in f:
            if subsample < 1.0 and rng.random() > subsample:
                continue
            row = _parse_line_with_db(line, conn, cf_cache)
            if not row:
                continue
            buf.append(row)
            if len(buf) >= chunk_size:
                fpath = flush(buf, len(tmp_files))
                tmp_files.append(fpath)
                total += len(buf)
                logger.info(f'  Chunk {len(tmp_files):3d} written '
                            f'({total:,} rows total)')
                buf = []
                cf_cache.clear()
                gc.collect()

    if buf:
        fpath = flush(buf, len(tmp_files))
        tmp_files.append(fpath)
        total += len(buf)

    conn.close()
    logger.info(f'  Total: {total:,} rows in {len(tmp_files)} chunks')
    return tmp_files

# Pass 3: vocab (one column at a time)

def collect_vocab(
    tmp_files: List[str],
) -> Tuple[Dict[str, LabelEncoder], Dict[str, int]]:
    logger.info('Pass 3: collecting vocabulary...')
    encoders:    Dict[str, LabelEncoder] = {}
    vocab_sizes: Dict[str, int]          = {}
    for col in SCALAR_COLS:
        values: set = set()
        for fpath in tmp_files:
            chunk = pd.read_parquet(fpath, columns=[col])
            values.update(chunk[col].astype(str).unique())
            del chunk
        le = LabelEncoder()
        le.fit(sorted(values))
        encoders[col]    = le
        vocab_sizes[col] = len(le.classes_) + 1
        logger.info(f'  {col}: {len(le.classes_):,} values')
        gc.collect()
    return encoders, vocab_sizes

# Pass 4: cold-start counts

def compute_cold_start_sets(
    tmp_files: List[str], min_count: int
) -> Tuple[Set[str], Set[str]]:
    logger.info(f'Pass 4: cold-start counts (min={min_count})...')
    user_counts: Dict[str, int] = defaultdict(int)
    item_counts: Dict[str, int] = defaultdict(int)
    for fpath in tmp_files:
        chunk = pd.read_parquet(fpath, columns=['user_id', 'item_id'])
        for v in chunk['user_id'].astype(str):
            user_counts[v] += 1
        for v in chunk['item_id'].astype(str):
            item_counts[v] += 1
        del chunk
        gc.collect()
    valid_users = {u for u, c in user_counts.items() if c >= min_count}
    valid_items = {i for i, c in item_counts.items() if c >= min_count}
    logger.info(f'  Valid users: {len(valid_users):,} / {len(user_counts):,}')
    logger.info(f'  Valid items: {len(valid_items):,} / {len(item_counts):,}')
    return valid_users, valid_items

# Pass 5: encode + filter + write final parquets

def encode_and_split(
    tmp_files:   List[str],
    encoders:    Dict[str, LabelEncoder],
    valid_users: Set[str],
    valid_items: Set[str],
    output_dir:  str,
    train_r:     float,
    val_r:       float,
) -> None:
    logger.info('Pass 5: encoding and writing final splits...')
    paths   = {s: os.path.join(output_dir, f'{s}.parquet')
               for s in ['train', 'val', 'test']}
    writers = {s: None for s in ['train', 'val', 'test']}
    counts  = {s: 0    for s in ['train', 'val', 'test']}

    for fpath in tmp_files:
        chunk = pd.read_parquet(fpath)
        chunk = chunk[chunk['user_id'].astype(str).isin(valid_users)]
        chunk = chunk[chunk['item_id'].astype(str).isin(valid_items)]
        if chunk.empty:
            continue

        for col in SCALAR_COLS:
            le      = encoders[col]
            known   = set(le.classes_)
            default = le.classes_[0]
            vals    = chunk[col].astype(str).apply(
                lambda x: x if x in known else default)
            chunk[col] = le.transform(vals) + 1

        chunk = chunk.sort_values('hour').reset_index(drop=True)
        n  = len(chunk)
        t1 = int(n * train_r)
        t2 = int(n * (train_r + val_r))
        splits = {'train': chunk.iloc[:t1],
                  'val':   chunk.iloc[t1:t2],
                  'test':  chunk.iloc[t2:]}

        for split, sdf in splits.items():
            if sdf.empty:
                continue
            table = pa.Table.from_pandas(sdf, preserve_index=False)
            if writers[split] is None:
                writers[split] = pq.ParquetWriter(paths[split], table.schema)
            writers[split].write_table(table)
            counts[split] += len(sdf)

        del chunk
        gc.collect()

    for w in writers.values():
        if w is not None:
            w.close()
    for split, n in counts.items():
        logger.info(f'  {split.capitalize()}: {n:,} rows')

# Main
def preprocess(
    data_dir:    str,
    output_dir:  str,
    max_seq_len: int   = 50,
    min_count:   int   = 5,
    subsample:   float = 1.0,
    chunk_size:  int   = CHUNK_SIZE,
    seed:        int   = 42,
    train_r:     float = 0.7,
    val_r:       float = 0.1,
    keep_db:     bool  = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, 'train.parquet')):
        logger.info('Output already exists. Delete output_dir to reprocess.')
        return

    skeleton_path = os.path.join(data_dir, 'sample_skeleton_train.csv')
    common_path   = os.path.join(data_dir, 'common_features_train.csv')

    if not os.path.exists(skeleton_path):
        raise FileNotFoundError(
            f'Ali-CCP not found at {skeleton_path}.\n'
            'Download from https://tianchi.aliyun.com/dataset/408\n'
            'Or run: python data/generate_synthetic.py'
        )

    tmp_dir = tempfile.mkdtemp(prefix='aliccp_tmp_')
    db_path = os.path.join(tmp_dir, 'common_features.db')
    logger.info(f'Temp dir: {tmp_dir}')

    try:
        # Step 1: index common features into SQLite
        if os.path.exists(common_path):
            build_common_db(common_path, db_path)
        else:
            logger.warning('common_features not found, skipping.')
            # Create empty DB so subsequent code works
            sqlite3.connect(db_path).execute(
                'CREATE TABLE common_features '
                '(cf_index TEXT PRIMARY KEY, feat_str TEXT)'
            ).connection.commit()

        # Step 2: stream skeleton -> temp parquets
        tmp_files = stream_to_temp_parquets(
            skeleton_path, db_path, tmp_dir,
            chunk_size, max_seq_len, subsample, seed)

        # Step 3: vocab
        encoders, vocab_sizes = collect_vocab(tmp_files)

        # Step 4: cold-start
        valid_users, valid_items = compute_cold_start_sets(
            tmp_files, min_count)

        # Step 5: encode + split + write
        encode_and_split(tmp_files, encoders, valid_users, valid_items,
                         output_dir, train_r, val_r)

        with open(os.path.join(output_dir, 'vocab_sizes.json'), 'w') as f:
            json.dump(vocab_sizes, f, indent=2)
        logger.info(f'Saved to {output_dir}/')

    finally:
        if not keep_db:
            for fname in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, fname))
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

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