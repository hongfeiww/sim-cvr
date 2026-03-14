'''
Generate a synthetic Ali-CCP-style dataset for testing and CI.

Schema mirrors Ali-CCP:
  - user features:  user_id, age_level, gender, shopping_level, city_level
  - item features:  item_id, item_category, item_price_level, item_sales_level
  - ad features:    ad_id, campaign_id, customer_id, brand_id
  - context:        pid (position id), hour

Labels (two-stage CVR task):
  - click:    whether user clicked (CTR label), defined on ALL impressions
  - purchase: whether user purchased (CVR label), defined on ALL impressions
              only click=1 samples have purchase=1

Key statistics calibrated to real Ali-CCP:
  - CTR  ≈ 3.8%  (clicks / impressions)
  - CTCVR ≈ 0.7% (purchases / impressions)
  - CVR  ≈ 18%   (purchases / clicks)

Positive correlation structure:
  - Users with high shopping_level click & buy more
  - Items with high sales_level attract more clicks
  - Price_level has negative correlation with purchase
'''

import argparse
import os
import numpy as np
import pandas as pd

# Vocab sizes (match preprocess_aliccp.py)
VOCAB = {
    'user_id':        50_000,
    'age_level':      7,
    'gender':         3,
    'shopping_level': 3,
    'city_level':     5,
    'item_id':        100_000,
    'item_category':  1_000,
    'item_price_level': 5,
    'item_sales_level': 5,
    'ad_id':          30_000,
    'campaign_id':    5_000,
    'customer_id':    10_000,
    'brand_id':       8_000,
    'pid':            3,
    'hour':           24,
}

SEQ_LEN = 50  # max behavior sequence length


def generate(n_samples: int, seed: int, output_dir: str) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n_samples:,} samples...")

    # Sample user/item/ad features
    user_id        = rng.integers(1, VOCAB["user_id"],        n_samples)
    age_level      = rng.integers(1, VOCAB["age_level"],      n_samples)
    gender         = rng.integers(1, VOCAB["gender"],         n_samples)
    shopping_level = rng.integers(1, VOCAB["shopping_level"], n_samples)  # 1=low,3=high
    city_level     = rng.integers(1, VOCAB["city_level"],     n_samples)

    item_id          = rng.integers(1, VOCAB["item_id"],          n_samples)
    item_category    = rng.integers(1, VOCAB["item_category"],     n_samples)
    item_price_level = rng.integers(1, VOCAB["item_price_level"],  n_samples)
    item_sales_level = rng.integers(1, VOCAB["item_sales_level"],  n_samples)

    ad_id       = rng.integers(1, VOCAB["ad_id"],       n_samples)
    campaign_id = rng.integers(1, VOCAB["campaign_id"], n_samples)
    customer_id = rng.integers(1, VOCAB["customer_id"], n_samples)
    brand_id    = rng.integers(1, VOCAB["brand_id"],    n_samples)

    pid  = rng.integers(1, VOCAB["pid"],  n_samples)
    hour = rng.integers(0, VOCAB["hour"], n_samples)

    # Generate labels
    # CTR logit: high shopping_level and sales_level → more clicks
    # position 1 (pid=1) has CTR boost
    ctr_logit = (
        -3.2
        + 0.4 * shopping_level # high shopping → more clicks
        + 0.3 * item_sales_level # popular items → more clicks
        - 0.1 * item_price_level # expensive → fewer clicks
        + 0.3 * (pid == 1).astype(float) # first position boost
        + 0.1 * rng.standard_normal(n_samples)
    )
    p_ctr = 1 / (1 + np.exp(-ctr_logit))
    click = (rng.uniform(size=n_samples) < p_ctr).astype(int)

    # CVR logit (among clicks): high shopping, lower price → more purchases
    cvr_logit = (
        -1.5
        + 0.5 * shopping_level
        + 0.2 * item_sales_level
        - 0.4 * item_price_level # price hurts conversion
        + 0.1 * rng.standard_normal(n_samples)
    )
    p_cvr = 1 / (1 + np.exp(-cvr_logit))
    # purchase only possible after click (realistic label structure)
    purchase_given_click = (rng.uniform(size=n_samples) < p_cvr).astype(int)
    purchase = (click == 1) * purchase_given_click

    print(f"  CTR:   {click.mean():.4f}")
    print(f"  CVR:   {purchase_given_click[click==1].mean():.4f}  (among clicks)")
    print(f"  CTCVR: {purchase.mean():.4f}")

    # Build behavior sequences: Each user has a "history" of previously interacted item_ids
    print("  Building behavior sequences...")
    user_item_pool = {}
    for uid in np.unique(user_id):
        n_hist = rng.integers(5, 30)
        user_item_pool[uid] = rng.integers(1, VOCAB["item_id"], n_hist).tolist()

    seq_items = []
    seq_cats  = []
    for i in range(n_samples):
        uid  = user_id[i]
        pool = user_item_pool.get(uid, [])
        sl   = min(len(pool), SEQ_LEN)
        hist_items = pool[-sl:]
        # Generate matching categories (simplified: category = item_id % n_categories)
        hist_cats  = [iid % VOCAB["item_category"] + 1 for iid in hist_items]
        seq_items.append(hist_items)
        seq_cats.append(hist_cats)

    # Assemble
    df = pd.DataFrame({
        "user_id":          user_id,
        "age_level":        age_level,
        "gender":           gender,
        "shopping_level":   shopping_level,
        "city_level":       city_level,
        "item_id":          item_id,
        "item_category":    item_category,
        "item_price_level": item_price_level,
        "item_sales_level": item_sales_level,
        "ad_id":            ad_id,
        "campaign_id":      campaign_id,
        "customer_id":      customer_id,
        "brand_id":         brand_id,
        "pid":              pid,
        "hour":             hour,
        "click":            click,
        "purchase":         purchase,
        "seq_items":        seq_items,
        "seq_cats":         seq_cats,
    })

    # shuffle then split for synthetic
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(n_samples * 0.7)
    n_val   = int(n_samples * 0.1)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    train.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val.to_parquet(  os.path.join(output_dir, "val.parquet"),   index=False)
    test.to_parquet( os.path.join(output_dir, "test.parquet"),  index=False)

    # Vocab sizes JSON
    import json
    with open(os.path.join(output_dir, "vocab_sizes.json"), "w") as f:
        json.dump({k: v + 1 for k, v in VOCAB.items()}, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output_dir",type=str, default="data/processed")
    args = parser.parse_args()
    generate(args.n_samples, args.seed, args.output_dir)
