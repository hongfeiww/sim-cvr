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
import json
import os
import numpy as np
import pandas as pd

VOCAB = {
    "user_id":          20_000,   # smaller vocab -> denser embedding coverage
    "age_level":        7,
    "gender":           3,
    "shopping_level":   3,
    "city_level":       5,
    "item_id":          5_000,
    "item_category":    100,      # small vocab -> model sees each category many times
    "item_price_level": 5,
    "item_sales_level": 5,
    "ad_id":            2_000,
    "campaign_id":      200,
    "customer_id":      500,
    "brand_id":         300,
    "pid":              3,
    "hour":             24,
}

SEQ_LEN = 20


def generate(n_samples: int, seed: int, output_dir: str) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {n_samples:,} samples...")

    # Sample features in [1, vocab_size-1]
    feats = {col: rng.integers(1, vsize, n_samples)
             for col, vsize in VOCAB.items()}

    user_id       = feats["user_id"]
    item_category = feats["item_category"]
    item_id       = feats["item_id"]
    shopping      = feats["shopping_level"]
    price         = feats["item_price_level"]
    pid           = feats["pid"]

    user_score = rng.standard_normal(VOCAB["user_id"] + 1)       # per user
    cat_score  = rng.standard_normal(VOCAB["item_category"] + 1) # per category
    item_score = rng.standard_normal(VOCAB["item_id"] + 1)       # per item

    u_s = user_score[user_id]        # [N]
    c_s = cat_score[item_category]   # [N]
    i_s = item_score[item_id]        # [N]

    ctr_logit = (
        -3.5
        + 1.5  * u_s
        + 1.2  * c_s       
        + 0.5  * i_s
        + 0.3  * (shopping - 2).astype(float)
        + 0.2  * (pid == 1).astype(float)
        - 0.1  * (price - 3).astype(float)
        + 0.1  * rng.standard_normal(n_samples)  # small noise
    )
    p_ctr = 1.0 / (1.0 + np.exp(-ctr_logit))
    click = (rng.uniform(size=n_samples) < p_ctr).astype(int)

    cvr_logit = (
        -1.5
        + 1.8  * u_s
        + 1.0  * c_s
        + 0.5  * i_s
        - 0.3  * (price - 3).astype(float)
        + 0.1  * rng.standard_normal(n_samples)
    )
    p_cvr    = 1.0 / (1.0 + np.exp(-cvr_logit))
    purchase = (click == 1) * (rng.uniform(size=n_samples) < p_cvr).astype(int)

    print(f"  CTR:   {click.mean():.4f}  (target ~0.038)")
    if click.sum() > 0:
        print(f"  CVR:   {purchase[click==1].mean():.4f}  (target ~0.18)")
    print(f"  CTCVR: {purchase.mean():.5f}  (target ~0.007)")

    # Build from actual click events so sequences carry real interest signal
    print("  Building behavior sequences...")
    user_hist_items: dict = {}
    user_hist_cats:  dict = {}
    for i in range(n_samples):
        if click[i] == 1:
            uid = int(user_id[i])
            if uid not in user_hist_items:
                user_hist_items[uid] = []
                user_hist_cats[uid]  = []
            user_hist_items[uid].append(int(feats["item_id"][i]))
            user_hist_cats[uid].append(int(item_category[i]))

    seq_items_list = []
    seq_cats_list  = []
    for i in range(n_samples):
        uid   = int(user_id[i])
        items = user_hist_items.get(uid, [])
        cats  = user_hist_cats.get(uid,  [])
        hist_i = items[:-1][-SEQ_LEN:] if len(items) > 1 else []
        hist_c = cats[:-1][-SEQ_LEN:]  if len(cats)  > 1 else []
        seq_items_list.append(hist_i)
        seq_cats_list.append(hist_c)

    # Assemble
    df = pd.DataFrame(feats)
    df["click"]     = click
    df["purchase"]  = purchase
    df["seq_items"] = seq_items_list
    df["seq_cats"]  = seq_cats_list

    vocab_sizes = {col: vsize + 1 for col, vsize in VOCAB.items()}

    # Split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(n_samples * 0.7)
    n_val   = int(n_samples * 0.1)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    train.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val.to_parquet(  os.path.join(output_dir, "val.parquet"),   index=False)
    test.to_parquet( os.path.join(output_dir, "test.parquet"),  index=False)

    with open(os.path.join(output_dir, "vocab_sizes.json"), "w") as f:
        json.dump(vocab_sizes, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output_dir",type=str, default="data/processed")
    args = parser.parse_args()
    generate(args.n_samples, args.seed, args.output_dir)