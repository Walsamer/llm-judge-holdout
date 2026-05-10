#!/usr/bin/env python3
"""Check model pool coverage in the local arena-human-preference-140k dataset."""
import glob
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
from src.models.config import ARENA_IDS, MODEL_POOL

DATA_DIR = Path(__file__).parents[1] / "data" / "arena-human-preference-140k" / "data"
TARGET_MODELS = ARENA_IDS


def main():
    parquet_files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    print(f"Shards found: {len(parquet_files)}")

    model_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    pool_matchups = 0
    total_rows = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # first-turn, valid winner only
        df = df[(df["evaluation_order"] == 1) & df["winner"].isin(["model_a", "model_b", "tie"])]
        total_rows += len(df)

        for _, row in df.iterrows():
            ma, mb = row["model_a"], row["model_b"]
            model_counts[ma] += 1
            model_counts[mb] += 1
            if ma in TARGET_MODELS and mb in TARGET_MODELS:
                pool_matchups += 1
                pair_counts[tuple(sorted([ma, mb]))] += 1

    print(f"Total first-turn rows (valid winner): {total_rows}")
    print(f"\nMatchups where BOTH models are in the pool: {pool_matchups}")

    if pair_counts:
        print("\nPer-pair breakdown:")
        for pair, count in pair_counts.most_common():
            print(f"  {pair[0]} vs {pair[1]}: {count}")
    else:
        print("\nNo pool-vs-pool matchups found.")
        print("Pool model IDs not present in this dataset (check naming):")
        top_models = {m for m, _ in model_counts.most_common(50)}
        for m in TARGET_MODELS:
            if m not in top_models:
                print(f"  MISSING: {m}")

    print("\nTop 30 models in dataset:")
    for model, count in model_counts.most_common(30):
        flag = " <-- POOL" if model in TARGET_MODELS else ""
        print(f"  {model}: {count}{flag}")


if __name__ == "__main__":
    main()
