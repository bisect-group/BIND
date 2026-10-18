"""Aggregate BIND results across seeds.

Reads final_results_{train,val,test}.csv for every seed under the output
directory (<value_type>/<split_group>/<split_name>/seed_N) and writes the mean
and variance of each metric per split to aggregate_bind_results.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import DEFAULT_OUTPUT_DIR

SPLITS = ("train", "val", "test")
METRICS = ("mae", "mse", "rmse", "r2_score", "pearson", "spearman", "loss")


def load_seed_results(seed_dir: Path):
    results = {}
    for split in SPLITS:
        path = seed_dir / f"final_results_{split}.csv"
        if not path.exists():
            return None
        results[split] = pd.read_csv(path).iloc[0]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = []
    for seed_dir in sorted(args.output_dir.glob("*/*/*/seed_*")):
        split_results = load_seed_results(seed_dir)
        if split_results is None:
            print(f"[skip] incomplete: {seed_dir.relative_to(args.output_dir)}")
            continue
        value_type, split_group, split_name, seed = seed_dir.relative_to(args.output_dir).parts
        for split, series in split_results.items():
            rows.append(
                dict(value_type=value_type, split_group=split_group, split_name=split_name, seed=seed, tvt_split=split)
                | {metric: series[metric] for metric in METRICS if metric in series.index}
            )
    if not rows:
        print("No complete results found.")
        return

    df = pd.DataFrame(rows)
    group_keys = ["value_type", "split_group", "split_name", "tvt_split"]
    grouped = df.groupby(group_keys)
    agg = grouped[[m for m in METRICS if m in df.columns]].agg(["mean", "var"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = grouped.size()
    agg = agg.reset_index()

    out_path = args.output_dir / "aggregate_bind_results.csv"
    agg.to_csv(out_path, index=False)
    print(f"Saved {len(agg)} rows to {out_path}")


if __name__ == "__main__":
    main()
