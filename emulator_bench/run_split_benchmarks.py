import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parents[1]))

from emulator_bench.common import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_VALUE_TYPES,
    discover_split_jobs,
    normalize_threshold_args,
    seed_result_dir,
    split_sizes,
    summarize_seed_runs,
)
from src.utils.rich_progress import progress

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_regression_tvt.py"
METRIC_COLS = ["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"]


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--datasets_dir",
        args.datasets_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--device",
        args.cache_device,
        "--value_types",
        *args.value_types,
        "--max_residues",
        str(args.max_residues),
        "--max_batch",
        str(args.max_batch),
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.amp:
        cmd.append("--amp")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def train_one(job, seed, args, gpu_id=None):
    result_root = seed_result_dir(args.output_dir, job, seed)
    if (result_root / "final_results_test.csv").exists() and not args.overwrite:
        return result_root
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        job["train_path"],
        "--val_path",
        job["val_path"],
        "--test_path",
        job["test_path"],
        "--value_type",
        job["value_type"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(result_root),
        "--task_name",
        f"{job['value_type']}_{job['split_group']}_{job['split_name']}_seed{seed}",
        "--device",
        "cuda:0" if gpu_id is not None else args.device,
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
    ]
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.overwrite:
        cmd.append("--no_resume")
    if args.amp:
        cmd.append("--amp")
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return result_root


def result_row(job, seed, result_root, **extra):
    test_metrics = pd.read_csv(result_root / "final_results_test.csv").iloc[0].to_dict()
    return {
        "value_type": job["value_type"],
        "split_group": job["split_group"],
        "split_name": job["split_name"],
        "difficulty": job["difficulty"],
        "seed": int(seed),
        **extra,
        "run_dir": str(result_root),
        **split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])),
        **test_metrics,
    }


def run_parallel_jobs(jobs, args):
    work_queue = queue.Queue()
    for job_index, job in enumerate(jobs, start=1):
        for seed in args.seeds:
            work_queue.put((job_index, job, seed))

    rows = []
    rows_lock = threading.Lock()

    def worker(gpu_id):
        while True:
            try:
                job_index, job, seed = work_queue.get_nowait()
            except queue.Empty:
                return
            print(
                f"[GPU {gpu_id}] Running job {job_index}/{len(jobs)}: "
                f"{job['value_type']} | {job['split_group']} | {job['split_name']} | seed {seed}",
                flush=True,
            )
            try:
                row = result_row(job, seed, train_one(job, seed, args, gpu_id=gpu_id), gpu_id=str(gpu_id))
            except Exception as exc:
                row = {
                    "value_type": job["value_type"],
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "seed": int(seed),
                    "gpu_id": str(gpu_id),
                    "status": "failed",
                    "error": str(exc),
                }
            with rows_lock:
                rows.append(row)

    threads = [
        threading.Thread(target=worker, args=(str(gpu_id),), daemon=True)
        for gpu_id in args.gpus
        for _ in range(args.runs_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run BIND with its original settings across EMULaToR splits and seeds.")
    parser.add_argument("--datasets_dir", type=str, default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR / "bind_results"))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--value_types", nargs="+", default=DEFAULT_VALUE_TYPES)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_residues", type=int, default=8192)
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Opt-in bf16 autocast and TF32 on Ampere+ (default: the original fp16 autocast for training, fp32 ESM-2 caching).")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    jobs = discover_split_jobs(Path(args.datasets_dir), value_types=args.value_types, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.datasets_dir}")
    print(
        f"Discovered {len(jobs)} split jobs across {len(args.value_types)} value types. "
        f"Running {len(jobs) * len(args.seeds)} job/seed combinations.",
        flush=True,
    )

    if args.gpus:
        rows = run_parallel_jobs(jobs, args)
    else:
        rows = []
        for job_index, job in enumerate(progress(jobs, desc="Split jobs", unit="job"), start=1):
            for seed in args.seeds:
                print(f"Running job {job_index}/{len(jobs)}: {job['value_type']} | {job['split_group']} | {job['split_name']} | seed {seed}", flush=True)
                rows.append(result_row(job, seed, train_one(job, seed, args)))

    summary_dir = Path(args.output_dir) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_dir / "all_seed_runs.csv", index=False)
    summarize_seed_runs(
        [row for row in rows if row.get("status") != "failed"],
        group_cols=["value_type", "split_group", "split_name", "difficulty"],
        metric_cols=METRIC_COLS,
    ).to_csv(summary_dir / "summary_by_split.csv", index=False)
    print(f"Saved aggregate results to {summary_dir}")


if __name__ == "__main__":
    main()
