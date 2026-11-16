import argparse
import json
import logging
import os
import time
import multiprocessing
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import transformers
try:
    from src.utils.rich_progress import progress
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.utils.rich_progress import progress

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    CACHE_VERSION,
    PROTEIN_CACHE_VERSION,
    DEFAULT_DATASETS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HIDDEN_STATE_INDICES,
    DEFAULT_VALUE_TYPES,
    discover_split_jobs,
    ensure_parent,
    ligand_cache_path,
    normalize_sequence,
    normalize_threshold_args,
    protein_cache_dir,
    protein_cache_path,
    read_table,
    reset_protein_cache,
    require_columns,
    load_json,
    save_json,
    stable_hash,
    truncate_sequence,
)
from emulator_bench.feature_pipeline import (
    PROTEIN_MODEL_NAME,
    build_esm_batches,
    build_token_lut,
    esm_sequence_limit,
    graph_cache_item,
    load_esm2_model,
    protein_cache_item,
    resolve_amp_dtype,
    save_esm_embedding,
    _esm_forward,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="modified_smiles_parser")
warnings.filterwarnings("ignore", category=UserWarning, module="pysmiles")


def _save_npz(path: Path, item: dict) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **item)
    tmp_path.replace(path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


def _save_protein(path: Path, hidden_states) -> None:
    _save_npy(path, protein_cache_item(hidden_states))


class AsyncNpzWriter:
    def __init__(self, max_workers: int = 8, max_pending: int = 64):
        self.max_workers = max(1, int(max_workers))
        self.max_pending = max(1, int(max_pending))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="bind_cache_write")
        self._pending = set()

    def submit(self, path: Path, item) -> None:
        future = self._executor.submit(_save_protein if isinstance(item, list) else _save_npz, path, item)
        self._pending.add(future)
        self._drain_if_needed()

    def _drain_if_needed(self) -> None:
        while len(self._pending) >= self.max_pending:
            done, not_done = wait(self._pending, return_when=FIRST_COMPLETED)
            self._pending = set(not_done)
            for future in done:
                future.result()

    def flush(self) -> None:
        while self._pending:
            done, not_done = wait(self._pending, return_when=FIRST_COMPLETED)
            self._pending = set(not_done)
            for future in done:
                future.result()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)


def _collect_unique_values(jobs, sequence_col: str, smiles_col: str, protein_length_cap: int):
    sequences = set()
    smiles_values = set()
    truncated_count = 0
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[split_key]))
            require_columns(frame, [sequence_col, smiles_col], Path(job[split_key]))
            for value in frame[sequence_col].dropna().astype(str):
                normalized = normalize_sequence(value)
                truncated = truncate_sequence(normalized, protein_length_cap)
                if len(truncated) < len(normalized):
                    truncated_count += 1
                sequences.add(truncated)
            smiles_values.update(str(value).strip() for value in frame[smiles_col].dropna().astype(str))
    return sorted(sequences), sorted(smiles_values), truncated_count


def _apply_shard(items, num_shards: int, shard_index: int):
    if int(num_shards) <= 1:
        return list(items)
    shard_items = []
    for item in items:
        if int(stable_hash(str(item)), 16) % int(num_shards) == int(shard_index):
            shard_items.append(item)
    return shard_items


def _check_cache_config(args, precision_mode: str, model_revision: str = "") -> str:
    """Pin the settings that change stored values. A resumed build must use the same settings.

    --overwrite first deletes the old entries, so a crashed rebuild never leaves them beside the new config.
    Sharded workers skip the deletion because the launcher does it once before they start."""
    config = {
        "protein_cache_version": PROTEIN_CACHE_VERSION,
        "protein_model_name": args.protein_model_name,
        "protein_model_revision": model_revision,
        "transformers_version": transformers.__version__,
        "hidden_state_indices": args.hidden_state_indices,
        "protein_length_cap": int(args.protein_length_cap),
        "dtype": "float16",
        "precision_mode": precision_mode,
    }
    path = protein_cache_dir(args.embeddings_dir, args.protein_model_name, args.hidden_state_indices, args.protein_length_cap) / "cache_config.json"
    if path.exists() and not args.overwrite and load_json(path) != config:
        raise RuntimeError(f"{path} holds different settings. Use the same settings or pass --overwrite.")
    if args.overwrite and int(args.num_shards) == 1:
        reset_protein_cache(args.embeddings_dir, args.protein_model_name, args.hidden_state_indices, args.protein_length_cap)
    if not path.exists() or args.overwrite:
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(config, sort_keys=True))
        tmp_path.replace(path)
    return stable_hash(json.dumps(config, sort_keys=True))


def _entry_is_valid(path: Path, sequence: str, layers: int, width: int) -> bool:
    """True when the file loads and holds the expected (layers, residues + cls/eos, width) array."""
    try:
        return np.load(path, mmap_mode="r").shape == (layers, len(sequence) + 2, width)
    except (OSError, ValueError, EOFError):
        return False


def cache_proteins(args, sequences):
    if not sequences:
        return {"proteins_total": 0, "proteins_written": 0, "proteins_skipped_long": 0}

    device = torch.device(args.device)
    autocast_dtype, precision_mode = resolve_amp_dtype(device, args.amp)
    model, tokenizer = load_esm2_model(device, model_name=args.protein_model_name)
    model_limit = esm_sequence_limit(model)
    config_fingerprint = _check_cache_config(args, precision_mode, getattr(model.config, "_commit_hash", None) or "")
    save_esm_embedding(
        args.embeddings_dir,
        args.protein_model_name,
        args.hidden_state_indices,
        args.protein_length_cap,
        model.embeddings.word_embeddings.weight.detach().cpu().numpy(),
        build_token_lut(tokenizer),
    )
    if args.long_sequence_strategy == "drop":
        allowed = [sequence for sequence in sequences if len(sequence) <= model_limit]
        skipped_long = len(sequences) - len(allowed)
    else:
        allowed = list(sequences)
        skipped_long = 0

    sharded_allowed = _apply_shard(allowed, num_shards=args.num_shards, shard_index=args.shard_index)
    pending = [
        sequence
        for sequence in sharded_allowed
        if args.overwrite
        or not _entry_is_valid(
            protein_cache_path(
                args.embeddings_dir,
                sequence,
                model_name=args.protein_model_name,
                layer_indices=args.hidden_state_indices,
                protein_length_cap=args.protein_length_cap,
            ),
            sequence,
            len(args.hidden_state_indices) - 1,
            model.config.hidden_size,
        )
    ]
    if not pending:
        print("Protein cache is already complete.")
        return {
            "proteins_total": len(sharded_allowed),
            "proteins_written": 0,
            "proteins_skipped_long": skipped_long,
            "precision_mode": precision_mode,
            "config_fingerprint": config_fingerprint,
            "model_sequence_limit": model_limit,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
        }

    print(
        f"Protein cache device: {device} | precision: {precision_mode} | "
        f"model residue limit: {model_limit} | protein cap: {int(args.protein_length_cap)} | "
        f"long sequence strategy: {args.long_sequence_strategy} | "
        f"shard {int(args.shard_index) + 1}/{int(args.num_shards)}"
    )

    if args.long_sequence_strategy == "sliding_window":
        direct_sequences = [sequence for sequence in pending if len(sequence) <= model_limit]
        long_sequences = [sequence for sequence in pending if len(sequence) > model_limit]
    else:
        direct_sequences = list(pending)
        long_sequences = []
    batches = build_esm_batches(
        direct_sequences,
        max_residues=args.max_residues,
        max_batch=args.max_batch,
    )

    written = 0
    writer = AsyncNpzWriter(max_workers=args.writer_threads, max_pending=args.max_pending_writes)
    iterator = progress(batches, desc="Caching protein embeddings", unit="batch")
    try:
        for batch in iterator:
            embedded = _esm_forward(
                model,
                tokenizer,
                batch,
                layer_indices=args.hidden_state_indices[1:],
                device=device,
                autocast_dtype=autocast_dtype,
            )
            for sequence in batch:
                writer.submit(
                    protein_cache_path(
                        args.embeddings_dir,
                        sequence,
                        model_name=args.protein_model_name,
                        layer_indices=args.hidden_state_indices,
                        protein_length_cap=args.protein_length_cap,
                    ),
                    [embedded[sequence][f"layer_{index}"] for index in args.hidden_state_indices[1:]],
                )
                written += 1
            iterator.set_postfix(written=written, remaining=len(pending) - written)
    finally:
        writer.close()

    if long_sequences:
        raise RuntimeError("Long sequences unexpectedly reached the sliding-window cache path.")

    return {
        "proteins_total": len(sharded_allowed),
        "proteins_written": written,
        "proteins_skipped_long": skipped_long,
        "precision_mode": precision_mode,
        "config_fingerprint": config_fingerprint,
        "model_sequence_limit": model_limit,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }


def _write_ligand(job):
    path, smiles = job
    try:
        _save_npz(path, graph_cache_item(smiles))
    except Exception as exc:
        return smiles, str(exc)
    return smiles, None


def cache_ligands(args, smiles_values):
    sharded_smiles = _apply_shard(smiles_values, num_shards=args.num_shards, shard_index=args.shard_index)
    pending = [
        smiles
        for smiles in sharded_smiles
        if args.overwrite or not ligand_cache_path(args.embeddings_dir, smiles).exists()
    ]
    if not pending:
        print("Ligand cache is already complete.")
        return {
            "ligands_total": len(sharded_smiles),
            "ligands_written": 0,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
        }

    written = 0
    failed = []
    jobs = [(ligand_cache_path(args.embeddings_dir, smiles), smiles) for smiles in pending]
    workers = max(1, min(int(args.ligand_workers), len(jobs)))
    chunksize = max(1, min(256, len(jobs) // (workers * 8)))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        iterator = progress(pool.imap_unordered(_write_ligand, jobs, chunksize), desc="Caching ligand graphs", unit="smiles", total=len(jobs))
        for smiles, error in iterator:
            if error is None:
                written += 1
            else:
                failed.append({"smiles": smiles, "error": error})
            iterator.set_postfix(written=written, failed=len(failed), remaining=len(jobs) - written - len(failed))
    if failed:
        save_json(args.embeddings_dir / "ligand_cache_failures.json", {"failures": failed})
    return {
        "ligands_total": len(sharded_smiles),
        "ligands_written": written,
        "ligands_failed": len(failed),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }


def main():
    parser = argparse.ArgumentParser(description="Cache reusable BIND protein hidden states and ligand graphs.")
    parser.add_argument("--datasets_dir", type=str, default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--value_types", nargs="+", default=DEFAULT_VALUE_TYPES)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protein_model_name", type=str, default=PROTEIN_MODEL_NAME)
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")
    parser.add_argument("--max_residues", type=int, default=8192)
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--sliding_window_stride", type=int, default=896)
    parser.add_argument("--amp", action="store_true", help="Opt-in bf16 autocast (default: the original fp16 autocast for training, fp32 ESM-2 caching).")
    parser.add_argument("--writer_threads", type=int, default=max(4, min(16, os.cpu_count() or 4)))
    parser.add_argument("--ligand_workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--max_pending_writes", type=int, default=128)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.datasets_dir = Path(args.datasets_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.hidden_state_indices = [int(index) for index in args.hidden_state_indices]
    if args.hidden_state_indices[0] != 0:
        raise ValueError("--hidden_state_indices must start with 0: layer 0 is rebuilt from the token embedding.")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive")
    if not (0 <= int(args.shard_index) < int(args.num_shards)):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_split_jobs(
        args.datasets_dir,
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.datasets_dir}")

    started = time.time()
    sequences, smiles_values, truncated_count = _collect_unique_values(
        jobs,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        protein_length_cap=args.protein_length_cap,
    )
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique normalized sequences after first-{int(args.protein_length_cap)} truncation: {len(sequences)}")
    print(f"Rows truncated to first {int(args.protein_length_cap)} residues: {truncated_count}")
    print(f"Unique smiles: {len(smiles_values)}")

    ligand_stats = cache_ligands(args, smiles_values)  # before CUDA starts, so the fork pool is safe
    protein_stats = cache_proteins(args, sequences)

    manifest = {
        "cache_version": CACHE_VERSION,
        "protein_cache_version": PROTEIN_CACHE_VERSION,
        "datasets_dir": str(args.datasets_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "value_types": list(args.value_types),
        "split_groups": list(args.split_groups) if args.split_groups else None,
        "thresholds": args.thresholds,
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "protein_model_name": args.protein_model_name,
        "hidden_state_indices": args.hidden_state_indices,
        "protein_length_cap": int(args.protein_length_cap),
        "long_sequence_strategy": args.long_sequence_strategy,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "truncated_rows_to_cap": int(truncated_count),
        "input_fingerprint": stable_hash("\n".join(sequences + smiles_values)),
        "protein_cache": protein_stats,
        "ligand_cache": ligand_stats,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.embeddings_dir / "manifest.json", manifest)
    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
