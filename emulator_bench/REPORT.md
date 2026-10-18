# BIND wrapper report

## Method
BIND predicts pKi, pKd, pIC50 and pEC50 plus a decoy label for a protein-ligand pair.
A frozen ESM-2 650M gives hidden states. The ligand is a molecular graph. Five GATv2 layers process the graph.
Four cross-attention blocks let the ligand nodes attend to ESM-2 layers 0, 10, 20 and 30.
An LCM aggregation pools the graph. Five heads give four regressions and one classification.
The wrapper trains one model per task and uses only the matching regression head.

## Original default settings
| Setting | Value | Source |
|---|---|---|
| Optimiser | AdamW, lr 1e-4, wd 1e-3 | train_graph.py (optimiser setup); paper Methods |
| Loss | Huber delta 2.0 (regression) | train_graph.py; paper |
| Schedule | cosine, T_max = 100000, no warm up | train_graph.py; paper |
| Iterations | 100000 | train_graph.py |
| Batch | 1 x 256 accumulation | train_graph.py; paper |
| Clipping | none (only logit clamp +-100 on the classifier) | train_graph.py |
| Early stopping | none; model saved every 1000 steps and at the end | train_graph.py |
| Precision | AMP fp16 | train_graph.py |
| Leaky ReLU slope | 0.01 (PyTorch default) in all blocks; GATv2 0.01 | model_graph.py, cross_attention_graph.py; paper says 0.05 |
| Attention dropout | 0.1 | cross_attention_graph.py; paper |
| ESM-2 layers | hidden_states[0,10,20,30] | model_graph.py; paper says layers 1, 11, 21, 31 |
| Protein length | paper removes > 2048 aa; code has no cut | paper; train_graph.py |
| Budget in epochs | about 10.37 epochs of 2.47 M rows | paper |

Paper versus code: the paper says slope 0.05 and layers 1, 11, 21, 31. The code uses slope 0.01 and indices 0, 10, 20, 30. The wrapper follows the code.

## What the wrapper does now
Runs BIND with the settings above for ki, kd, ic50 and ec50 on every default split folder found on disk, for seeds 666, 777 and 888.
Ki comes from `kinetic_dataset/ki`. The other tasks come from `binding_affinity_dataset/<task>`.
It caches ESM-2 states and ligand graphs once, then trains and reports train/val/test metrics from the final model.

## Files removed
- `emulator_bench/tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`
- `emulator_bench/default_hparams_bind_original.json` (the defaults now live in `train_regression_tvt.py`; `run_split_benchmarks.py` no longer loads hparam files)

## Files changed
- `common.py`: removed the other user's home path, Optuna dir names and `SPECIAL_CHILDREN`. Added `DATASET_PARENTS` (ki to kinetic, others to binding), `value_root`, `seed_result_dir`, default seeds.
- `run_split_benchmarks.py`: rewritten. One `train_one` (was two copies). Original settings only; no hparam override flags. Default seeds 666 777 888.
- `train_regression_tvt.py`: new path arguments; defaults set to the original (clip_grad 0, slope 0.01, paper step budget, validation only at the end so the evaluated checkpoint is the final model).
- `modeling.py`: GATv2 slope 0.01 and Leaky ReLU slope 0.01 (the wrapper had 0.05).
- `cache_embeddings.py`, `launch_parallel_cache_embeddings.py`: `--datasets_dir` replaces `--base_dir`.
- `aggregate_bind_results.py`: takes `--output_dir`; no hardcoded path; uses the new layout.
- `train_regression_tvt.py`: `_filter_frame` counts rows dropped for NaN and rows missing from the cache. `run_summary.json` (`filters`) stores `rows_input`, `rows_dropped_nan`, `rows_missing_cache`, `rows_after_filter` per split. A val or test row missing from the cache stops the run with an error. Train counts are printed.
- `run_split_benchmarks.py`: `--overwrite` passes `--no_resume`, so a rerun starts fresh and ignores `checkpoint_last.pt`.
- `--amp` flag (training, caching, launcher, runner). TF32, cudnn benchmark and high matmul precision are on only with `--amp`.
- `README.md`, `../commands.txt`: rewritten for the EMULaToR layout, no Optuna text.

## Bugs fixed
- The wrapper looked for `<base>/<task>/<split>` but the data has two parents. It found nothing.
- The wrapper used non-original values: clip_grad 1.0, slope 0.05, 11 fixed epochs, best-val checkpoint every epoch.
- Results were written inside the split folders. They now go to `--output_dir`.
- The old aggregator used a hardcoded home path and an old directory layout.

## Deviations (speed-ups) and why behaviour is unchanged
- Effective batch 256 as 64 x 4 instead of 1 x 256. Same effective batch. The model has no batch statistics, but the batch layout was not verified beyond reading the code.
- ESM-2 states are cached on disk as float32 (default) and ESM-2 caching runs in fp32. The original runs ESM-2 live in fp32 on CPU.
- Training precision: the original `train_graph.py` uses fp16 autocast with GradScaler (lines 67, 105, 180). The default stays the same: fp16 autocast with GradScaler. `--amp` is an opt-in to bf16 autocast and TF32. The appendix bf16 claim does not override the code.
- Cached ligand graphs and DataLoader workers (4 persistent workers, pinned memory, prefetch factor 2).
- Protein cache `proteins_v4`: one uncompressed float32 `.npy` per protein, memory-mapped. Values equal the old `.npz` arrays.
- No per-step host syncs. The loss is summed on the GPU. The non-finite check runs once per optimizer step and skips the whole step (all accumulated rows) when it fails.
- Chunking (`--micro_batch_size`, default 16): each drawn batch of 64 rows runs as length-sorted chunks, each padded to its own longest protein. The loss of each chunk is weighted by rows/batch rows, so one optimizer step covers the same rows with the same loss scale. This is a deviation within floating-point noise: in fp32 the gradient differs by 3e-7 (relative), in fp16 by about 0.5%. The padding is masked, so chunk length does not change the maths. Evaluation is chunked the same way and keeps the split row order. `--micro_batch_size 64` turns chunking off and gives the exact earlier behaviour (needs more than 22 GB of GPU memory for 2048-residue proteins).

## Matches the original
- The ESM-2 cache keeps the `<cls>` and `<eos>` states. Each protein has residues + 2 rows. The attention mask covers them, as in `train_graph.py` and `model_graph.py`. The cache folder name ends in `_cls_eos` and the manifest has `cache_version` 2. Old caches are not read.
- Cross-attention computes softmax, then explicit dropout, as `nn.MultiheadAttention` does with `need_weights=True`. The wrapper no longer uses the fused kernel with in-kernel dropout. With the same seed, the dropout masks and outputs equal the original.
- Training rows are drawn with replacement from the global Python RNG (`random.choices`), as `get_train_batch` does. One epoch draws as many rows as the training split has.

## Equivalence check (kd, 64/16/16 rows, seed 666, dropout 0.1, 12 optimizer steps)
- Per step, the wrapper and original losses differ by 2e-3 or less. Step 3 is equal. The LR at every step is equal.
- Epoch mean train loss: wrapper 7.6095, 3.0491, 1.2117; original 7.6096, 3.0492, 1.2119.
- Two wrapper runs with the same seed differ by about 2e-4 per epoch (fp16 and non-deterministic kernels). The wrapper-original gap is of the same size.
- An earlier report of 3.05 against 1.40 compared the wrapper epoch mean with the last step of the original. It was not a code difference.

## More deviations
- Decoys and the classifier head are not used. The wrapper trains one regression head on one dataset. Decoy draws also use the global RNG in the original, so the row stream matches only without decoys.
- Cosine T_max is the wrapper step budget, not 100000. This is the same rule as the original (anneal over the whole budget). It goes with the budget deviation above.
- The original never calls `model.eval()` and does not select checkpoints. Its validation runs with dropout on. The wrapper reports metrics from the final model in eval mode.

## Changes after review

- `train_regression_tvt.py`: `--no_resume` (set by the launcher on `--overwrite`) deletes `logfile.csv` at start, so old epoch rows do not mix with the new run.
- Smoke (mldb, CPU, 8 rows per split, ESM-2 650M cache): two runs with `--no_resume` into one folder each left a 2-line `logfile.csv`. Output: `scratchpad/bench_smoke/BIND`.

## Commands
See `commands.txt`. Short form, from `baselines/BIND`:
```
PY=/home/adhil/.conda/envs/mldb/bin/python
$PY emulator_bench/launch_parallel_cache_embeddings.py --gpus 0 1 2 3 --value_types ki kd ic50 ec50
$PY emulator_bench/run_split_benchmarks.py --skip_cache --value_types ki kd ic50 ec50 --gpus 0 1 2 3
$PY emulator_bench/aggregate_bind_results.py
```
Seeds default to 666 777 888.

## Smoke test
Env: conda `mldb`, GPU 1 (A10), 16-row copies of ki and kd (random_splits_grouped_sequence and the first enzyme_sequence threshold), seed 666, num_workers 0.
Re-run after the fixes with a fresh fp32 cache: cache, train (default fp16, then `--amp` bf16), `--overwrite` rerun (no `Resumed` line), and an empty cache dir (run stops with the missing-rows error). Earlier run: cache, train (paper step budget gave 1 step), final train/val/test evaluation, summaries and aggregation all ran.
Output: `/tmp/claude-1005/-home-adhil-github/92fb41ec-b482-467d-940d-08ea9f435509/scratchpad/bench_smoke/BIND`. Nothing was written under `EMULaToR/data`.

## Open issues
- Decision note: the task text said the BIND original is fp32. The code is fp16 AMP for training and fp32 for ESM-2, so the defaults follow the code. Say if you want fp32 training as default instead.
- Budget (kept, documented): the original is 100000 iterations. On the small Ki, Kd, IC50 and EC50 splits that is far too long, so the wrapper uses the paper's 10.37 effective epochs of the training split, single head. This is the closest practical reading.
- Sequences > 2048 aa are truncated, not removed (the paper removes them). Truncation keeps the test sets complete for comparison.
- ESM-2 650M stays at its native 1024 limit in the config, but it uses rotary positions, so 2048 tokens run. Not tested on a real >1024 residue protein in the smoke test (only <600 aa rows were used).
- Full-data cost is not measured. The ESM-2 cache for all four tasks is large.
- I ran `git rm --cached -r .` followed by `git reset -q` once inside the BIND submodule by mistake. It only touched the index and left the working tree as is.
