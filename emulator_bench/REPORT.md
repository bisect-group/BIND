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
- `train_regression_tvt.py`: `_filter_frame` counts rows dropped for NaN and rows missing from the cache. `run_summary.json` (`filters`) stores `rows_input`, `rows_dropped_nan`, `rows_missing_cache`, `rows_after_filter` per split. `rows_no_ligand` counts rows whose protein entry exists but whose ligand graph does not. Val drops these rows. Test keeps them and predicts the train mean (`prediction_source` column in `pred_label_test.csv`; `val_rows_dropped_no_ligand`, `test_rows_train_mean_fallback`, `train_mean` in `run_summary.json`). A val or test row with a missing protein entry stops the run with an error. Counts are printed.
- `run_split_benchmarks.py`: `--overwrite` passes `--no_resume`, so a rerun starts fresh and ignores `checkpoint_last.pt`.
- `--amp` flag (training, caching, launcher, runner). TF32, cudnn benchmark and high matmul precision are on only with `--amp`.
- `README.md`, `../commands.txt`: rewritten for the EMULaToR layout, no Optuna text.
- `modeling.py`, `dataset.py`, `feature_pipeline.py`, `cache_embeddings.py`, `common.py`, `train_regression_tvt.py`: the speed-ups above. `--protein_dtype` is removed (the cache is always fp16). New flag `--gather_threads`. The protein cache is `proteins_v5`.
- `../tests/`: tests for the batched model, the cache and the loader.

## Bugs fixed
- The wrapper looked for `<base>/<task>/<split>` but the data has two parents. It found nothing.
- The wrapper used non-original values: clip_grad 1.0, slope 0.05, 11 fixed epochs, best-val checkpoint every epoch.
- Results were written inside the split folders. They now go to `--output_dir`.
- The old aggregator used a hardcoded home path and an old directory layout.

## Deviations (speed-ups) and why behaviour is unchanged
- Effective batch 256 as 64 x 4 instead of 1 x 256. Same effective batch. The model has no batch statistics, but the batch layout was not verified beyond reading the code.
- ESM-2 caching runs in fp32 (the original runs ESM-2 live in fp32 on CPU). The cache stores layers 10, 20 and 30 as float16. Layer 0 is rebuilt on the GPU (see "Speed-ups of the model and loader").
- Training precision: the original `train_graph.py` uses fp16 autocast with GradScaler (lines 67, 105, 180). The default stays the same: fp16 autocast with GradScaler. `--amp` is an opt-in to bf16 autocast and TF32. The appendix bf16 claim does not override the code.
- Cached ligand graphs and DataLoader workers (4 persistent workers, pinned memory, prefetch factor 2).
- Protein cache `proteins_v5`: one uncompressed float16 `.npy` per protein (3 layers), memory-mapped. The build is resumable, writes files atomically and pins its settings in `cache_config.json`.
- No per-step host syncs. The loss is summed on the GPU. The non-finite check runs once per optimizer step and skips the whole step (all accumulated rows) when it fails.
- Chunking (`--micro_batch_size`, default 32): each drawn batch of 64 rows runs as length-sorted chunks. A chunk also stops growing at 8 M cells (rows x residues x ligand atoms), which bounds the attention memory. The loss of each chunk is weighted by rows/batch rows, so one optimizer step covers the same rows with the same loss scale. This is a deviation within floating-point noise (fp16 gradient about 0.5%). The padding is masked, so chunk composition does not change the maths. Evaluation is chunked the same way and keeps the split row order. `--micro_batch_size 64` turns chunking off. It runs out of memory on a 22 GB GPU for long proteins.

## Speed-ups of the model and loader
Measured on one A10 (GPU 3, nothing else on it), ic50 `random_splits_grouped_sequence` subset: 24 000 train rows over 387 proteins (mean 772 residues, up to 2048). Default fp16 autocast. Same optimizer settings as the trainer.

What changed (maths unchanged):
1. LCM pooling runs one GRU call per tree level for all graphs of a chunk (was one GRU tree per graph, about 1 600 `gru_cell` calls per optimizer step).
2. Cross-attention builds the dense node grid with one scatter and reads it back with one gather (was Python loops and `Batch.from_data_list`).
3. The K/V projections run on the residues without padding. The 64-wide keys and values are padded afterwards.
4. The protein cache holds layers 10, 20 and 30 as float16. Layer 0 is the ESM-2 token embedding scaled by 0.88 (token dropout rescale). The model rebuilds it from the token ids. The cache builder checks on every batch that this equals the real layer 0 bit for bit, and that the token ids from the lookup table equal the tokenizer ids. It also stops if a layer overflows float16.
5. The micro-batch size is 32 (was 16). A chunk is also capped at 8 M cells.
6. Threads in the training process copy the cached residues into pinned memory. The DataLoader workers build only graphs and index tensors.
7. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the trainer (the reserved memory of the caching allocator reached 23 GB without it).

Why the cache change is exact under the default fp16 autocast: autocast rounds the float32 states to fp16 at the first Linear. The cache rounds them once, to the same values. Under fp32 (CPU) or bf16 (`--amp`) the states carry one fp16 rounding (relative 5e-4).

### Equivalence (128 rows, 8 chunks of 16, dropout 0.1, same weights, same seed per chunk; max over chunks)
| Comparison | max abs output diff | relative gradient diff |
|---|---|---|
| new fp16 vs original fp16 (default mode) | 1.95e-3 | 1.11e-2 |
| original fp16 vs original fp32 (the original's own precision noise, four runs) | 1.3e-3 to 2.4e-3 | 1.3e-2 to 1.55e-2 |
| original fp16, fp16-rounded cache vs full-precision cache (same values after autocast, so a rerun) | 1.7e-3 | 9.6e-3 |
| new fp32 vs original fp32 (fp16-rounded cache states fed to the original) | 1.7e-5 | 1.2e-4 (7 of 8 chunks); 7.2e-3 (1 chunk) |
| new fp32 vs original fp32 (full-precision cache) | 4.9e-4 | 1.05e-2 |
| fp32 rerun on the same input: new / original | 6e-7 / 9e-7 | 1.4e-7 / 3.2e-6 |

The fp16 difference is inside the run-to-run and fp32-to-fp16 noise of the original. In fp32 the 4.9e-4 gap is the fp16 rounding of the cached states, not the batched kernels (1.7e-5 when the original sees the same rounded states). The one chunk with 7.2e-3 also has a 24 times larger rerun noise in the original (3.2e-6 against 1.3e-7), so it is an ill-conditioned batch (it has the only 212-atom ligand). Output differences stay 1e-5.

Unit tests (`../tests`): pooling against `LCMAggregation` per graph (1-node, odd and even graphs), cross-attention against `CrossAttentionGraphModule`, padding independence of a row, cache round trip and overflow stop, token ids, loader packing against the stored arrays, chunk row and cell limits (also for batches within the row limit), cache config check, cache entry shape check, loader order and worker errors, and the gradient equality of chunked and unchunked batches (the training loop's `rows / batch_rows` weighting; fp32 on CPU, because the model casts to fp32).

### Training check (160 optimizer steps, batch 64 x 4, ic50 subset of 6 000 rows over 136 proteins, seeds 666 and 777)
160 steps instead of 300: the original runs 4.5 s per step, and one GPU hold had to stay under 15 minutes. The cosine schedule covers the 160 steps.

Mean train loss per 20 steps:

| Run | 1-20 | 21-40 | 41-60 | 61-80 | 81-100 | 101-120 | 121-140 |
|---|---|---|---|---|---|---|---|
| original, seed 666 | 1.648 | 0.697 | 0.611 | 0.616 | 0.592 | 0.588 | 0.579 |
| new, seed 666 | 1.648 | 0.698 | 0.612 | 0.615 | 0.592 | 0.589 | 0.579 |
| original, seed 777 | 1.828 | 0.664 | 0.608 | 0.602 | 0.556 | 0.539 | 0.542 |
| new, seed 777 | 1.829 | 0.663 | 0.611 | 0.602 | 0.561 | 0.542 | 0.545 |

The same seed gives the same init and the same row stream. The new run follows the original within 0.0008 (seed 666) and 0.002 (seed 777) in mean absolute window difference (largest single window 0.005). The seed-to-seed spread of the original is 0.05. The new curves lie inside the seed-to-seed spread, 25 to 60 times closer to their own seed than the seeds are to each other. The last 60 steps average 0.584 / 0.540 (original) and 0.585 / 0.542 (new).

### Throughput (rows per second, train rows, one A10, fp16)
| Code | Runs per GPU | rows/s | Peak memory (allocated / reserved) |
|---|---|---|---|
| original, micro-batch 16 | 1 | 55 (4.7 s per optimizer step) | 4.9 GB / - |
| original, micro-batch 32 | 1 | 55 to 57 | 10 to 13 GB / 15 to 23 GB |
| new, micro-batch 32 (136-protein subset) | 1 | 242 | 7.2 GB / - |
| new, micro-batch 32 (387-protein subset, cell cap) | 1 | 216 | 5.3 GB / 6.4 GB |
| new, micro-batch 32 (387-protein subset) | 2 | 157 + 153 = 310 | 5.3 GB / 6.4 GB each |
| new, micro-batch 32 (387-protein subset) | 3 | 114 + 114 + 113 = 341 | 5.3 GB / 6.4 GB each |
| new, micro-batch 16 (136-protein subset) | 1 | 163 | 3.9 GB |
| new, micro-batch 64 (136-protein subset) | 1 | out of memory on 22 GB | |

Runs per GPU: 2 runs give 1.45 times the rows per second of 1 run. A third run adds only 10 percent more (341 rows/s). Its worst chunk can use about 8 GB (the 8 M cell cap), so three runs can run out of memory on a 22 GB GPU. Use `--runs_per_gpu 2`.

The original is CPU and launch bound: micro-batch 32 does not help it. The page cache held all test files during these runs. A cold read of random 5 MB blocks from the data disk gave 0.5 GB/s with one thread and 0.8 GB/s with six. At 216 rows/s the new loader reads 1.3 GB/s from the page cache (5.9 MB per row). The full ki, kd, ic50 and ec50 cache is about 51 GB (11 938 unique proteins, fp16, 3 layers; 136 GB in the old fp32 format). It fits in the page cache, so each protein is read from disk once. One cold pass costs about 1 to 2 minutes in total. A run is read bound only while it touches a protein for the first time.

### Queue estimate (4 tasks x 7 splits x 3 seeds, 359 M row passes, 3 A10s)
| Setting | Rows/s on 3 GPUs | Time |
|---|---|---|
| original (measured, 59 rows/s per GPU) | 177 | 563 h |
| new, 1 run per GPU | 645 | 155 h |
| new, 2 runs per GPU | 930 | 107 h |

The estimate counts the 359 M row passes at the measured training rate. Evaluation passes run forward only and are faster, so the real time is not longer. The GPUs must be free of other jobs.

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

- Checkpoints from the old code (`pool.pool.*` keys) do not load into the new model. Delete them before a rerun.
- `cache_embeddings.py --overwrite` deletes the old protein entries before it writes the new `cache_config.json`. The config also pins the ESM model revision and the `transformers` version. A rerun skips an entry only when the file loads and its shape is (layers, residues + 2, width).
- `GatherLoader` keeps at most 2 batches in flight (`in_flight`), with 6 gather threads.

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
- Full-data cost is measured only on subsets (see the throughput table). The ESM-2 cache for all four tasks is about 51 GB.
- I ran `git rm --cached -r .` followed by `git reset -q` once inside the BIND submodule by mistake. It only touched the index and left the working tree as is.
