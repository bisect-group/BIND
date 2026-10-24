# emulator_bench

Multi-seed evaluation of `BIND` with its original published settings on the default EMULaToR splits. The original `train_graph.py` path is not changed.

## Tasks and data

| Task | Split parent |
|---|---|
| `ki` | `data/processed/datasets/kinetic_dataset/ki` |
| `kd`, `ic50`, `ec50` | `data/processed/datasets/binding_affinity_dataset/<task>` |

Each task folder holds `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_<t>` and `substrate_splits/threshold_<t>`. Each split has `train/val/test.parquet`. The wrapper finds the folders on disk. The wrapper never uses the unified datasets.

Inputs per row: `sequence`, `smiles`, `log10_value` (target).

## Model and settings

- Protein: frozen `facebook/esm2_t33_650M_UR50D`. The cross-attention uses hidden states 0, 10, 20 and 30, as in the released `model_graph.py`. The wrapper caches these.
- Ligand: the graph from `loading.py`, cached once.
- One BIND model per task. Only the matching regression head is trained. The classifier head is not used.
- AdamW, lr 1e-4, weight decay 1e-3, cosine schedule, Huber loss with delta 2.0, no gradient clipping, no early stopping.
- Effective batch size 256 (64 x 4 accumulation). The original uses 1 x 256.
- Each drawn batch of 64 rows runs as length-sorted chunks of `--micro_batch_size` rows (default 32). Gradients accumulate so one optimizer step covers the same rows with the same loss scale. Results match the unchunked run only within floating-point noise. `--micro_batch_size 64` (the batch size) turns chunking off. A chunk of 64 is slower and needs more memory.
- The forward pass computes the released maths with batched kernels: LCM pooling runs one GRU call per tree level for all graphs of a chunk, cross-attention builds its dense node grid with one scatter, and the K/V projections run on the residues without padding. The outputs match the per-graph code within fp32 rounding (see `REPORT.md`).
- Budget: 10.37 epochs of the training split, as in the paper (100 000 iterations over 2.47 M rows). The test metrics come from the final model.
- Leaky ReLU slope 0.01 and GATv2 slope 0.01, as in the released code. The paper text says 0.05.
- Proteins are cut to the first 2048 residues. No row is dropped.
- Training uses fp16 autocast with a grad scaler, as in the original `train_graph.py`. ESM-2 caching runs in fp32 and the cache stores layers 10, 20 and 30 as float16 (the original runs ESM-2 in fp32 on CPU). The fp16 autocast of the original rounds these layers to fp16 anyway, so training sees the same values. Layer 0 is not stored. The model rebuilds it from the token ids and the ESM-2 token embedding table, and the cache builder checks that this equals the real layer 0. `--amp` opts in to bf16 autocast and TF32 on Ampere or newer.
- A row without a ligand graph (for example a one-atom ion, or a SMILES the parser rejects) is dropped from train and val. A test row without a ligand graph stays in the test set. Its prediction is the mean of the train targets. `pred_label_test.csv` has a `prediction_source` column (`model` or `train_mean_fallback`), and the test metrics cover all test rows. `run_summary.json` stores `val_rows_dropped_no_ligand`, `test_rows_train_mean_fallback` and `train_mean`. A val or test row with a missing protein entry still stops the run with an error. Counts per split are in `run_summary.json` (`filters`) and are printed.

See `REPORT.md` for the sources and the paper-versus-code differences.

## Files

- `cache_embeddings.py`, `launch_parallel_cache_embeddings.py`: build the shared protein and ligand cache. The build is resumable: finished proteins are skipped, and files are written atomically. `cache_config.json` pins the settings of the cache. A rerun with other settings stops, unless `--overwrite` is set.
- `train_regression_tvt.py`: train and evaluate one task on one split with one seed.
- `run_split_benchmarks.py`: run all discovered splits and seeds (default seeds 666 777 888), optionally over many GPUs (`--gpus 0 1 --runs_per_gpu 2`). Two runs per GPU give about 1.45 times the throughput of one on an A10. Three can run out of memory. `--micro_batch_size` is passed to the trainer. `--amp` switches the fp16 default to bf16.
- `aggregate_bind_results.py`: mean and variance over seeds.

## Paths

- `--datasets_dir`: default `<EMULaToR>/data/processed/datasets`.
- `--embeddings_dir`: default `<EMULaToR>/outputs/baselines/BIND/embeddings`.
- `--output_dir`: default `<EMULaToR>/outputs/baselines/BIND`. Each run writes to `<value_type>/<split_group>/<split_name>/seed_<N>`. `<split_name>` is `all` for flat split groups. Summaries go to `<output_dir>/summaries` and `<output_dir>/aggregate_bind_results.csv`.
- The cache has `proteins_v5/<model>_layers_<ids>_cap_<protein_length_cap>_cls_eos/` and `ligands_v3/`. Each protein is one uncompressed float16 `.npy` of shape (3 layers, residues + 2, 1280), read with `np.load(mmap_mode="r")`. The folder also holds `esm_embedding.npz` (token embedding table and token ids) and `cache_config.json`. Threads in the training process copy the residues into pinned memory, so the workers only prepare graphs and indices. All runs share the OS page cache. `cache_embeddings.py --overwrite` deletes the old protein entries first. A rerun skips an entry only when its `.npy` shape matches the sequence. Change `PROTEIN_CACHE_VERSION` or `CACHE_VERSION` in `common.py` to force a rebuild. A `proteins_v4` cache is not read; rebuild with `cache_embeddings.py`.

The trainer sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless it is already set. `--gather_threads` (default 6) sets the threads that fill pinned memory with protein states.

Each run writes `bestmodel.pth`, `bestmodel_state_dict.pth`, `checkpoint_last.pt`, `final_results_{train,val,test}.csv`, `pred_label_{train,val,test}.csv`, `logfile.csv` and `run_summary.json`. A finished seed is skipped unless `--overwrite` is set. `--overwrite` also retrains from scratch and ignores `checkpoint_last.pt`. Checkpoints from the old code (`pool.pool.*` keys) do not load. Delete them before a rerun. See `../commands.txt` for the commands. Run them from `baselines/BIND`.
