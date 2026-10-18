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
- Budget: 10.37 epochs of the training split, as in the paper (100 000 iterations over 2.47 M rows). The test metrics come from the final model.
- Leaky ReLU slope 0.01 and GATv2 slope 0.01, as in the released code. The paper text says 0.05.
- Proteins are cut to the first 2048 residues. No row is dropped.
- Training uses fp16 autocast with a grad scaler, as in the original `train_graph.py`. ESM-2 caching runs in fp32 and the cache is stored as float32 (the original runs ESM-2 in fp32 on CPU). `--amp` opts in to bf16 autocast and TF32 on Ampere or newer.
- A val or test row missing from the cache stops the run with an error. Train rows missing from the cache are counted in `run_summary.json` (`filters`) and printed.

See `REPORT.md` for the sources and the paper-versus-code differences.

## Files

- `cache_embeddings.py`, `launch_parallel_cache_embeddings.py`: build the shared protein and ligand cache.
- `train_regression_tvt.py`: train and evaluate one task on one split with one seed.
- `run_split_benchmarks.py`: run all discovered splits and seeds (default seeds 666 777 888), optionally over many GPUs (`--gpus 0 1 --runs_per_gpu 2`). `--amp` switches the fp16 default to bf16.
- `aggregate_bind_results.py`: mean and variance over seeds.

## Paths

- `--datasets_dir`: default `<EMULaToR>/data/processed/datasets`.
- `--embeddings_dir`: default `<EMULaToR>/outputs/baselines/BIND/embeddings`.
- `--output_dir`: default `<EMULaToR>/outputs/baselines/BIND`. Each run writes to `<value_type>/<split_group>/<split_name>/seed_<N>`. `<split_name>` is `all` for flat split groups. Summaries go to `<output_dir>/summaries` and `<output_dir>/aggregate_bind_results.csv`.
- The cache has `proteins_v3/<model>_layers_<ids>_cap_<protein_length_cap>_cls_eos/` and `ligands_v3/`. Change `CACHE_VERSION` in `common.py` to force a rebuild.

Each run writes `bestmodel.pth`, `bestmodel_state_dict.pth`, `checkpoint_last.pt`, `final_results_{train,val,test}.csv`, `pred_label_{train,val,test}.csv`, `logfile.csv` and `run_summary.json`. A finished seed is skipped unless `--overwrite` is set. `--overwrite` also retrains from scratch and ignores `checkpoint_last.pt`. See `../commands.txt` for the commands. Run them from `baselines/BIND`.
