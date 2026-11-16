#!/usr/bin/env bash
# BIND Ki, Kd, EC50, IC50 runs: original fp16 autocast (default, no --amp), micro batch 32. train also evaluates.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1
# shellcheck disable=SC2086  # PY, GPUS, SEEDS and the task lists split into words on purpose
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/BIND}
PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
SEEDS=${SEEDS:-"666 777 888"}
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache|train|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

TYPES="ki kd ec50 ic50"

# The paper cache had two passes: Ki, Kd, EC50 first, then IC50 (finished entries are skipped).
want cache && for vt in "ki kd ec50" ic50; do
  run $PY emulator_bench/launch_parallel_cache_embeddings.py --gpus $GPUS --value_types $vt \
    --datasets_dir "$DATA_ROOT" --embeddings_dir "$OUT_ROOT/embeddings"
done
want train && run $PY emulator_bench/run_split_benchmarks.py --skip_cache --value_types $TYPES --seeds $SEEDS \
  --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU" \
  --datasets_dir "$DATA_ROOT" --embeddings_dir "$OUT_ROOT/embeddings" --output_dir "$OUT_ROOT"
want aggregate && run $PY emulator_bench/aggregate_bind_results.py --output_dir "$OUT_ROOT"
exit 0
