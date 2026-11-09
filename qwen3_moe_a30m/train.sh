#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data=${1:?Usage: bash qwen3_moe_a30m/train.sh DATA_DIR OUTPUT_DIR [DDP options]}
output=${2:?Usage: bash qwen3_moe_a30m/train.sh DATA_DIR OUTPUT_DIR [DDP options]}
shift 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKEN_CACHE_VALIDATION_DIR=${TOKEN_CACHE_VALIDATION_DIR:-"$output/cache_validation"}
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" \
  "$root/training/src/train_ddp.py" --config "${TRAIN_CONFIG:-$root/qwen3_moe_a30m/training.yaml}" \
  --data-dir "$data" --output-dir "$output" "$@"
