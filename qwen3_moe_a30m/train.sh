#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data=${1:?Usage: bash qwen3_moe_a30m/train.sh DATA_DIR OUTPUT_DIR [DDP options]}
output=${2:?Usage: bash qwen3_moe_a30m/train.sh DATA_DIR OUTPUT_DIR [DDP options]}
shift 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [[ -z "${TOKEN_CACHE_VALIDATION_DIR:-}" ]]; then
    if [[ -d "$output/cache_validation" ]]; then
        # Resume launches can reuse records created by older versions.
        TOKEN_CACHE_VALIDATION_DIR="$output/cache_validation"
    elif [[ -w "$data" && ( ! -e "$data/.validation" || -w "$data/.validation" ) ]]; then
        TOKEN_CACHE_VALIDATION_DIR="$data/.validation"
    else
        data_key=$(realpath -e -- "$data" | sha256sum)
        output_parent=$(dirname -- "$(realpath -m -- "$output")")
        TOKEN_CACHE_VALIDATION_DIR="$output_parent/.token-cache-validation/${data_key%% *}"
    fi
fi
export TOKEN_CACHE_VALIDATION_DIR
python_bin=${PYTHON:-python}
config=${TRAIN_CONFIG:-"$root/qwen3_moe_a30m/training.yaml"}
ranks=${NPROC_PER_NODE:-}
if [[ -z "$ranks" ]]; then
    ranks=$("$python_bin" -c 'import sys,yaml; from pathlib import Path; print(yaml.safe_load(Path(sys.argv[1]).read_text()).get("submission", {}).get("data_parallel_size", 4))' "$config")
fi
exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node="$ranks" \
  "$root/training/src/train_ddp.py" --config "$config" \
  --data-dir "$data" --output-dir "$output" "$@"
