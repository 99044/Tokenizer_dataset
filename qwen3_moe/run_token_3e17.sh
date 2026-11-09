#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo 'Usage: bash qwen3_moe/run_token_3e17.sh VARIANT TRAIN_CACHE TOKENIZER_DIR EVAL_CACHE NEW_OUTPUT_DIR' >&2
    exit 2
fi
variant=$1
shift
selector=--variant
if [[ "$variant" =~ ^[0-9]+$ ]]; then
    selector=--width
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
plan_dir=$(mktemp -d)
trap 'rm -f -- "$plan_dir/training.yaml" "$plan_dir/token-plan.json"; rmdir -- "$plan_dir"' EXIT
export TRAIN_CONFIG="$plan_dir/training.yaml"
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
"${PYTHON:-python}" "$root/qwen3_moe/token_sweep.py" "$selector=$variant" --config "$TRAIN_CONFIG" \
    | tee "$plan_dir/token-plan.json"
bash "$root/qwen3_moe/run_all.sh" 3e17 "$@"
cp -- "$plan_dir/token-plan.json" "$4/token-plan.json"
