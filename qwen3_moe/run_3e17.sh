#!/usr/bin/env bash
# Run the paper-scaling 3e17 recipe: train, export, evaluate, then audit FLOPs.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
    echo 'Usage: bash qwen3_moe/run_3e17.sh TRAIN_CACHE TOKENIZER_DIR EVAL_CACHE [NEW_OUTPUT_DIR]' >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
variant=${MOE_EXPERT_VARIANT:-e24-k2-w512}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
output=${4:-"$root/runs/qwen3-moe-paper-3e17-$variant-$(date +%Y%m%d-%H%M%S)-$$"}
config_dir=$(mktemp -d)
trap 'rm -f -- "$config_dir/training.yaml"; rmdir -- "$config_dir"' EXIT
export TRAIN_CONFIG="$config_dir/training.yaml"
"${PYTHON:-python}" "$root/qwen3_moe/expert_ablation.py" plan "$variant" "$TRAIN_CONFIG"
echo "3e17 run output: $output"
bash "$root/qwen3_moe/run_all.sh" 3e17 "$1" "$2" "$3" "$output"
