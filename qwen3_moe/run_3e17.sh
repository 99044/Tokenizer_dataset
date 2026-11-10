#!/usr/bin/env bash
# Train, export, evaluate, and audit the 3e17 Dense/Hi-MoE recipe.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
    echo 'Usage: bash qwen3_moe/run_3e17.sh TRAIN_CACHE TOKENIZER_DIR EVAL_CACHE [NEW_OUTPUT_DIR]' >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TRAIN_CONFIG="$root/qwen3_moe/configs/dense-prefix/3e17/training.yaml"
output=${4:-"$root/runs/qwen3-himoe-dense4-3e17-$(date +%Y%m%d-%H%M%S)-$$"}
echo "3e17 run output: $output"
exec bash "$root/qwen3_moe/run_all.sh" 3e17 "$1" "$2" "$3" "$output"
