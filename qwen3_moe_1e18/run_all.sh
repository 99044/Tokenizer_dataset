#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TRAIN_CONFIG="$root/qwen3_moe_1e18/training.yaml"
exec bash "$root/qwen3_moe_a30m/run_all.sh" "$@"
