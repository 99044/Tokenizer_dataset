#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
exec "${PYTHON:-python3}" "$root/training/run_jd.py" \
  --config "$root/training/configs/FLOPs_1e19.yaml" "$@"
