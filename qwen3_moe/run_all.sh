#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo 'Usage: bash qwen3_moe/run_all.sh TIER TRAIN_CACHE TOKENIZER_DIR EVAL_CACHE NEW_OUTPUT_DIR' >&2
    exit 2
fi
tier=$1
shift
case "$tier" in
    3e17) default_eval_batch=8 ;;
    1e18) default_eval_batch=4 ;;
    3e18|1e19) default_eval_batch=1 ;;
    *) echo "Unknown FLOPs tier: $tier" >&2; exit 2 ;;
esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TRAIN_CONFIG=${TRAIN_CONFIG:-"$root/qwen3_moe/configs/$tier/training.yaml"}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$default_eval_batch}
export EVAL_FUSION_MODE=${EVAL_FUSION_MODE:-rmsnorm}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TRITON_F32_DEFAULT=${TRITON_F32_DEFAULT:-ieee}
if [[ ! "$EVAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'EVAL_BATCH_SIZE must be a positive integer' >&2
    exit 2
fi
if [[ "$EVAL_FUSION_MODE" != none && "$EVAL_FUSION_MODE" != rmsnorm ]]; then
    echo 'Qwen3 MoE EVAL_FUSION_MODE must be none or rmsnorm' >&2
    exit 2
fi
PYTHONPATH="$root/training/src${PYTHONPATH:+:$PYTHONPATH}" \
"${PYTHON:-python}" - "$TRAIN_CONFIG" "$tier" "$CUDA_VISIBLE_DEVICES" <<'PY'
from pathlib import Path
import sys
import yaml
from qwen3_moe_token_rules import validate_qwen3_moe_settings

settings = yaml.safe_load(Path(sys.argv[1]).read_text())
validate_qwen3_moe_settings(settings, check_runtime=True)
devices = sys.argv[3].split(",")
if settings["submission"]["budget_tier"] != sys.argv[2]:
    raise ValueError("Requested tier disagrees with TRAIN_CONFIG")
if len(devices) != settings["submission"]["data_parallel_size"] or len(set(devices)) != len(devices):
    raise ValueError("Visible GPU count differs from the fixed training plan")
PY
bash "$root/qwen3_moe_a30m/run_all.sh" "$@"
cp -- "$TRAIN_CONFIG" "$4/training-config.yaml"
"${PYTHON:-python}" "$root/training/audit_qwen3_moe_flops.py" \
    --config "$4/training-config.yaml" --metrics "$4/train/metrics.jsonl" --output "$4/flops-report.json"
