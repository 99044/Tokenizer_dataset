#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
    echo 'Usage: bash qwen3_moe_a30m/run_all.sh TRAIN_CACHE TOKENIZER_DIR EVAL_CACHE NEW_OUTPUT_DIR' >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
train_cache=$1
tokenizer=$2
eval_cache=$3
output=$4
python_bin=${PYTHON:-python}
export TRAIN_CONFIG=${TRAIN_CONFIG:-"$root/qwen3_moe_a30m/training.yaml"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 2
fi

plan=$(PYTHONPATH="$root/training/src:$root/evaluation${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" - "$TRAIN_CONFIG" "$train_cache" "$tokenizer" "$eval_cache" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import yaml

from packed_cache import load_packed_cache, validate_packed_identity
from training_data import normalize_data_metadata

config_path, train, tokenizer, evaluation = map(Path, sys.argv[1:])
settings = yaml.safe_load(config_path.read_text())
model = settings["model"]["hf_config"]
identity = dict(
    vocab_size=model["vocab_size"],
    eos_token_id=model["eos_token_id"],
    tokenizer_json_sha256=settings["data"]["tokenizer_json_sha256"],
)
metadata = json.loads((train / "metadata.json").read_text())
metadata = normalize_data_metadata(metadata, model["max_position_embeddings"])
if metadata.get("tokenizer_details", {}).get("tokenizer_json_sha256") != identity["tokenizer_json_sha256"]:
    raise ValueError("Training cache tokenizer does not match the model")
if hashlib.sha256((tokenizer / "tokenizer.json").read_bytes()).hexdigest() != identity["tokenizer_json_sha256"]:
    raise ValueError("Export tokenizer does not match the model")
for filename in ("train.npy", "train_is_text.npy", "validation.npy", "validation_is_text.npy"):
    if not (train / filename).is_file():
        raise FileNotFoundError(train / filename)
tokens, _, cache = load_packed_cache(evaluation)
if cache["format"] != "npy":
    raise ValueError("The pipeline expects an NPY evaluation cache")
validate_packed_identity(cache, identity, model["vocab_size"], tokens=tokens)
print(settings["training"]["max_steps"], settings["submission"]["data_parallel_size"])
PY
)
read -r final_step ranks <<< "$plan"
visible=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 "$((ranks-1))")}
eval_gpu=${EVAL_GPU:-${visible%%,*}}
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"

CUDA_VISIBLE_DEVICES="$visible" NPROC_PER_NODE="$ranks" \
    bash "$root/qwen3_moe_a30m/train.sh" "$train_cache" "$output/train" \
    2>&1 | tee "$output/train.log"

CUDA_VISIBLE_DEVICES= "$python_bin" "$root/training/export_hf.py" \
    --checkpoint "$output/train/step-$final_step.pt" --tokenizer-dir "$tokenizer" \
    --output-dir "$output/hf-model" --validation-device cpu \
    2>&1 | tee "$output/export.log"

CUDA_VISIBLE_DEVICES="$eval_gpu" NPROC_PER_NODE=1 BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" \
    SEQUENCE_LENGTH=8192 FUSION_MODE=none \
    bash "$root/evaluation/eval.sh" "$eval_cache" "$output/hf-model" "$output/eval-result.json" \
    2>&1 | tee "$output/eval.log"
