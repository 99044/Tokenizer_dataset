#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dataset_path=${1:?Usage: bash eval.sh /path/to/token-cache-or-dataset.jsonl /path/to/hf-model [output.json]}
model_path=${2:?Usage: bash eval.sh /path/to/token-cache-or-dataset.jsonl /path/to/hf-model [output.json]}
output=${3:-${EVAL_OUTPUT:-result.json}}

python_cmd=${PYTHON:-python}
nproc_per_node=${NPROC_PER_NODE:-1}
batch_size=${BATCH_SIZE:-32}
sequence_length=${SEQUENCE_LENGTH:-8192}
fusion_mode=${FUSION_MODE:-ffn_rmsnorm}

if [[ -d "$dataset_path" ]]; then
    data_args=(--packed-cache-dir "$dataset_path")
elif [[ -f "$dataset_path" ]]; then
    data_args=(--dataset-path "$dataset_path")
else
    echo "Token cache directory or dataset JSONL not found" >&2
    exit 2
fi
if [[ ! -d "$model_path" ]]; then
    echo "HF model directory not found" >&2
    exit 2
fi
if [[ ! "$nproc_per_node" =~ ^[0-9]+$ ]] || (( nproc_per_node < 1 )); then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
if [[ ! "$batch_size" =~ ^[0-9]+$ ]] || (( batch_size < 1 )); then
    echo "BATCH_SIZE must be a positive integer" >&2
    exit 2
fi
if [[ "$fusion_mode" != none && "$fusion_mode" != ffn_rmsnorm ]]; then
    echo "FUSION_MODE must be none or ffn_rmsnorm" >&2
    exit 2
fi
if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 2
fi

"$python_cmd" -m torch.distributed.run --standalone --nnodes=1 --local-addr=127.0.0.1 \
    --nproc-per-node="$nproc_per_node" "$script_dir/eval.py" \
    "${data_args[@]}" --model-path "$model_path" \
    --batch-size "$batch_size" --sequence-length "$sequence_length" \
    --fusion "$fusion_mode" --output "$output"
