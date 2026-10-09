#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dataset_path=${1:?Usage: bash eval.sh /path/to/dataset.jsonl /path/to/model}
model_path=${2:?Usage: bash eval.sh /path/to/dataset.jsonl /path/to/model}

# Editable evaluation parameters. Activate the intended Python environment first.
# Sequences per GPU; organizers increase this to the largest stable batch size.
batch_size=1
sequence_length=8192
output=result.json

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    "$script_dir/eval.py" \
    --dataset-path "$dataset_path" \
    --model-path "$model_path" \
    --batch-size "$batch_size" \
    --sequence-length "$sequence_length" \
    --output "$output"
