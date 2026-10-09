#!/usr/bin/env bash
# Run on the host: encode published samples on CPU, then train with DDP.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
COMP_DIR="$(dirname -- "$PROJECT_DIR")"
IMAGE=scicomp26:cu121-train-v1
CONFIG_PATH=training/configs/FLOPs_3e17.yaml
GPUS=0,3
STEPS=10
EVAL_EVERY=5
SAVE_EVERY=5
SHM_SIZE=64g
CORPUS_DIR=/home/pretrain10/dolma3_official
DATA_DIR="$COMP_DIR/data/splits_v2-tok32k-smoke-ctx8192-v1"
RUN_NAME="ddp-smoke-$(date +%Y%m%d-%H%M%S)-$$"
RUNS_DIR="$COMP_DIR/runs"
WANDB_ENABLED=false
WANDB_ENTITY=""
WANDB_PROJECT=""
PREPARE_ONLY=false
DRY_RUN=false

usage() {
  cat <<'HELP'
Usage (on the host): bash scripts/start_train.sh [options]

Default: encode 10,000 train / 1,000 dev documents, then train 10 steps
on host GPUs 0,3. Completed token arrays are validated and reused.
Each training invocation creates a new run directory; it does not resume.

  --gpus IDS          Comma-separated host GPU indices (default: 0,3)
  --config PATH       Model YAML relative to the project (default: training/configs/FLOPs_3e17.yaml)
  --steps N           Optimizer steps (default: 10)
  --eval-every N      Validation interval in steps (default: 5)
  --save-every N      Checkpoint interval in steps (default: 5)
  --image NAME        Docker image (default: scicomp26:cu121-train-v1)
  --corpus-dir PATH   Root containing published samples and tokenizer
  --data-dir PATH     Generated smoke dataset directory
  --run-name NAME     New directory name under ../runs
  --runs-dir PATH     Parent directory for training runs (default: ../runs)
  --wandb            Upload metrics to W&B (requires WANDB_API_KEY or wandb login)
  --wandb-entity NAME Override W&B entity (default: echo-lab)
  --wandb-project NAME Override W&B project (default: scicomp26)
  --shm-size SIZE     Docker shared-memory limit (default: 64g)
  --prepare-only     Encode/validate data on CPU, without using GPUs
  --dry-run          Print commands without creating files or containers
  -h, --help         Show this help

Changing --steps still trains on smoke samples, not the full corpus.
HELP
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) need_value "$@"; CONFIG_PATH="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPUS="$2"; shift 2 ;;
    --steps) need_value "$@"; STEPS="$2"; shift 2 ;;
    --eval-every) need_value "$@"; EVAL_EVERY="$2"; shift 2 ;;
    --save-every) need_value "$@"; SAVE_EVERY="$2"; shift 2 ;;
    --image) need_value "$@"; IMAGE="$2"; shift 2 ;;
    --corpus-dir) need_value "$@"; CORPUS_DIR="$2"; shift 2 ;;
    --data-dir) need_value "$@"; DATA_DIR="$2"; shift 2 ;;
    --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --runs-dir) need_value "$@"; RUNS_DIR="$2"; shift 2 ;;
    --wandb) WANDB_ENABLED=true; shift ;;
    --wandb-entity) need_value "$@"; WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-project) need_value "$@"; WANDB_PROJECT="$2"; shift 2 ;;
    --shm-size) need_value "$@"; SHM_SIZE="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1 (see --help)" ;;
  esac
done

[[ ! -f /.dockerenv ]] || fail "Run this script in a host terminal, outside Docker."
[[ "$CONFIG_PATH" != /* ]] || fail "--config must be relative to the project directory"
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "--gpus must look like 0,3"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail "--steps must be a positive integer"
[[ "$EVAL_EVERY" =~ ^[1-9][0-9]*$ ]] || fail "--eval-every must be a positive integer"
[[ "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]] || fail "--save-every must be a positive integer"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || fail "Invalid --run-name"
IFS=, read -r -a GPU_LIST <<< "$GPUS"
NPROC=${#GPU_LIST[@]}
declare -A SEEN_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
  [[ -z "${SEEN_GPUS[$gpu]+present}" ]] || fail "Duplicate GPU index: $gpu"
  SEEN_GPUS[$gpu]=1
done

for cmd in docker realpath tee; do
  command -v "$cmd" >/dev/null || fail "Required command not found: $cmd"
done
CORPUS_DIR="$(realpath -e -- "$CORPUS_DIR")"
DATA_DIR="$(realpath -m -- "$DATA_DIR")"
DATA_PARENT="$(dirname -- "$DATA_DIR")"
DATA_NAME="$(basename -- "$DATA_DIR")"
RUNS_DIR="$(realpath -m -- "$RUNS_DIR")"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
[[ -d "$CORPUS_DIR" ]] || fail "Corpus root is not a directory"
[[ "$DATA_NAME" != / && "$DATA_NAME" != . ]] || fail "Invalid --data-dir"
for file in \
  "$CORPUS_DIR/github_staging/scicomp26_data_v0.1/data/splits_v2/sample_train_10000.jsonl.zst" \
  "$CORPUS_DIR/github_staging/scicomp26_data_v0.1/data/splits_v2/sample_dev_1000.jsonl.zst" \
  "$CORPUS_DIR/tokenizer_v1_bpe32k/tokenizer.json" \
  "$PROJECT_DIR/$CONFIG_PATH" \
  "$PROJECT_DIR/scripts/prepare_smoke_data.py" \
  "$PROJECT_DIR/scripts/train_ddp.py"; do
  [[ -r "$file" ]] || fail "Cannot read: $file"
done
if [[ "$DRY_RUN" == false ]]; then
  docker info >/dev/null 2>&1 || fail "Cannot access Docker; check daemon and account permissions"
  docker image inspect "$IMAGE" >/dev/null || fail "Build or load the image first: $IMAGE"
fi
if [[ "$PREPARE_ONLY" == false ]]; then
  [[ ! -e "$RUN_DIR" ]] || fail "Run directory already exists: $RUN_DIR (use a new --run-name)"
fi

COMMON=(
  --rm --pull=never
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env OMP_NUM_THREADS=1
  --workdir /workspace/comp/scicomp26
  --mount "type=bind,src=$PROJECT_DIR,dst=/workspace/comp/scicomp26,readonly"
)
PREPARE=(
  docker run "${COMMON[@]}"
  --mount "type=bind,src=$CORPUS_DIR,dst=/corpus,readonly"
  --mount "type=bind,src=$DATA_PARENT,dst=/prepared"
  "$IMAGE" python scripts/prepare_smoke_data.py
  --source-dir /corpus/github_staging/scicomp26_data_v0.1/data/splits_v2
  --tokenizer-dir /corpus/tokenizer_v1_bpe32k
  --output-dir "/prepared/$DATA_NAME" --reuse
)
GPU_ARGS=(--runtime=nvidia --gpus "\"device=$GPUS\"" --shm-size "$SHM_SIZE")
WANDB_DOCKER_ARGS=()
WANDB_TRAIN_ARGS=()
if [[ "$WANDB_ENABLED" == true && "$PREPARE_ONLY" == false ]]; then
  WANDB_DOCKER_ARGS=(
    # Host networking makes loopback proxy endpoints reachable inside the container.
    --network host
    --env HTTP_PROXY --env HTTPS_PROXY --env ALL_PROXY --env NO_PROXY
    --env http_proxy --env https_proxy --env all_proxy --env no_proxy
    --env WANDB_MODE=online
    --env WANDB_DIR=/runs
    --env WANDB_CACHE_DIR=/tmp/wandb-cache
    --env WANDB_CONFIG_DIR=/tmp/wandb-config
  )
  # Pass only the variable name so dry-run output never reveals the API key.
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_DOCKER_ARGS+=(--env WANDB_API_KEY)
  elif [[ -r "$HOME/.netrc" ]]; then
    WANDB_DOCKER_ARGS+=(--mount "type=bind,src=$HOME/.netrc,dst=/tmp/.netrc,readonly")
  elif [[ "$DRY_RUN" == true ]]; then
    WANDB_DOCKER_ARGS+=(--env WANDB_API_KEY)
  else
    fail "W&B credentials missing: run wandb login or export WANDB_API_KEY in this terminal"
  fi
  WANDB_TRAIN_ARGS=(--wandb --wandb-name "$RUN_NAME")
  [[ -z "$WANDB_ENTITY" ]] || WANDB_TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  [[ -z "$WANDB_PROJECT" ]] || WANDB_TRAIN_ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
TRAIN=(
  docker run "${COMMON[@]}" "${GPU_ARGS[@]}" "${WANDB_DOCKER_ARGS[@]}"
  --mount "type=bind,src=$DATA_DIR,dst=/data,readonly"
  --mount "type=bind,src=$RUN_DIR,dst=/runs"
  "$IMAGE" torchrun --standalone --nnodes=1 "--nproc-per-node=$NPROC"
  scripts/train_ddp.py --config "$CONFIG_PATH" --data-dir /data --output-dir /runs
  --micro-batch-size 1 --grad-accum-steps 1 --max-steps "$STEPS"
  --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY"
  "${WANDB_TRAIN_ARGS[@]}"
)

print_command() { printf '%q ' "$@"; printf '\n'; }
printf 'Smoke data: %s\n' "$DATA_DIR"
if [[ "$DRY_RUN" == true ]]; then
  print_command "${PREPARE[@]}"
  if [[ "$PREPARE_ONLY" == false ]]; then
    printf 'New run: %s\n' "$RUN_DIR"
    print_command "${TRAIN[@]}"
  fi
  exit 0
fi

mkdir -p -- "$DATA_PARENT"
printf '\n[1/2] Encode samples on CPU, or validate existing arrays\n'
"${PREPARE[@]}"
if [[ "$PREPARE_ONLY" == true ]]; then
  printf 'Data ready: %s\n' "$DATA_DIR"
  exit 0
fi

printf '\n[2/2] Check GPUs, then start %s-process DDP for %s steps\n' "$NPROC" "$STEPS"
docker run "${COMMON[@]}" "${GPU_ARGS[@]}" "$IMAGE" python -c \
  'import sys, torch; n=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count()==n, "GPU count mismatch"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n)])' \
  "$NPROC"
mkdir -p -- "$RUNS_DIR"
mkdir -- "$RUN_DIR"
printf 'Log and checkpoints: %s\n' "$RUN_DIR"
"${TRAIN[@]}" 2>&1 | tee "$RUN_DIR/console.log"
printf 'Completed: %s\n' "$RUN_DIR"
