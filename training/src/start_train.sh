#!/usr/bin/env bash
# Run on the host: prepare raw JSONL when needed, then run Hybrid DDP training.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "$0")/../.." && pwd -P)"
COMP_DIR="$(dirname -- "$PROJECT_DIR")"
IMAGE=scicomp26:submission-env-hybrid-v3
CONFIG=training/configs/FLOPs_3e17.yaml
STAGE=stage1
GPUS=0,3
STEPS=""
MICRO_BATCH_SIZE=""
GRAD_ACCUM_STEPS=""
LEARNING_RATE=""
WARMUP_STEPS=""
STOP_AFTER=""
RESUME=""
BRANCH_FROM=""
EVAL_EVERY=""
SAVE_EVERY=""
SHM_SIZE=64g
DATA_DIR=""
DATA_FORMAT=auto
TOKENIZER_DIR=""
CACHE_DIR=""
VALIDATION_PERMYRIAD=100
ENCODE_BATCH_SIZE=128
PREPROCESS_THREADS=8
RUN_NAME=""
RUNS_DIR="$COMP_DIR/runs"
WANDB_ENABLED=false
WANDB_ENTITY=""
WANDB_PROJECT=""
DRY_RUN=false

usage() {
  cat <<'HELP'
Usage (on the host): bash training/train.sh [options]

Prepare raw JSONL if needed, then launch the branch's Hybrid DDP runner.
Training steps are explicit: dense OLMo3 FLOPs planning does not apply to GDN.

  --config PATH       Model YAML inside this repository (default: training/configs/FLOPs_3e17.yaml)
  --stage NAME        stage1 or stage-b (default: stage1)
  --gpus IDS          Comma-separated host GPU indices (default: 0,3)
  --steps N           Override training.max_steps (null/missing fallback: 20; no automatic FLOPs certification)
  --micro-batch-size N Override training.micro_batch_size (null/missing fallback: 1)
  --grad-accum-steps N  Override training.gradient_accumulation_steps (null/missing fallback: 1)
  --learning-rate LR   Override optimizer.learning_rate (null/missing fallback: 2e-4)
  --warmup-steps N     Override scheduler.warmup_steps (null/missing fallback: 0)
  --stop-after N       Stop at this cumulative step, preserving the LR schedule
  --resume PATH        Restart the same training plan from a native checkpoint
  --branch-from PATH   Start a declared Stage B continuation from a Stage A checkpoint
  --eval-every N       Override logging.eval_every_steps (null/missing fallback: 10)
  --save-every N       Override logging.checkpoint_every_steps (null/missing fallback: 10)
  --image NAME         Docker image (default: scicomp26:submission-env-hybrid-v3)
  --data-dir PATH      Packed dataset, or a directory of raw *.jsonl shards
  --data-format NAME   auto (default), packed, or jsonl
  --tokenizer-dir PATH Existing tokenizer; required for raw JSONL
  --cache-dir PATH     Cache parent (default: ../data/<source-name>-token-cache)
  --validation-permyriad N Raw validation share by ID hash (default: 100 = 1%)
  --encode-batch-size N Documents per tokenization batch (default: 128)
  --preprocess-threads N Tokenizer CPU threads (default: 8)
  --run-name NAME      New output directory under ../runs
  --runs-dir PATH      Output parent (default: ../runs)
  --wandb              Upload metrics to W&B
  --wandb-entity NAME   Override the configured W&B entity
  --wandb-project NAME  Override the configured W&B project
  --shm-size SIZE      Docker shared memory (default: 64g)
  --dry-run            Print commands without creating files or starting containers
  -h, --help           Show this help

The original sampler and loss remain unchanged; is_text masks and document
boundaries are not consumed by this runner. See REPRODUCTION.md before training.
HELP
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --stage) need_value "$@"; STAGE="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPUS="$2"; shift 2 ;;
    --steps) need_value "$@"; STEPS="$2"; shift 2 ;;
    --micro-batch-size) need_value "$@"; MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum-steps) need_value "$@"; GRAD_ACCUM_STEPS="$2"; shift 2 ;;
    --learning-rate) need_value "$@"; LEARNING_RATE="$2"; shift 2 ;;
    --warmup-steps) need_value "$@"; WARMUP_STEPS="$2"; shift 2 ;;
    --stop-after) need_value "$@"; STOP_AFTER="$2"; shift 2 ;;
    --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
    --branch-from) need_value "$@"; BRANCH_FROM="$2"; shift 2 ;;
    --eval-every) need_value "$@"; EVAL_EVERY="$2"; shift 2 ;;
    --save-every) need_value "$@"; SAVE_EVERY="$2"; shift 2 ;;
    --image) need_value "$@"; IMAGE="$2"; shift 2 ;;
    --data-dir) need_value "$@"; DATA_DIR="$2"; shift 2 ;;
    --data-format) need_value "$@"; DATA_FORMAT="$2"; shift 2 ;;
    --tokenizer-dir) need_value "$@"; TOKENIZER_DIR="$2"; shift 2 ;;
    --cache-dir) need_value "$@"; CACHE_DIR="$2"; shift 2 ;;
    --validation-permyriad) need_value "$@"; VALIDATION_PERMYRIAD="$2"; shift 2 ;;
    --encode-batch-size) need_value "$@"; ENCODE_BATCH_SIZE="$2"; shift 2 ;;
    --preprocess-threads) need_value "$@"; PREPROCESS_THREADS="$2"; shift 2 ;;
    --run-name) need_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --runs-dir) need_value "$@"; RUNS_DIR="$2"; shift 2 ;;
    --wandb) WANDB_ENABLED=true; shift ;;
    --wandb-entity) need_value "$@"; WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-project) need_value "$@"; WANDB_PROJECT="$2"; shift 2 ;;
    --shm-size) need_value "$@"; SHM_SIZE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --target-flops|--training-tokens|--global-batch-tokens|--init-from) fail "Use --steps for a fixed plan, --resume to restart it, or --branch-from for declared Stage B continuation. GDN FLOPs planning needs separate accounting." ;;
    *) fail "Unknown option: $1 (see --help)" ;;
  esac
done

[[ ! -f /.dockerenv ]] || fail "Run this script in a host terminal, outside Docker."
[[ "$STAGE" == stage1 || "$STAGE" == stage-b ]] || fail "--stage must be stage1 or stage-b"
[[ -z "$RESUME" || -z "$BRANCH_FROM" ]] || fail "--resume and --branch-from cannot be used together"
if [[ "$STAGE" == stage-b && -z "$RESUME" && -z "$BRANCH_FROM" ]]; then
  fail "stage-b needs --branch-from, or --resume for an interrupted Stage B run"
fi
[[ -z "$BRANCH_FROM" || "$STAGE" == stage-b ]] || fail "--branch-from requires --stage stage-b"
[[ -n "$DATA_DIR" ]] || fail "Training needs an explicit --data-dir"
[[ "$DATA_FORMAT" == auto || "$DATA_FORMAT" == packed || "$DATA_FORMAT" == jsonl ]] || fail "--data-format must be auto, packed or jsonl"
RUN_NAME="${RUN_NAME:-olmo-hybrid-$STAGE-$(date +%Y%m%d-%H%M%S)-$$}"
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "--gpus must look like 0,3"
for value in "$STEPS" "$MICRO_BATCH_SIZE" "$GRAD_ACCUM_STEPS" "$STOP_AFTER" "$ENCODE_BATCH_SIZE" "$PREPROCESS_THREADS"; do
  [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || fail "Steps, tokens and batch sizes must be positive integers"
done
[[ -z "$WARMUP_STEPS" || "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || fail "--warmup-steps must be a nonnegative integer"
[[ -z "$EVAL_EVERY" || "$EVAL_EVERY" =~ ^[1-9][0-9]*$ ]] || fail "--eval-every must be a positive integer"
[[ -z "$SAVE_EVERY" || "$SAVE_EVERY" =~ ^[1-9][0-9]*$ ]] || fail "--save-every must be a positive integer"
[[ "$VALIDATION_PERMYRIAD" =~ ^[1-9][0-9]{0,3}$ ]] || fail "--validation-permyriad must be in 1..9999"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || fail "Invalid --run-name"
IFS=, read -r -a GPU_LIST <<< "$GPUS"
NPROC=${#GPU_LIST[@]}
SEEN_GPUS=,
for gpu in "${GPU_LIST[@]}"; do
  [[ "$SEEN_GPUS" != *",$gpu,"* ]] || fail "Duplicate GPU index: $gpu"
  SEEN_GPUS+="$gpu,"
done

for cmd in docker realpath tee sha256sum; do
  command -v "$cmd" >/dev/null || fail "Required command not found: $cmd"
done
[[ "$CONFIG" == /* ]] || CONFIG="$PROJECT_DIR/$CONFIG"
CONFIG="$(realpath -e -- "$CONFIG")"
[[ "$CONFIG" == "$PROJECT_DIR/"* && -f "$CONFIG" ]] || fail "--config must be a file inside this repository"
CONFIG_CONTAINER="/workspace/comp/scicomp26/${CONFIG#"$PROJECT_DIR/"}"
DATA_DIR="$(realpath -m -- "$DATA_DIR")"
DATA_NAME="$(basename -- "$DATA_DIR")"
RUNS_DIR="$(realpath -m -- "$RUNS_DIR")"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
[[ "$DATA_NAME" != / && "$DATA_NAME" != . ]] || fail "Invalid --data-dir"
[[ -d "$DATA_DIR" ]] || fail "Cannot read data directory: $DATA_DIR"
if [[ "$DATA_FORMAT" == auto ]]; then
  if [[ -r "$DATA_DIR/train.npy" && -r "$DATA_DIR/validation.npy" && -r "$DATA_DIR/metadata.json" ]]; then
    DATA_FORMAT=packed
  else
    DATA_FORMAT=jsonl
  fi
fi
if [[ "$DATA_FORMAT" == jsonl ]]; then
  shopt -s nullglob
  RAW_FILES=("$DATA_DIR"/*.jsonl)
  shopt -u nullglob
  [[ ${#RAW_FILES[@]} -gt 0 ]] || fail "No raw *.jsonl files under $DATA_DIR"
  [[ -n "$TOKENIZER_DIR" ]] || fail "Raw JSONL requires --tokenizer-dir"
  TOKENIZER_DIR="$(realpath -e -- "$TOKENIZER_DIR")"
  for file in tokenizer.json tokenizer_config.json; do
    [[ -r "$TOKENIZER_DIR/$file" ]] || fail "Cannot read tokenizer: $TOKENIZER_DIR/$file"
  done
  RAW_DATA_DIR="$DATA_DIR"
  CACHE_DIR="$(realpath -m -- "${CACHE_DIR:-$COMP_DIR/data/$DATA_NAME-token-cache}")"
else
  for file in train.npy validation.npy metadata.json; do
    [[ -r "$DATA_DIR/$file" ]] || fail "Cannot read packed data: $DATA_DIR/$file"
  done
fi
CHECKPOINT_DOCKER_ARGS=()
CHECKPOINT_TRAIN_ARGS=()
if [[ -n "$RESUME" || -n "$BRANCH_FROM" ]]; then
  CHECKPOINT_SOURCE="${RESUME:-$BRANCH_FROM}"
  CHECKPOINT="$(realpath -e -- "$CHECKPOINT_SOURCE")"
  [[ -f "$CHECKPOINT" && -r "$CHECKPOINT" ]] || fail "Checkpoint must be a readable file"
  CHECKPOINT_DOCKER_ARGS=(--mount "type=bind,src=$CHECKPOINT,dst=/checkpoints/input.pt,readonly")
  if [[ -n "$RESUME" ]]; then
    CHECKPOINT_TRAIN_ARGS=(--resume /checkpoints/input.pt)
  else
    CHECKPOINT_TRAIN_ARGS=(--branch-from /checkpoints/input.pt)
  fi
fi
if [[ "$DRY_RUN" == false ]]; then
  docker info >/dev/null 2>&1 || fail "Cannot access Docker; check daemon and account permissions"
  docker image inspect "$IMAGE" >/dev/null || fail "Build or load the image first: $IMAGE"
fi
if [[ -e "$RUN_DIR" ]]; then
  [[ -n "$RESUME" && -d "$RUN_DIR" ]] || fail "Run directory already exists: $RUN_DIR (use --resume or a new --run-name)"
fi

COMMON=(
  --rm --pull=never
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env PYTHONDONTWRITEBYTECODE=1
  --env TRITON_F32_DEFAULT=ieee
  --env PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch-kernel-cache
  --env OMP_NUM_THREADS=1
  --workdir /workspace/comp/scicomp26
  --mount "type=bind,src=$PROJECT_DIR,dst=/workspace/comp/scicomp26,readonly"
)
GPU_ARGS=(--runtime=nvidia --gpus "\"device=$GPUS\"" --shm-size "$SHM_SIZE")
WANDB_DOCKER_ARGS=()
WANDB_TRAIN_ARGS=()
if [[ "$WANDB_ENABLED" == true ]]; then
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
PREPARE=()
if [[ "$DATA_FORMAT" == jsonl ]]; then
  PREPARE=(
    docker run "${COMMON[@]}" --network none
    --env "RAYON_NUM_THREADS=$PREPROCESS_THREADS" --env TOKENIZERS_PARALLELISM=true
    --mount "type=bind,src=$RAW_DATA_DIR,dst=/raw,readonly"
    --mount "type=bind,src=$TOKENIZER_DIR,dst=/tokenizer,readonly"
    --mount "type=bind,src=$CACHE_DIR,dst=/cache"
    "$IMAGE" python training/src/prepare_token_cache.py
    --source-root /raw --tokenizer /tokenizer --output /cache
    --source-label "$RAW_DATA_DIR" --tokenizer-label "$TOKENIZER_DIR"
    --model-config "$CONFIG_CONTAINER"
    --validation-permyriad "$VALIDATION_PERMYRIAD" --encode-batch-size "$ENCODE_BATCH_SIZE"
  )
  if [[ "$DRY_RUN" == true ]]; then
    DATA_DIR="$CACHE_DIR/<cache-key>"
  else
    mkdir -p -- "$CACHE_DIR"
    printf 'Preparing or reusing token cache from %s\n' "$RAW_DATA_DIR"
    CACHE_KEY="$("${PREPARE[@]}")"
    [[ "$CACHE_KEY" =~ ^[0-9a-f]{64}$ ]] || fail "Preprocessor did not return a valid cache key"
    DATA_DIR="$CACHE_DIR/$CACHE_KEY"
    for file in train.npy validation.npy metadata.json; do
      [[ -r "$DATA_DIR/$file" ]] || fail "Preprocessing output missing: $DATA_DIR/$file"
    done
  fi
fi
VALIDATION_KEY="$(printf '%s' "$DATA_DIR" | sha256sum)"
VALIDATION_DIR="$COMP_DIR/data/token-cache-validation/${VALIDATION_KEY%% *}"
DATA_REFERENCE_MOUNTS=()
for token_file in train.npy validation.npy; do
  if [[ -L "$DATA_DIR/$token_file" ]]; then
    resolved_token_file="$(realpath -e -- "$DATA_DIR/$token_file")"
    DATA_REFERENCE_MOUNTS+=(--mount "type=bind,src=$resolved_token_file,dst=$resolved_token_file,readonly")
  fi
done
VALIDATE=(
  docker run "${COMMON[@]}" --network none
  --env TOKEN_CACHE_VALIDATION_DIR=/validation
  --mount "type=bind,src=$DATA_DIR,dst=/data,readonly"
  "${DATA_REFERENCE_MOUNTS[@]}"
  --mount "type=bind,src=$VALIDATION_DIR,dst=/validation"
  "$IMAGE" python training/src/token_cache_validation.py
  --data-dir /data --model-config "$CONFIG_CONTAINER"
)
TRAIN=(
  docker run "${COMMON[@]}" "${GPU_ARGS[@]}" "${WANDB_DOCKER_ARGS[@]}"
  "${CHECKPOINT_DOCKER_ARGS[@]}"
  --env TOKEN_CACHE_VALIDATION_DIR=/validation
  --mount "type=bind,src=$VALIDATION_DIR,dst=/validation,readonly"
  --mount "type=bind,src=$DATA_DIR,dst=/data,readonly"
  "${DATA_REFERENCE_MOUNTS[@]}"
  --mount "type=bind,src=$RUN_DIR,dst=/runs"
  "$IMAGE" torchrun --standalone --nnodes=1 "--nproc-per-node=$NPROC"
  training/src/train_ddp.py --config "$CONFIG_CONTAINER" --data-dir /data --output-dir /runs
  "${WANDB_TRAIN_ARGS[@]}"
  "${CHECKPOINT_TRAIN_ARGS[@]}"
)
[[ -z "$STEPS" ]] || TRAIN+=(--max-steps "$STEPS")
[[ -z "$MICRO_BATCH_SIZE" ]] || TRAIN+=(--micro-batch-size "$MICRO_BATCH_SIZE")
[[ -z "$EVAL_EVERY" ]] || TRAIN+=(--eval-every "$EVAL_EVERY")
[[ -z "$SAVE_EVERY" ]] || TRAIN+=(--save-every "$SAVE_EVERY")
[[ -z "$GRAD_ACCUM_STEPS" ]] || TRAIN+=(--grad-accum-steps "$GRAD_ACCUM_STEPS")
[[ -z "$LEARNING_RATE" ]] || TRAIN+=(--learning-rate "$LEARNING_RATE")
[[ -z "$WARMUP_STEPS" ]] || TRAIN+=(--warmup-steps "$WARMUP_STEPS")
[[ -z "$STOP_AFTER" ]] || TRAIN+=(--stop-after "$STOP_AFTER")

print_command() { printf '%q ' "$@"; printf '\n'; }
printf 'Stage: %s; config: %s; data: %s\n' "$STAGE" "$CONFIG" "$DATA_DIR"
if [[ "$DRY_RUN" == true ]]; then
  printf 'New run: %s\n' "$RUN_DIR"
  if [[ "$DATA_FORMAT" == jsonl ]]; then
    printf 'Preprocessing command (cache-key is resolved before training):\n'
    print_command "${PREPARE[@]}"
  fi
  printf 'Data validation command (before GPU launch):\n'
  print_command "${VALIDATE[@]}"
  print_command "${TRAIN[@]}"
  exit 0
fi

mkdir -p -- "$VALIDATION_DIR"
printf 'Checking token cache validation records\n'
"${VALIDATE[@]}"
printf '\nCheck GPUs, then start %s-process DDP (%s plan)\n' "$NPROC" "$STAGE"
docker run "${COMMON[@]}" "${GPU_ARGS[@]}" "$IMAGE" python -c \
  'import sys, torch; n=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count()==n, "GPU count mismatch"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n)])' \
  "$NPROC"
mkdir -p -- "$RUNS_DIR"
if [[ ! -d "$RUN_DIR" ]]; then
  mkdir -- "$RUN_DIR"
fi
printf 'Log and checkpoints: %s\n' "$RUN_DIR"
if [[ -n "$RESUME" ]]; then
  "${TRAIN[@]}" 2>&1 | tee -a "$RUN_DIR/console.log"
else
  "${TRAIN[@]}" 2>&1 | tee "$RUN_DIR/console.log"
fi
printf 'Completed: %s\n' "$RUN_DIR"
