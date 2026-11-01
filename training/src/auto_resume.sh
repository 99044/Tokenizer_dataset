#!/usr/bin/env bash
# Restart an interrupted start_train.sh run from its latest completed checkpoint.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: auto_resume.sh --run-dir PATH --max-steps N [--max-restarts N]
                      [--max-stalls N] [--retry-delay SECONDS] -- START_TRAIN_COMMAND...

The command must pass a fixed --run-name and --runs-dir to start_train.sh.
Do not pass --resume, --branch-from, or --stop-after; this script manages resuming.
Keep the model, data, GPU count, and training options unchanged across attempts.
HELP
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
run_dir=
max_steps=
max_restarts=20
max_stalls=2
retry_delay=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir|--max-steps|--max-restarts|--max-stalls|--retry-delay)
      [[ $# -ge 2 ]] || fail "$1 needs a value"
      case "$1" in
        --run-dir) run_dir=$2 ;;
        --max-steps) max_steps=$2 ;;
        --max-restarts) max_restarts=$2 ;;
        --max-stalls) max_stalls=$2 ;;
        --retry-delay) retry_delay=$2 ;;
      esac
      shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

[[ -n "$run_dir" && -n "$max_steps" && $# -gt 0 ]] || { usage >&2; exit 1; }
[[ "$max_steps" =~ ^[1-9][0-9]*$ ]] || fail "--max-steps must be positive"
[[ "$max_restarts" =~ ^[0-9]+$ ]] || fail "--max-restarts must be nonnegative"
[[ "$max_stalls" =~ ^[1-9][0-9]*$ ]] || fail "--max-stalls must be positive"
[[ "$retry_delay" =~ ^[0-9]+$ ]] || fail "--retry-delay must be nonnegative"
if [[ "${1##*/}" == bash && $# -ge 2 ]]; then
  [[ "${2##*/}" == start_train.sh ]] || fail "Command must run start_train.sh"
else
  [[ "${1##*/}" == start_train.sh ]] || fail "Command must run start_train.sh"
fi
command_args=("$@")
command_run_name=
command_runs_dir=
for ((index = 0; index < ${#command_args[@]}; index++)); do
  arg=${command_args[index]}
  case "$arg" in
    --resume|--branch-from|--stop-after) fail "$arg is managed by auto_resume.sh" ;;
    --run-name)
      (( index + 1 < ${#command_args[@]} )) || fail "--run-name needs a value"
      command_run_name=${command_args[index+1]} ;;
    --runs-dir)
      (( index + 1 < ${#command_args[@]} )) || fail "--runs-dir needs a value"
      command_runs_dir=${command_args[index+1]} ;;
  esac
done
[[ -n "$command_run_name" && -n "$command_runs_dir" ]] || fail "Command needs fixed --run-name and --runs-dir"
command -v flock >/dev/null || fail "flock is required on the training host"

run_dir=${run_dir%/}
[[ -n "$run_dir" && "$run_dir" != / ]] || fail "Invalid --run-dir"
[[ "$run_dir" == "${command_runs_dir%/}/$command_run_name" ]] ||
  fail "--run-dir must match the command's --runs-dir/--run-name"
mkdir -p -- "$(dirname -- "$run_dir")"
exec 9>"$run_dir.auto-resume.lock"
flock -n 9 || fail "Another auto-resume process is managing $run_dir"
status_file="$run_dir.auto-resume.tsv"
final_checkpoint="$run_dir/step-$max_steps.pt"

record() { printf '%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$status_file"; }

latest_checkpoint() {
  local path name step best_step=0 best=
  shopt -s nullglob
  for path in "$run_dir"/step-*.pt; do
    [[ -s "$path" ]] || continue
    name=${path##*/}
    step=${name#step-}
    step=${step%.pt}
    [[ "$step" =~ ^[1-9][0-9]*$ ]] || continue
    if (( step > best_step && step <= max_steps )); then
      best_step=$step
      best=$path
    fi
  done
  shopt -u nullglob
  printf '%s' "$best"
}

if [[ -s "$final_checkpoint" ]]; then
  record "COMPLETE_ALREADY step=$max_steps"
  exit 0
fi

restarts=0
stalls=0
while :; do
  checkpoint=$(latest_checkpoint)
  if (( restarts > 0 )) && [[ -z "$checkpoint" ]]; then
    record "STOP no_checkpoint_to_resume"
    exit 1
  fi
  if [[ -n "$checkpoint" ]]; then
    record "START resume=${checkpoint##*/} restart=$restarts"
    set +e
    "$@" --resume "$checkpoint"
    rc=$?
    set -e
  else
    record "START from_scratch"
    set +e
    "$@"
    rc=$?
    set -e
  fi

  if [[ -s "$final_checkpoint" ]]; then
    record "COMPLETE step=$max_steps train_exit=$rc"
    exit 0
  fi

  next_checkpoint=$(latest_checkpoint)
  record "INTERRUPTED train_exit=$rc latest=${next_checkpoint##*/}"
  if [[ -z "$next_checkpoint" ]]; then
    record "STOP no_checkpoint_to_resume"
    exit 1
  fi
  if [[ "$next_checkpoint" == "$checkpoint" ]]; then
    stalls=$((stalls + 1))
  else
    stalls=0
  fi
  if (( stalls >= max_stalls )); then
    record "STOP no_progress_after_${stalls}_attempts"
    exit 1
  fi
  if (( restarts >= max_restarts )); then
    record "STOP max_restarts=$max_restarts"
    exit 1
  fi
  restarts=$((restarts + 1))
  sleep "$retry_delay"
done
