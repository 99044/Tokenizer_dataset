#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
output_dir=""
resume=false
evaluate_only=false
args=()

while (($#)); do
  case "$1" in
    --output-dir)
      if (($# < 2)); then
        printf 'Missing value for --output-dir\n' >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --resume) resume=true; shift ;;
    --evaluate-only) evaluate_only=true; args+=("$1"); shift ;;
    --config|--config=*)
      printf 'This launcher selects the four tier configs; omit --config.\n' >&2
      exit 2
      ;;
    -h|--help)
      printf '%s\n' \
        'Usage: bash run_submission_all.sh --output-dir ROOT [run_jd.py options]' \
        'Runs 3e17, 1e18, 3e18, 1e19 sequentially, each in ROOT/TIER.' \
        '--resume resumes existing tier directories and starts missing tiers.' \
        'Other options match training/run_jd.py, including --dry-run.'
      exit 0
      ;;
    *) args+=("$1"); shift ;;
  esac
done

if [[ -z "$output_dir" ]]; then
  printf 'Required: --output-dir ROOT (each tier uses ROOT/TIER)\n' >&2
  exit 2
fi
if "$resume" && "$evaluate_only"; then
  printf -- '--resume and --evaluate-only cannot be combined\n' >&2
  exit 2
fi

for tier in 3e17 1e18 3e18 1e19; do
  run_dir="${output_dir%/}/$tier"
  resume_args=()
  if "$resume" && [[ -d "$run_dir" ]]; then
    resume_args=(--resume)
  fi
  printf '\nRunning %s: %s\n' "$tier" "$run_dir"
  "${PYTHON:-python3}" "$root/training/run_jd.py" \
    --config "$root/training/configs/FLOPs_${tier}.yaml" \
    "${args[@]}" --output-dir "$run_dir" "${resume_args[@]}"
done
