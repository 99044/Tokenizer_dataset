#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
output_dir=""
resume=false
args=()

while (($#)); do
  case "$1" in
    --output-dir)
      if (($# < 2)) || [[ -z "$2" ]]; then
        printf 'Missing value for --output-dir\n' >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --resume) resume=true; shift ;;
    --config|--config=*)
      printf 'This launcher selects its configs; omit --config.\n' >&2
      exit 2
      ;;
    -h|--help)
      printf '%s\n' \
        'Usage: bash run_submission_from_1e18.sh --output-dir ROOT [run_jd.py options]' \
        'Runs 1e18, 3e18, 1e19, 1e19_1 sequentially, each in ROOT/TIER.' \
        '--resume continues existing tier directories and starts missing tiers.' \
        'Each tier trains, exports HF, and evaluates before the next tier starts.'
      exit 0
      ;;
    *) args+=("$1"); shift ;;
  esac
done

if [[ -z "$output_dir" ]]; then
  printf 'Required: --output-dir ROOT (each tier uses ROOT/TIER)\n' >&2
  exit 2
fi

for tier in 1e18 3e18 1e19 1e19_1; do
  run_dir="${output_dir%/}/$tier"
  if [[ -e "$run_dir" ]]; then
    if ! "$resume"; then
      printf 'Output exists: %s; pass --resume to continue it.\n' "$run_dir" >&2
      exit 2
    fi
  fi
  printf '\nRunning %s: %s\n' "$tier" "$run_dir"
  if [[ -e "$run_dir" ]]; then
    "${PYTHON:-python3}" "$root/training/run_jd.py" \
      --config "$root/training/configs/FLOPs_${tier}.yaml" \
      "${args[@]}" --output-dir "$run_dir" --resume
  else
    "${PYTHON:-python3}" "$root/training/run_jd.py" \
      --config "$root/training/configs/FLOPs_${tier}.yaml" \
      "${args[@]}" --output-dir "$run_dir"
  fi
done
