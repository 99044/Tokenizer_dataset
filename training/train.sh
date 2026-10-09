#!/usr/bin/env bash
# Common host entrypoint; defaults to Stage 1 from random initialization.
set -euo pipefail
SUBMISSION_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
exec bash "$SUBMISSION_DIR/training/src/start_train.sh" "$@"
