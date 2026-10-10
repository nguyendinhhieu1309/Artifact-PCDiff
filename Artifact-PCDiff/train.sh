#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_CMD="$(printenv PYTHON 2>/dev/null || true)"
if [[ -z "$PYTHON_CMD" ]]; then
  PYTHON_CMD="python"
fi

"$PYTHON_CMD" -m Artifact-PCDiff.train_completion \
  --path ../data/precol/train.csv \
  --dataset precol \
  --bs 8 \
  --workers "${TRAIN_WORKERS:-0}" \
  --niter 4000 \
  --num_points 9012 \
  --num_nn 820 \
  --lr 1e-4 \
  --manualSeed 42 \
  "$@"
