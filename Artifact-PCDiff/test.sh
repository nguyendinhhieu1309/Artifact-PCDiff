#!/usr/bin/env bash
set -euo pipefail

CALLER_DIR="$PWD"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

if [[ $# -gt 0 ]]; then
  MODEL_ARG="$1"
  shift
else
  MODEL_ARG="$(printenv MODEL_PATH 2>/dev/null || true)"
fi

if [[ -z "$MODEL_ARG" ]]; then
  echo "Usage: bash test.sh <checkpoint.pth> [test_completion.py options...]" >&2
  echo "Or set MODEL_PATH to an existing checkpoint." >&2
  exit 2
fi

# Accept Windows drive paths when called from Git Bash. WSL users should pass
# a mounted POSIX path such as /mnt/c/Users/... instead.
case "$MODEL_ARG" in
  [A-Za-z]:*)
    if ! command -v cygpath >/dev/null 2>&1; then
      echo "Use a POSIX path in WSL (for example /mnt/c/Users/...)." >&2
      exit 2
    fi
    MODEL_ARG="$(cygpath -u "$MODEL_ARG")"
    ;;
  /*) ;;
  *) MODEL_ARG="$CALLER_DIR/$MODEL_ARG" ;;
esac

if [[ ! -f "$MODEL_ARG" ]]; then
  echo "Checkpoint not found: $MODEL_ARG" >&2
  exit 2
fi

cd "$SCRIPT_DIR"
EVAL_PATH="$(printenv EVAL_PATH 2>/dev/null || true)"
if [[ -z "$EVAL_PATH" ]]; then
  EVAL_PATH="Artifact-PCDiff/results"
fi
PYTHON_CMD="$(printenv PYTHON 2>/dev/null || true)"
if [[ -z "$PYTHON_CMD" ]]; then
  PYTHON_CMD="python"
fi

"$PYTHON_CMD" -m Artifact-PCDiff.test_completion \
  --path ../data/precol/test.csv \
  --dataset precol \
  --model "$MODEL_ARG" \
  --eval_path "$EVAL_PATH" \
  --bs "${TEST_BS:-1}" \
  --workers "${TEST_WORKERS:-0}" \
  "$@"

if [[ "$(printenv COMPUTE_ARTIFACT_METRICS 2>/dev/null || true)" == "1" ]]; then
  "$PYTHON_CMD" -m Artifact-PCDiff.evaluate_artifact_metrics \
    --csv ../data/precol/test.csv \
    --results-dir "$EVAL_PATH/syn" \
    --output-csv "$EVAL_PATH/artifact_metrics.csv"
fi
