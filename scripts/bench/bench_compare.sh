#!/bin/bash
# Thin wrapper: activate conda env, then delegate to bench_compare.py.
# The benchmark matrix lives in bench_cases.py; this executor only runs cases.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="${CONDA_ENV-llm_sim}"

if [ -n "$CONDA_ENV" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -z "$CONDA_BASE" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
  fi
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

exec python3 "$SCRIPT_DIR/bench_compare.py" "$@"
