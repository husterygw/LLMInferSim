#!/bin/bash
# Compatibility wrapper. Prefer scripts/run_bench_suite.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--list" ]; then
  python "$SCRIPT_DIR/bench_cases.py" --list
  exit 0
fi

SUITE="${1:-single_tp1_roofline}"
shift || true

exec bash "$SCRIPT_DIR/run_bench_suite.sh" "$SUITE" "$@"
