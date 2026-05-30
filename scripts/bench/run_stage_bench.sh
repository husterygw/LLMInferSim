#!/bin/bash
# Compatibility wrapper. Prefer scripts/run_bench_suite.sh with semantic suite names.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-A}"
shift || true

case "$STAGE" in
  A) SUITE="single_tp1_roofline" ;;
  B) SUITE="tp_comm_sweep" ;;
  C) SUITE="batch_tp1_sweep" ;;
  D) SUITE="tp_batch_sweep" ;;
  E) SUITE="multi_model_regression" ;;
  *) SUITE="$STAGE" ;;
esac

echo ">>> run_stage_bench.sh is a compatibility wrapper."
echo ">>> $STAGE -> $SUITE"
exec bash "$SCRIPT_DIR/run_bench_suite.sh" "$SUITE" "$@"
