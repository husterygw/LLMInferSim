#!/bin/bash
# Generate and run a semantic benchmark suite.

set -e

source /data1/home/ygw268/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-llm_sim}" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE="${1:-single_tp1_roofline}"
shift || true

case "$SUITE" in
  A) SUITE="single_tp1_roofline" ;;
  B) SUITE="tp_comm_sweep" ;;
  C) SUITE="batch_tp1_sweep" ;;
  D) SUITE="tp_batch_sweep" ;;
  E) SUITE="multi_model_regression" ;;
esac

OUT_ROOT="${OUT_ROOT:-/tmp/llm_infer_sim_bench}"
CASES_JSONL="${CASES_JSONL:-$OUT_ROOT/${SUITE}_cases.jsonl}"
DRY_RUN_FLAG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN_FLAG="--dry-run"; shift ;;
    --filter-case) FILTER_CASE="$2"; shift 2 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT"

case_args=(--suite "$SUITE" --out "$CASES_JSONL")
if [ -n "${FILTER_CASE:-}" ]; then
  case_args+=(--filter-case "$FILTER_CASE")
fi

echo ">>> suite=$SUITE"
echo ">>> out=$OUT_ROOT"
python "$SCRIPT_DIR/bench_cases.py" "${case_args[@]}"

bash "$SCRIPT_DIR/bench_compare.sh" \
  --cases "$CASES_JSONL" \
  --out "$OUT_ROOT" \
  $DRY_RUN_FLAG

echo
echo ">>> suite $SUITE done. results: $OUT_ROOT/$SUITE"
if [ -z "$DRY_RUN_FLAG" ]; then
  echo ">>> analysis"
  python "$SCRIPT_DIR/analyze_bench.py" "$OUT_ROOT" --suite "$SUITE" --cases "$CASES_JSONL"
else
  echo ">>> analyze: python scripts/bench/analyze_bench.py $OUT_ROOT --suite $SUITE --cases $CASES_JSONL"
fi
