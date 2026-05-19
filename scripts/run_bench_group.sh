#!/bin/bash
# Run all cases of a benchmark group (sim vs real).
#
# 用法:
#   bash scripts/run_bench_group.sh <group_name>       # 跑一个 group 所有 case
#   bash scripts/run_bench_group.sh --list             # 列所有 group
#   bash scripts/run_bench_group.sh <group> --dry-run  # 预览跑哪些 case 不实际跑
#
# 例:
#   bash scripts/run_bench_group.sh single_request_tp1
#   bash scripts/run_bench_group.sh moe_single_request_tp_only
#
# 输出: /tmp/llm_infer_sim_bench/<group>/<case_id>/{real,sim}_<scenario>.txt
#       /tmp/llm_infer_sim_bench/<group>/<case_id>/{real,sim}_server.log
#       /tmp/llm_infer_sim_bench/<group>/<case_id>/metrics.json
#       /tmp/llm_infer_sim_bench/<group>/<case_id>/run.log
#
# CALIBRATION_METHODOLOGY.md §5 对应 stage 命名:
#   single_request_tp1        ≡ Stage A
#   single_request_multi_tp   ≡ Stage B
#   concurrent_tp1            ≡ Stage C
#   concurrent_multi_tp       ≡ Stage D
#   moe_single_request_*      ≡ Stage M-A (TP-only / EP 分两类)
#   moe_concurrent_*          ≡ Stage M-B
#   multi_model_regression    ≡ Stage E

set -e
source /data/ygw/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate llm_sim 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CASES_JSONL="${CASES_JSONL:-/tmp/llm_infer_sim_bench/cases.jsonl}"
OUT_ROOT="${OUT_ROOT:-/tmp/llm_infer_sim_bench}"

if [ ! -f "$CASES_JSONL" ]; then
  echo "Generating $CASES_JSONL..."
  python "$SCRIPT_DIR/bench_cases.py" --out "$CASES_JSONL"
fi

GROUP="${1:-}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

if [ "$GROUP" = "--list" ] || [ -z "$GROUP" ]; then
  echo "Available groups:"
  python3 -c "
import json
from collections import Counter
groups = Counter()
for line in open('$CASES_JSONL'):
    g = json.loads(line)['group']
    groups[g] += 1
for g, n in sorted(groups.items()):
    print(f'  {g:<35} {n:>3} cases')
"
  exit 0
fi

# 用一次 python 把这个 group 所有 case 序列化成 shell-friendly 行
# 字段间用 |, 顺序: case_id|model_path|tp|ep|hint|mode|conc|num_prompts|num_warmups|rate|input|output|wkld_type|gpu_mem|enable_ep
mapfile -t CASES < <(python3 - "$CASES_JSONL" "$GROUP" "$SCRIPT_DIR" <<'PY'
import json, sys
cases_path, group, script_dir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, script_dir)
from bench_cases import MODEL_ALIASES
for line in open(cases_path):
    c = json.loads(line)
    if c['group'] != group:
        continue
    w = c['workload']
    if w['type'] == 'fixed':
        inp, out = w['input_len'], w['output_len']
    else:
        inp, out = -1, -1   # mixed 暂未支持
    # request_rate 可能存成 "inf" 字符串 (bench_cases 写出时) 或 float
    rate = c.get('request_rate', 'inf')
    rate_str = 'inf' if (isinstance(rate, str) and rate.lower() == 'inf') else str(rate)
    # num_prompts 缺省 = concurrency (向后兼容老 cases.jsonl)
    num_prompts = c.get('num_prompts', c.get('concurrency', 1))
    fields = [
        c['case_id'],
        MODEL_ALIASES.get(c['model_alias'], c['model_alias']),
        str(c.get('tp', 1)),
        str(c.get('ep', 1)),
        c.get('topology_hint', 'concentrated'),
        c.get('execution_mode', 'eager'),
        str(c.get('concurrency', 1)),
        str(num_prompts),
        str(c.get('num_warmups', 1)),
        rate_str,
        str(inp), str(out),
        w['type'],
        str(c.get('gpu_mem_util', 0.5)),
        '1' if c.get('enable_expert_parallel') else '0',
    ]
    print('|'.join(fields))
PY
)

if [ "${#CASES[@]}" -eq 0 ]; then
  echo "No cases in group: $GROUP"
  echo "Run: bash $0 --list"
  exit 1
fi

echo ">>> group=$GROUP, ${#CASES[@]} cases"
echo ">>> dry-run=$DRY_RUN"

OUT_DIR="$OUT_ROOT/$GROUP"
mkdir -p "$OUT_DIR"

PASS=0; SKIP=0; FAIL=0

run_case() {
  local line="$1"
  IFS='|' read -r case_id model_path tp ep hint mode conc num_prompts num_warmups rate input_len output_len wkld_type gpu_mem enable_ep <<< "$line"

  echo
  echo "=== $case_id ==="
  echo "  model=$(basename "$model_path") tp=$tp ep=$ep hint=$hint mode=$mode conc=$conc np=$num_prompts nw=$num_warmups rate=$rate gpu_mem=$gpu_mem ep_on=$enable_ep"
  echo "  workload=$wkld_type input=$input_len output=$output_len"

  if [ "$wkld_type" != "fixed" ]; then
    echo "  SKIP (workload=$wkld_type 暂未支持)"
    SKIP=$((SKIP+1)); return 0
  fi
  if [ ! -d "$model_path" ]; then
    echo "  SKIP (model not found: $model_path)"
    SKIP=$((SKIP+1)); return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  (dry-run)"
    return 0
  fi

  local case_dir="$OUT_DIR/$case_id"
  mkdir -p "$case_dir"

  # cudagraph mode → ENFORCE_EAGER=off; eager → on
  local eager_flag="on"
  [ "$mode" = "cudagraph" ] && eager_flag="off"
  local ep_flag="off"
  [ "$enable_ep" = "1" ] && ep_flag="on"

  export LLM_INFER_SIM_NUMA_HINT="$hint"

  # MAX_MODEL_LEN 动态: ISL+OSL+512 buffer, 下限 4096 (跟 bench_compare 默认对齐), 对齐 8 倍数
  local max_len=$(( input_len + output_len + 512 ))
  [ "$max_len" -lt 4096 ] && max_len=4096
  max_len=$(( (max_len + 7) / 8 * 8 ))

  # GPU 拓扑映射 (RTX 4090: 4 GPU/root × 2 root = 8 GPU)
  # concentrated = 同 root (越近 PCIe BW 越高);  balanced = 跨 root (更慢)
  # 不设这个 sim 侧 NUMA_HINT 跟 real 侧实际拓扑不匹配, 对照无效
  local cuda_devs=""
  case "${tp}_${hint}" in
    1_*)              cuda_devs="0" ;;
    2_concentrated)   cuda_devs="0,1" ;;
    2_balanced)       cuda_devs="0,4" ;;
    4_concentrated)   cuda_devs="0,1,2,3" ;;
    4_balanced)       cuda_devs="0,1,4,5" ;;
    8_*)              cuda_devs="0,1,2,3,4,5,6,7" ;;
    *)                cuda_devs=$(seq -s, 0 $((tp - 1))) ;;
  esac
  export CUDA_VISIBLE_DEVICES="$cuda_devs"
  echo "  CUDA_VISIBLE_DEVICES=$cuda_devs (TP=$tp hint=$hint)"

  # SCENARIO_OVERRIDE = "name input output num_prompts", num_prompts = concurrency
  local rc=0
  # scenario 文件名固定 "case", 因为每个 case 已经独立目录, 不需再区分
  # SCENARIO_OVERRIDE 第 4 位是 bench --num-prompts, 用 num_prompts (不是语义 conc)
  SCENARIO_OVERRIDE="case ${input_len} ${output_len} ${num_prompts}" \
    MODELS="$model_path" \
    TP="$tp" \
    GPU_MEM_UTIL="$gpu_mem" \
    REQUEST_RATE="$rate" \
    RESULTS_DIR="$case_dir" \
    ENFORCE_EAGER="$eager_flag" \
    ENABLE_EP="$ep_flag" \
    MAX_MODEL_LEN="$max_len" \
    NUM_WARMUPS="$num_warmups" \
    bash "$SCRIPT_DIR/bench_compare.sh" \
    > "$case_dir/run.log" 2>&1 || rc=$?

  if [ $rc -ne 0 ]; then
    echo "  FAIL (bench_compare exit $rc, see $case_dir/run.log)"
    FAIL=$((FAIL+1))
    # don't abort: 仍然尝试 metrics extract, 可能部分 scenario 有数据
  fi

  python3 "$SCRIPT_DIR/_extract_metrics.py" \
    --case-id "$case_id" --group "$GROUP" --case-dir "$case_dir" \
    > "$case_dir/metrics.json" 2> "$case_dir/metrics.err" || \
    echo "  WARN: metrics extract failed (see $case_dir/metrics.err)"

  python3 -c "
import json, sys
try:
    m = json.load(open('$case_dir/metrics.json'))
    r = m.get('real', {}); s = m.get('sim', {})
    rt, rp = r.get('TTFT_ms_mean'), r.get('TPOT_ms_mean')
    st, sp = s.get('TTFT_ms_mean'), s.get('TPOT_ms_mean')
    def fmt(x): return f'{x:.1f}' if isinstance(x,(int,float)) else 'NA'
    print(f'  real TTFT={fmt(rt)}ms TPOT={fmt(rp)}ms | sim TTFT={fmt(st)}ms TPOT={fmt(sp)}ms')
except Exception as e:
    print(f'  (summary failed: {e})')
" 2>/dev/null
  [ $rc -eq 0 ] && PASS=$((PASS+1))
}

for line in "${CASES[@]}"; do
  run_case "$line" || true
done

echo
echo ">>> group $GROUP done. PASS=$PASS SKIP=$SKIP FAIL=$FAIL"
echo ">>> results: $OUT_DIR"
echo ">>> analyze: python scripts/analyze_bench.py $OUT_ROOT --group $GROUP"
