#!/bin/bash
# Stage A/B/C/D/E 校准 benchmark driver (详 docs/CALIBRATION_METHODOLOGY.md §5).
#
# 跑法:
#   bash scripts/run_stage_bench.sh A      # TP=1 单请求
#   bash scripts/run_stage_bench.sh B      # TP>1 单请求 (通信校准)
#   bash scripts/run_stage_bench.sh C      # TP=1 多请求 (并发)
#   bash scripts/run_stage_bench.sh D      # TP>1 多请求 (完整 production)
#   bash scripts/run_stage_bench.sh E      # 多模型回归

set -e
source /data/ygw/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate llm_sim 2>/dev/null || true

STAGE="${1:-A}"
MODEL="${MODEL:-/data/ygw/models/Qwen3-4B-Instruct-2507}"
OUT_BASE="${OUT_BASE:-/tmp/stage_${STAGE}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

rm -rf "$OUT_BASE"; mkdir -p "$OUT_BASE"
echo ">>> Stage $STAGE  model=$MODEL  out=$OUT_BASE"

# 单请求 = num_prompts=3 + request-rate=0.5 (顺序串行)
# 多请求 = num_prompts=10 + request-rate=inf (并发)
# scenarios (input output): 128/128, 256/256, 512/512, 1024/512, 2048/512

run_one() {
  local tp=$1 hint=$2 num_prompts=$3 rate=$4 label=$5
  echo
  echo "=== TP=$tp hint=$hint num_prompts=$num_prompts rate=$rate label=$label ==="
  export LLM_INFER_SIM_NUMA_HINT="$hint"
  RESULTS_DIR="$OUT_BASE/$label" \
    MODELS="$MODEL" \
    TP=$tp GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}" \
    REQUEST_RATE="$rate" \
    bash "$SCRIPT_DIR/bench_compare.sh" \
    > "$OUT_BASE/${label}.log" 2>&1 || echo "  WARN: $label exit $?"
  grep -A 8 "scenario           real" "$OUT_BASE/${label}.log" 2>/dev/null | head -10
}

# Stage 内部场景遍历, 改 bench_compare 的 SCENARIOS 阵列
# (假设 bench_compare 已经包含 5 个 input/output 组合)

case "$STAGE" in
  A)
    # TP=1 单请求 (顺序), 5 scenarios in bench_compare.sh
    # 临时把 bench script 中的 num_prompts 改成 1+rate 0.5 通过 env
    # 这里简化: 暂时 reuse bench_compare 默认 (5 scenarios), TP=1
    run_one 1 concentrated 10 inf TP1_default
    ;;
  B)
    # TP>1 单请求 (串行), 各 NUMA 拓扑
    run_one 2 concentrated 10 0.5 TP2_same
    run_one 2 balanced     10 0.5 TP2_cross
    run_one 4 concentrated 10 0.5 TP4_same
    run_one 4 balanced     10 0.5 TP4_cross
    run_one 8 balanced     10 0.5 TP8
    ;;
  C)
    # TP=1 多请求 (5/10/20 并发)
    # bench_compare 已经是多请求 (num_prompts=10 in scenarios), 这就是 C 默认
    run_one 1 concentrated 10 inf TP1_concurrent
    ;;
  D)
    # TP>1 多请求 (完整 production)
    run_one 2 concentrated 10 inf TP2_same_multi
    run_one 4 concentrated 10 inf TP4_same_multi
    run_one 4 balanced     10 inf TP4_cross_multi
    run_one 8 balanced     10 inf TP8_multi
    ;;
  E)
    # 多模型回归 (Qwen2.5-3B + Qwen3-32B 抽样)
    if [ -d /data/ygw/models/Qwen2.5-3B-Instruct ]; then
      MODEL=/data/ygw/models/Qwen2.5-3B-Instruct run_one 1 concentrated 10 inf Qwen25_TP1
    fi
    if [ -d /data/ygw/models/Qwen3-32B ]; then
      MODEL=/data/ygw/models/Qwen3-32B run_one 4 concentrated 10 inf Qwen32B_TP4
    fi
    ;;
  M-A)
    # MoE Stage A: Qwen3-30B-A3B 单请求 (60GB BF16 需 TP=4 才 fit RTX 4090 24G)
    # 这是 MoE-specific 路径单测, 验证 _build_moe_ffn_block / AllToAll / routing skew 建模.
    export MODEL=/data/ygw/models/Qwen3-30B-A3B-Instruct-2507
    # MoE 60GB / 4 = 15G/GPU, GPU_MEM_UTIL=0.85 防 KV cache OOM
    export GPU_MEM_UTIL=0.85
    run_one 4 concentrated 10 0.5 MoE_TP4_serial
    ;;
  M-B)
    # MoE Stage B: 不同 TP/EP/拓扑
    export MODEL=/data/ygw/models/Qwen3-30B-A3B-Instruct-2507
    export GPU_MEM_UTIL=0.85
    run_one 4 concentrated 10 0.5 MoE_TP4_same
    run_one 4 balanced     10 0.5 MoE_TP4_cross
    run_one 8 balanced     10 0.5 MoE_TP8
    ;;
  *)
    echo "Unknown stage: $STAGE (use A/B/C/D/E)"; exit 1
    ;;
esac

echo
echo ">>> Stage $STAGE DONE. results: $OUT_BASE/"
echo ">>> Run: python scripts/analyze_stage.py $OUT_BASE  to compare against SLA"
