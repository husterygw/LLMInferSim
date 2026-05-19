#!/bin/bash
# 真 GPU vs 仿真 (LLMInferSim) TTFT/TPOT 对比 bench.
#
# 跑同一份 `vllm bench serve` 命令两次:
#   1) 直连真 GPU 上的 vLLM server  → 实测 TTFT / TPOT
#   2) 直连 LLMInferSim virtual backend 跑的 vLLM server → 仿真 TTFT / TPOT
# 最后输出 5 个 scenario 的对比表 (gap%). 支持一次跑多模型对比.
#
# Usage:
#   bash scripts/bench_compare.sh                                       # 默认 Qwen3-4B
#   MODELS=/path/to/A,/path/to/B bash scripts/bench_compare.sh          # 多模型
#   HW=H100 bash scripts/bench_compare.sh                               # 换硬件 profile
#
# Env vars (可选):
#   MODELS             模型路径列表, 逗号分隔. 默认 = $MODEL (单模型 back-compat)
#   MODEL              单模型路径 (默认 /data/ygw/models/Qwen3-4B-Instruct-2507)
#   HW                 LLM_INFER_SIM_HW (默认 RTX_4090)
#   CONDA_ENV          conda 环境名 (默认 llm_sim, 设 "" 跳过激活)
#   CUDA_VISIBLE_DEVICES  默认 0
#   RESULTS_DIR        输出根目录 (默认 /tmp/bench_compare_results), 多模型每个一子目录
#   GPU_MEM_UTIL       gpu-memory-utilization (默认 0.5)
#   MAX_MODEL_LEN      max-model-len (默认 4096)
#   MAX_NUM_SEQS       max-num-seqs (默认 16)
#   MAX_BATCH_TOKENS   max-num-batched-tokens (默认 8192)
#   PREFIX_CACHE       on / off (默认 off, 干净对比)
#   ENFORCE_EAGER      on / off (默认 on, 跟 cost model module 级 overhead 语义对齐)
#   SCENARIO_OVERRIDE  一条单独的 scenario, 覆盖默认 5 条. 格式: "name input_len output_len num_prompts"
#                      例: SCENARIO_OVERRIDE="case 256 256 10"
#   ENABLE_EP          on / off (默认 off). on 时给 vllm serve 加 --enable-expert-parallel
#   NUM_WARMUPS        bench --num-warmups (默认 1). 让 sim/real 各自跑 warmup prompt 排除冷启动

set -e

# ---------- defaults ----------
MODEL="${MODEL:-/data/ygw/models/Qwen3-4B-Instruct-2507}"
MODELS="${MODELS:-$MODEL}"
HW="${HW:-RTX_4090}"
CONDA_ENV="${CONDA_ENV-llm_sim}"
RESULTS_DIR_ROOT="${RESULTS_DIR:-/tmp/bench_compare_results}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-8192}"
PREFIX_CACHE="${PREFIX_CACHE:-off}"
ENFORCE_EAGER="${ENFORCE_EAGER:-on}"
ENABLE_EP="${ENABLE_EP:-off}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
TP="${TP:-1}"
# 没显式给 CUDA_VISIBLE_DEVICES 时, 根据 TP 默认 0,1,...,TP-1
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
  _gpus=$(seq -s, 0 $((TP - 1)))
  export CUDA_VISIBLE_DEVICES="$_gpus"
else
  export CUDA_VISIBLE_DEVICES
fi

# ---------- conda env ----------
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

# 必备 env
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RESULTS_DIR_ROOT"

# ---------- scenarios: name | input_len | output_len | num_prompts ----------
if [ -n "${SCENARIO_OVERRIDE:-}" ]; then
  SCENARIOS=("$SCENARIO_OVERRIDE")
else
  SCENARIOS=(
    "128_128     128  128  10"
    "256_256     256  256  10"
    "512_512     512  512  10"
    "1024_512    1024 512  10"
    "2048_512    2048 512  10"
  )
fi

PREFIX_FLAG=""
if [ "$PREFIX_CACHE" = "off" ]; then
  PREFIX_FLAG="--no-enable-prefix-caching"
fi

EAGER_FLAG=""
if [ "$ENFORCE_EAGER" = "on" ]; then
  EAGER_FLAG="--enforce-eager"
fi

EP_FLAG=""
if [ "$ENABLE_EP" = "on" ]; then
  EP_FLAG="--enable-expert-parallel"
fi

# 关 vLLM V1 异步调度, 让 real 路径跟 sim 的串行 time.sleep 行为对齐
# (默认开启时 GPU compute 跟 CPU prepare/emit 是 pipeline 的, sim time.sleep 重现不了)
ASYNC_FLAG=""
if [ "${DISABLE_ASYNC_SCHED:-off}" = "on" ]; then
  ASYNC_FLAG="--no-async-scheduling"
fi

# 当前 model / results_dir, set_for_model() 改写
CUR_MODEL=""
CUR_RESULTS_DIR=""

set_for_model() {
  CUR_MODEL="$1"
  local short
  short="$(basename "$CUR_MODEL")"
  CUR_RESULTS_DIR="$RESULTS_DIR_ROOT/$short"
  mkdir -p "$CUR_RESULTS_DIR"
}

start_server() {
  local mode=$1   # "real" or "sim"
  local port=$2
  local logfile=$3

  if [ "$mode" = "real" ]; then
    unset VLLM_VIRTUAL_BACKEND
    unset LLM_INFER_SIM_HW
    unset LLM_INFER_SIM_TIME_MODE
  else
    export VLLM_VIRTUAL_BACKEND=1
    export LLM_INFER_SIM_HW="$HW"
    export LLM_INFER_SIM_TIME_MODE="${LLM_INFER_SIM_TIME_MODE:-realtime}"
  fi

  # shellcheck disable=SC2086
  vllm serve "$CUR_MODEL" \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
    $PREFIX_FLAG \
    $EAGER_FLAG \
    $EP_FLAG \
    $ASYNC_FLAG \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-logprobs 0 \
    --disable-log-stats \
    > "$logfile" 2>&1 &
  echo $!
}

wait_ready() {
  local port=$1
  local pid=$2
  local timeout=$3
  for i in $(seq 1 $timeout); do
    if curl -fsS "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

run_bench() {
  local port=$1
  local input_len=$2
  local output_len=$3
  local num_prompts=$4
  local out_file=$5

  # Stage A 用 REQUEST_RATE=0.5 让 prompt 间隔 2s, 完全串行
  # Stage C/D 用 REQUEST_RATE=inf 让所有 prompt 同时到达 (满并发)
  local rate="${REQUEST_RATE:-inf}"
  vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port "$port" \
    --model "$CUR_MODEL" \
    --dataset-name random \
    --num-prompts "$num_prompts" \
    --num-warmups "$NUM_WARMUPS" \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --request-rate "$rate" \
    --ignore-eos \
    > "$out_file" 2>&1
}

extract_metric() {
  local file=$1
  local metric=$2
  grep -m1 "Mean ${metric}" "$file" 2>/dev/null | awk -F: '{print $NF}' | tr -d ' '
}

run_all() {
  local mode=$1
  local port=$2

  echo ">>> [$(basename "$CUR_MODEL")] starting $mode server on port $port..."
  local srvlog="$CUR_RESULTS_DIR/${mode}_server.log"
  : > "$srvlog"
  local pid
  pid=$(start_server "$mode" "$port" "$srvlog")

  echo ">>> waiting for $mode server ready (max 240s)..."
  if ! wait_ready "$port" "$pid" 240; then
    echo "ERROR: $mode server failed to start"
    tail -50 "$srvlog"
    kill -9 "$pid" 2>/dev/null || true
    return 1
  fi

  for scen in "${SCENARIOS[@]}"; do
    # shellcheck disable=SC2086
    read -r name input_len output_len num_prompts <<< "$scen"
    local out="$CUR_RESULTS_DIR/${mode}_${name}.txt"
    echo ">>> [$mode] scenario=$name input=$input_len output=$output_len num=$num_prompts"
    run_bench "$port" "$input_len" "$output_len" "$num_prompts" "$out" || \
      echo "  (bench failed for $name)"
  done

  kill "$pid" 2>/dev/null || true
  pkill -P "$pid" 2>/dev/null || true
  wait 2>/dev/null || true
  sleep 5
}

print_summary_for_model() {
  echo
  echo "================================================================"
  echo "Comparison: REAL vs SIM — $(basename "$CUR_MODEL")"
  echo "  hw         = $HW"
  echo "  TP         = $TP  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
  echo "  prefix     = $PREFIX_CACHE"
  echo "  eager      = $ENFORCE_EAGER"
  echo "  ep         = $ENABLE_EP"
  echo "  warmups    = $NUM_WARMUPS"
  echo "  max_seqs   = $MAX_NUM_SEQS"
  echo "  max_btoks  = $MAX_BATCH_TOKENS"
  echo "================================================================"
  printf "%-15s %12s %12s %8s    %12s %12s %8s\n" \
    "scenario" "real_TTFT" "sim_TTFT" "gap%" "real_TPOT" "sim_TPOT" "gap%"
  echo "--------------------------------------------------------------------"
  for scen in "${SCENARIOS[@]}"; do
    read -r name _ <<< "$scen"
    real_ttft=$(extract_metric "$CUR_RESULTS_DIR/real_${name}.txt" "TTFT (ms)")
    sim_ttft=$(extract_metric "$CUR_RESULTS_DIR/sim_${name}.txt" "TTFT (ms)")
    real_tpot=$(extract_metric "$CUR_RESULTS_DIR/real_${name}.txt" "TPOT (ms)")
    sim_tpot=$(extract_metric "$CUR_RESULTS_DIR/sim_${name}.txt" "TPOT (ms)")
    if [ -z "$real_ttft" ] || [ -z "$sim_ttft" ]; then
      printf "%-15s %12s %12s %8s    %12s %12s %8s\n" \
        "$name" "${real_ttft:-MISS}" "${sim_ttft:-MISS}" "-" "${real_tpot:-MISS}" "${sim_tpot:-MISS}" "-"
      continue
    fi
    ttft_gap=$(awk "BEGIN { printf \"%.1f\", ($sim_ttft - $real_ttft) / $real_ttft * 100 }" 2>/dev/null || echo "-")
    tpot_gap=$(awk "BEGIN { printf \"%.1f\", ($sim_tpot - $real_tpot) / $real_tpot * 100 }" 2>/dev/null || echo "-")
    printf "%-15s %12s %12s %7s%%    %12s %12s %7s%%\n" \
      "$name" "$real_ttft" "$sim_ttft" "$ttft_gap" "$real_tpot" "$sim_tpot" "$tpot_gap"
  done
  echo "--------------------------------------------------------------------"
  echo "raw results: $CUR_RESULTS_DIR"
}

# ===== main loop: 每个模型独立 server pair + 独立 summary =====
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
echo "Will benchmark ${#MODEL_LIST[@]} model(s):"
for m in "${MODEL_LIST[@]}"; do
  echo "  - $m"
done

for m in "${MODEL_LIST[@]}"; do
  m="$(echo "$m" | xargs)"   # trim spaces
  [ -z "$m" ] && continue
  echo
  echo "==================== model: $(basename "$m") ===================="
  set_for_model "$m"
  run_all real 8810
  run_all sim 8811
  print_summary_for_model
done

echo
echo "ALL DONE. results root: $RESULTS_DIR_ROOT"
