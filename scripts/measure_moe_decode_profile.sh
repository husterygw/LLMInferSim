#!/bin/bash
# Wrapper: 启动带 torch profiler 的 vllm serve, 发 1 prompt 触发 profile,
# 收 trace, 调 parse 脚本输出 kernel category 分解.
#
# 用法:
#   bash scripts/measure_moe_decode_profile.sh
#
# 输出:
#   /tmp/moe_prof/*.json.gz   (chrome trace)
#   /tmp/moe_decode_profile.json (categorized 分解)
#
# 默认配置: Qwen3-30B-A3B + TP=4 + concentrated (GPUs 0-3), eager mode, 1 prompt
# input=128 output=8 (8 个 decode token 给 profiler 抓清晰)

set -e
source /data/ygw/miniconda3/etc/profile.d/conda.sh
conda activate llm_sim

MODEL="${MODEL:-/data/ygw/models/Qwen3-30B-A3B-Instruct-2507}"
TP="${TP:-4}"
PROF_DIR="${PROF_DIR:-/tmp/moe_prof}"
PORT="${PORT:-8812}"

rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"

export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset VLLM_VIRTUAL_BACKEND   # real GPU, not sim

# --profiler-config: 让 vllm serve 启用 torch profiler
PROFILER_JSON='{"profiler":"torch","torch_profiler_dir":"'$PROF_DIR'","torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true}'

echo ">>> starting vllm serve with torch profiler..."
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --max-logprobs 0 \
    --disable-log-stats \
    --profiler-config "$PROFILER_JSON" \
    > "$PROF_DIR/server.log" 2>&1 &
SERVER_PID=$!

# wait ready
echo ">>> waiting for server ready (max 180s)..."
for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server died"; tail -50 "$PROF_DIR/server.log"; exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: server not ready"; kill "$SERVER_PID"; exit 1
fi

# bench with --profile (vllm bench serve 自动 wrap start_profile/stop_profile)
echo ">>> running bench with --profile (1 prompt, 8 decode tokens)..."
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts 1 \
    --num-warmups 1 \
    --random-input-len 128 \
    --random-output-len 8 \
    --request-rate inf \
    --ignore-eos \
    --profile \
    > "$PROF_DIR/bench.log" 2>&1 || echo "bench finished (some warns OK)"

# server flushes profile on /stop_profile, give it a moment
sleep 3
kill "$SERVER_PID" 2>/dev/null || true
wait 2>/dev/null || true

echo
echo ">>> trace files dropped:"
ls -lh "$PROF_DIR"/*.json* 2>&1 | head

echo
TRACE=$(ls "$PROF_DIR"/*.json* 2>/dev/null | head -1)
if [ -n "$TRACE" ]; then
    echo ">>> parsing trace: $TRACE"
    python /data/ygw/llm_sim/LLMInferSim/scripts/measure_moe_decode_profile.py --parse "$TRACE"
else
    echo "WARN: no trace produced. check $PROF_DIR/server.log"
fi
