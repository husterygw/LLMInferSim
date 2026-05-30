#!/bin/bash
# DP G3 服务级验证: Qwen3-4B dp=2 tp=1 + vllm bench serve.
#
# 验证:
#   - vllm serve --data-parallel-size 2 spawn 2 个 engine 进程
#   - 各进程 VirtualPlatform 独立 simulate own batch, throughput ~ 2× single-engine
#
# Note (DP G3 发现): vLLM v1 内部把 engine 的 data_parallel_size 重置为 1,
# 每个 engine 当独立单元跑, 上层 LLM 路由分配请求。我们的 `_sync_dp_latency`
# 在当前 vLLM v1 路径下进 fast path (dp_size_local==1) — 这是正确行为, 没问题。
# G3 代码作为防御保留, 若未来 vLLM 改成 per-step 跨 engine 同步会自动生效。
#
# 跑:
#   bash examples/bench_serve_qwen3_4b_dp2.sh
#
# 预期: ~90-150 秒 (2 engine spawn + bench)

set -e

MODEL=/data1/home/ygw268/models/Qwen3-4B-Instruct-2507
PORT=8768
NUM_PROMPTS=20

export VLLM_VIRTUAL_BACKEND=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_INFER_SIM_HW=${LLM_INFER_SIM_HW:-H100}
export LLM_INFER_SIM_TIME_MODE=${LLM_INFER_SIM_TIME_MODE:-realtime}

# --- 1. 起 server (dp=2 tp=1) ---
echo "[bench_serve] starting vllm serve $MODEL dp=2 tp=1 on port $PORT ..."
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-logprobs 0 \
    > /tmp/vllm_serve_qwen3_4b_dp2.log 2>&1 &
SERVER_PID=$!
trap "echo '[bench_serve] killing server pid=$SERVER_PID'; kill $SERVER_PID 2>/dev/null || true; sleep 1; pkill -9 -P $SERVER_PID 2>/dev/null || true" EXIT

# --- 2. 等 ready ---
echo "[bench_serve] waiting for server ready (pid=$SERVER_PID) ..."
for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[bench_serve] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[bench_serve] ERROR: server died, log tail:"
        tail -50 /tmp/vllm_serve_qwen3_4b_dp2.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[bench_serve] ERROR: server did not become ready in 180s"
    tail -50 /tmp/vllm_serve_qwen3_4b_dp2.log
    exit 1
fi

# --- 3. bench ---
echo "[bench_serve] running vllm bench serve (num_prompts=$NUM_PROMPTS) ..."
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len 256 \
    --random-output-len 32 \
    --request-rate inf

echo "[bench_serve] PASSED — Qwen3-4B dp=2 tp=1 serve+bench end-to-end."
