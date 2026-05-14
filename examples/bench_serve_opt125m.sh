#!/bin/bash
# 阶段 0/3 起服务 + bench 端到端验证 (opt-125m, 单 GPU 模拟)
#
# 验证 VirtualPlatform 在 vllm OpenAI server + `vllm bench serve` 全链路跑通。
# 不依赖真 GPU, 全部走 VirtualPlatform sim 路径 (realtime 模式 sleep 模拟 latency).
#
# 跑:
#   bash examples/bench_serve_opt125m.sh
#
# 预期: ~60-90 秒后看到 bench summary (throughput / TTFT / TPOT).

set -e

MODEL=facebook/opt-125m
PORT=8765
NUM_PROMPTS=20

export VLLM_VIRTUAL_BACKEND=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_INFER_SIM_HW=${LLM_INFER_SIM_HW:-H100}
export LLM_INFER_SIM_TIME_MODE=${LLM_INFER_SIM_TIME_MODE:-realtime}

# --- 1. 起 server (后台) ---
echo "[bench_serve] starting vllm serve $MODEL on port $PORT ..."
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 512 \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-logprobs 0 \
    > /tmp/vllm_serve_opt125m.log 2>&1 &
SERVER_PID=$!
trap "echo '[bench_serve] killing server pid=$SERVER_PID'; kill $SERVER_PID 2>/dev/null || true" EXIT

# --- 2. 等 server ready ---
echo "[bench_serve] waiting for server to be ready (server pid=$SERVER_PID) ..."
for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[bench_serve] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[bench_serve] ERROR: server died, log tail:"
        tail -40 /tmp/vllm_serve_opt125m.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[bench_serve] ERROR: server did not become ready in 120s"
    tail -40 /tmp/vllm_serve_opt125m.log
    exit 1
fi

# --- 3. 跑 bench serve ---
echo "[bench_serve] running vllm bench serve (num_prompts=$NUM_PROMPTS) ..."
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len 128 \
    --random-output-len 32 \
    --request-rate inf

echo "[bench_serve] PASSED — vllm serve + bench end-to-end works."
