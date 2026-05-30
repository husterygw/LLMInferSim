#!/bin/bash
# 阶段 6 服务级验证: Qwen3-30B-A3B (MoE + TP + EP) + vllm bench serve.
#
# 比 opt-125m 验证多覆盖:
#   - tp=2 + enable_expert_parallel=True → 多 worker + AllToAll
#   - 真实 MoE 模型 (128 experts × top-8) + shared expert + EP path
#   - 中等规模 prefill (256 tok) + 中等 decode (64 tok)
#
# 跑:
#   bash examples/bench_serve_qwen3_30b_a3b.sh
#
# 预期: ~90-180 秒 (2 worker spawn + bench)

set -e

MODEL=Qwen/Qwen3-30B-A3B
PORT=8766
NUM_PROMPTS=20
TP=2

export VLLM_VIRTUAL_BACKEND=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_INFER_SIM_HW=${LLM_INFER_SIM_HW:-H200}
export LLM_INFER_SIM_TIME_MODE=${LLM_INFER_SIM_TIME_MODE:-realtime}

# --- 1. 起 server (TP=2 + EP) ---
echo "[bench_serve] starting vllm serve $MODEL tp=$TP ep=on on port $PORT ..."
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --enable-expert-parallel \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-logprobs 0 \
    > /tmp/vllm_serve_qwen3_30b_a3b.log 2>&1 &
SERVER_PID=$!
trap "echo '[bench_serve] killing server pid=$SERVER_PID'; kill $SERVER_PID 2>/dev/null || true; sleep 1; pkill -9 -P $SERVER_PID 2>/dev/null || true" EXIT

# --- 2. 等 server ready (180s, MoE 加载慢) ---
echo "[bench_serve] waiting for server ready (pid=$SERVER_PID) ..."
for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[bench_serve] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[bench_serve] ERROR: server died, log tail:"
        tail -50 /tmp/vllm_serve_qwen3_30b_a3b.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[bench_serve] ERROR: server did not become ready in 180s"
    tail -50 /tmp/vllm_serve_qwen3_30b_a3b.log
    exit 1
fi

# --- 3. 跑 bench serve (random dataset, 中等规模) ---
echo "[bench_serve] running vllm bench serve (num_prompts=$NUM_PROMPTS) ..."
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len 256 \
    --random-output-len 64 \
    --request-rate inf

echo "[bench_serve] PASSED — Qwen3-30B-A3B MoE+EP serve+bench end-to-end."
