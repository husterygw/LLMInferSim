#!/bin/bash
# 阶段 9.1 服务级验证: DeepSeek-V3.2-Exp DSA + vllm bench serve.
#
# 比 Qwen3-30B-A3B 多覆盖:
#   - MLA + lightning indexer + sparse-attended MLA kernel (DSA)
#   - 8 worker + trust_remote_code + custom tokenizer
#
# 跑:
#   bash examples/bench_serve_v32_tp8.sh
#
# 预期: ~3-5 分钟 (8 worker spawn + custom modeling 加载 + bench)

set -e

MODEL=deepseek-ai/DeepSeek-V3.2-Exp
PORT=8767
NUM_PROMPTS=20
TP=8

export VLLM_VIRTUAL_BACKEND=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_INFER_SIM_HW=${LLM_INFER_SIM_HW:-H200}
export LLM_INFER_SIM_TIME_MODE=${LLM_INFER_SIM_TIME_MODE:-realtime}

# --- 1. 起 server (tp=8 + EP + trust_remote_code + skip_tokenizer_init) ---
echo "[bench_serve] starting vllm serve $MODEL tp=$TP ep=on on port $PORT ..."
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --enable-expert-parallel \
    --trust-remote-code \
    --tokenizer-mode slow \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-logprobs 0 \
    > /tmp/vllm_serve_v32.log 2>&1 &
SERVER_PID=$!
trap "echo '[bench_serve] killing server pid=$SERVER_PID'; kill $SERVER_PID 2>/dev/null || true; sleep 1; pkill -9 -P $SERVER_PID 2>/dev/null || true" EXIT

# --- 2. 等 server ready (300s, V3.2 8 worker + custom modeling 加载慢) ---
echo "[bench_serve] waiting for server ready (pid=$SERVER_PID) ..."
for i in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[bench_serve] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[bench_serve] ERROR: server died, log tail:"
        tail -50 /tmp/vllm_serve_v32.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[bench_serve] ERROR: server did not become ready in 300s"
    tail -50 /tmp/vllm_serve_v32.log
    exit 1
fi

# --- 3. 跑 bench serve ---
# server 用 --skip-tokenizer-init (V3.2 custom tokenizer 需 sentencepiece path),
# bench client 自己 init tokenizer (sentencepiece 0.2.1 已装).
echo "[bench_serve] running vllm bench serve (num_prompts=$NUM_PROMPTS) ..."
vllm bench serve \
    --backend vllm \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL" \
    --trust-remote-code \
    --tokenizer-mode slow \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len 512 \
    --random-output-len 32 \
    --request-rate inf

echo "[bench_serve] PASSED — DeepSeek-V3.2-Exp DSA serve+bench end-to-end."
