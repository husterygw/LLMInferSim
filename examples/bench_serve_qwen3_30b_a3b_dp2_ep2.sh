#!/bin/bash
# DP + EP MoE 服务级验证: Qwen3-30B-A3B dp=2 tp=1 enable_expert_parallel.
#
# 这是 production 部署里 DP 真正发挥价值的场景: routed experts 跨 dp×tp 个 GPU
# 切分 (ep_world=2), 每 engine 持 1/2 expert, AllToAll dispatch/combine 在
# ep_group 上跑。dense weights 仍是每 engine 一份。
#
# 验证:
#   - vllm serve --data-parallel-size 2 --enable-expert-parallel spawn 2 engine
#   - 各 engine 用 ep_world=2 计 routed expert sharding (sizing.per_rank_param_bytes)
#   - 各 engine 用 alltoall_time(bytes, ep=2, hw) 计 dispatch/combine cost
#
# 跑:
#   bash examples/bench_serve_qwen3_30b_a3b_dp2_ep2.sh
#
# 预期: ~3-5 分钟

set -e

MODEL=Qwen/Qwen3-30B-A3B
PORT=8769
NUM_PROMPTS=20

export VLLM_VIRTUAL_BACKEND=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_INFER_SIM_HW=${LLM_INFER_SIM_HW:-H100}
export LLM_INFER_SIM_TIME_MODE=${LLM_INFER_SIM_TIME_MODE:-realtime}

echo "[bench_serve] starting vllm serve $MODEL dp=2 ep=on tp=1 ..."
vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 1024 \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-logprobs 0 \
    > /tmp/vllm_serve_qwen3_30b_a3b_dp2_ep2.log 2>&1 &
SERVER_PID=$!
trap "echo '[bench_serve] killing server pid=$SERVER_PID'; kill $SERVER_PID 2>/dev/null || true; sleep 1; pkill -9 -P $SERVER_PID 2>/dev/null || true" EXIT

echo "[bench_serve] waiting for server ready (pid=$SERVER_PID) ..."
for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[bench_serve] server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[bench_serve] ERROR: server died, log tail:"
        tail -50 /tmp/vllm_serve_qwen3_30b_a3b_dp2_ep2.log
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[bench_serve] ERROR: server did not become ready in 240s"
    tail -50 /tmp/vllm_serve_qwen3_30b_a3b_dp2_ep2.log
    exit 1
fi

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

echo "[bench_serve] PASSED — Qwen3-30B-A3B dp=2 ep=on tp=1 serve+bench."
