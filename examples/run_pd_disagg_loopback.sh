#!/bin/bash
# PD 分离 sim-only e2e 对比: PD off baseline vs PD on (kv_both / P2pNccl / 25 GB/s).
#
# 单进程跑两次 (vLLM 进程独立, 状态不串), 抓 TTFT + transfer stats, 由 awk 算 delta.
#
# 跑:
#   bash examples/run_pd_disagg_loopback.sh

set -e

export VLLM_VIRTUAL_BACKEND=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_INFER_SIM_TIME_MODE=instant

MODEL=${VLLM_INFER_SIM_MODEL:-/data/ygw/models/Qwen3-4B-Instruct-2507}

# ---- baseline ----
echo "================================================================"
echo "[step 1/2] baseline (PD off)"
echo "================================================================"
unset LLM_INFER_SIM_PD_ROLE
BASE_OUT=$(VLLM_INFER_SIM_MODEL="$MODEL" python examples/run_pd_disagg_loopback.py 2>&1 || true)
echo "$BASE_OUT" | tail -15
TTFT_BASE=$(echo "$BASE_OUT" | grep "ttft_ms" | awk '{print $NF}')

# ---- PD on (kv_both, P2pNccl, default 25 GB/s) ----
echo
echo "================================================================"
echo "[step 2/2] PD on — kv_role=kv_both, P2pNcclConnector, 25 GB/s"
echo "================================================================"
PD_OUT=$(LLM_INFER_SIM_PD_ROLE=kv_both \
         LLM_INFER_SIM_PD_CONNECTOR=P2pNcclConnector \
         VLLM_INFER_SIM_MODEL="$MODEL" \
         python examples/run_pd_disagg_loopback.py 2>&1 || true)
echo "$PD_OUT" | tail -18
TTFT_PD=$(echo "$PD_OUT" | grep "ttft_ms" | awk '{print $NF}')
XFER_MS=$(echo "$PD_OUT" | grep "pd_total_xfer_ms" | awk '{print $NF}')
XFER_MB=$(echo "$PD_OUT" | grep "pd_total_xfer_MB" | awk '{print $NF}')

echo
echo "================================================================"
echo "[summary]"
echo "  TTFT baseline (PD off): ${TTFT_BASE} ms"
echo "  TTFT PD on:             ${TTFT_PD} ms"
echo "  Delta:                  $(awk "BEGIN { print ${TTFT_PD} - ${TTFT_BASE} }") ms"
echo "  Reported xfer bytes:    ${XFER_MB} MB"
echo "  Reported xfer time:     ${XFER_MS} ms"
echo "================================================================"

# delta 应 ≈ xfer_ms (cost path 的唯一新增项)
DELTA=$(awk "BEGIN { print ${TTFT_PD} - ${TTFT_BASE} }")
DIFF=$(awk "BEGIN { d = ${DELTA} - ${XFER_MS}; if (d<0) d=-d; print d }")
echo "  |delta - xfer_ms| = ${DIFF} ms (期望 < 0.5 ms)"
if awk "BEGIN { exit !(${DIFF} < 0.5) }"; then
    echo
    echo "PD DISAGG LOOPBACK e2e PASSED."
    exit 0
else
    echo
    echo "PD DISAGG LOOPBACK e2e FAILED — TTFT delta 与 reported xfer 不匹配"
    exit 1
fi
