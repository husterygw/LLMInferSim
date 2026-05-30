"""vllm_attention runner GPU smoke — 真跑 tiny prefill/decode (skipif no CUDA)。

CPU mock (build_record / input validation) 在
collector/tests/contract/runners/test_vllm_attention.py。
"""
from __future__ import annotations

import os

import pytest

from collector.runners import vllm_attention
from collector.schemas import (
    Case,
    OpKind,
    RawRecord,
)


def _prefill_case() -> Case:
    return Case.make(OpKind.ATTENTION, {
        "phase": "prefill",
        "batch_size": 1, "isl": 512, "kv_prefill": 0,
        "n_decode": 0, "kv_decode": 0,
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "dtype": "bf16", "tp": 1, "execution_mode": "cudagraph",
    })


def _decode_case(ctx: int = 128) -> Case:
    return Case.make(OpKind.ATTENTION, {
        "phase": "decode",
        "batch_size": 1, "isl": 0, "kv_prefill": 0,
        "n_decode": 1, "kv_decode": ctx,
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
        "dtype": "bf16", "tp": 1, "execution_mode": "cudagraph",
    })


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(
    not _cuda_available() or os.environ.get("COLLECTOR_SKIP_GPU") == "1",
    reason="requires CUDA + vLLM; set COLLECTOR_SKIP_GPU=1 to skip on GPU box",
)
class TestRunCaseGpuSmoke:
    def test_tiny_prefill(self):
        case = _prefill_case()
        r = vllm_attention.run_case(case, device=0)
        assert isinstance(r, RawRecord)
        assert r.op_kind == OpKind.ATTENTION
        assert r.metrics.latency_us_p50 > 0
        assert r.metrics.latency_us_p10 <= r.metrics.latency_us_p50 <= r.metrics.latency_us_p90

    def test_tiny_decode(self):
        case = _decode_case(ctx=512)
        r = vllm_attention.run_case(case, device=0)
        assert isinstance(r, RawRecord)
        assert r.params["phase"] == "decode"
        assert r.metrics.latency_us_p50 > 0
