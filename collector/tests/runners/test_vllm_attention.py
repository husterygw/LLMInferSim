"""vllm_attention runner — unit (mock) + GPU smoke."""
from __future__ import annotations

import os

import pytest

from collector.harness import BenchResult
from collector.runners import vllm_attention
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
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


class TestBuildRecord:
    def test_basic(self):
        case = _prefill_case()
        bench = BenchResult(
            latency_us_p50=200.0, latency_us_p10=190.0, latency_us_p90=210.0,
            used_cuda_graph=True, n_warmups=3, n_iters=10,
        )
        r = vllm_attention.build_record(case, bench,
                                          framework_version="0.19.1",
                                          device_name="RTX 4090")
        assert r.op_kind == OpKind.ATTENTION
        assert r.framework == Framework.VLLM
        assert r.execution_mode == ExecutionMode.CUDAGRAPH
        assert r.metrics.latency_us_p50 == 200.0

    def test_kernel_source_default(self):
        case = _prefill_case()
        bench = BenchResult(
            latency_us_p50=200.0, latency_us_p10=190.0, latency_us_p90=210.0,
            used_cuda_graph=True, n_warmups=3, n_iters=10,
        )
        r = vllm_attention.build_record(case, bench,
                                          framework_version="0.19.1",
                                          device_name="X")
        assert r.kernel_source == "vllm_flash_attn"


class TestInputValidation:
    def test_tp_gt_1_not_implemented(self):
        case = Case.make(OpKind.ATTENTION, {
            "phase": "decode",
            "batch_size": 1, "isl": 0, "kv_prefill": 0,
            "n_decode": 1, "kv_decode": 128,
            "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
            "dtype": "bf16", "tp": 2, "execution_mode": "eager",
        })
        with pytest.raises(NotImplementedError, match="distributed"):
            vllm_attention.run_case(case, 0)

    def test_unsupported_dtype(self):
        case = Case.make(OpKind.ATTENTION, {
            "phase": "prefill",
            "batch_size": 1, "isl": 128, "kv_prefill": 0,
            "n_decode": 0, "kv_decode": 0,
            "num_heads": 4, "num_kv_heads": 2, "head_dim": 32,
            "dtype": "fp8", "tp": 1, "execution_mode": "eager",
        })
        with pytest.raises(NotImplementedError, match="BF16"):
            vllm_attention.run_case(case, 0)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


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
