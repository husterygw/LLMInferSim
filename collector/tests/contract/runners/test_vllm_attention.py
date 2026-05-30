"""vllm_attention runner build_record + input validation (CPU mock, no GPU).

GPU smoke 拆到 collector/tests/gpu/test_vllm_attention_smoke.py。
"""
from __future__ import annotations

import pytest

from collector.harness import BenchResult
from collector.runners import vllm_attention
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    OpKind,
)


def _prefill_case() -> Case:
    return Case.make(OpKind.ATTENTION, {
        "phase": "prefill",
        "batch_size": 1, "isl": 512, "kv_prefill": 0,
        "n_decode": 0, "kv_decode": 0,
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
                                          device_name="RTX 4090",
                                          kernel_source="vllm_flash_attn")
        assert r.op_kind == OpKind.ATTENTION
        assert r.framework == Framework.VLLM
        assert r.execution_mode == ExecutionMode.CUDAGRAPH
        assert r.metrics.latency_us_p50 == 200.0

    def test_kernel_source_recorded(self):
        # build_record 是 source-agnostic: 记录调用方传入的 kernel_source
        # (实际后端选择由 run_case 在 GPU path 决定, 见 vllm_attention.run_case).
        case = _prefill_case()
        bench = BenchResult(
            latency_us_p50=200.0, latency_us_p10=190.0, latency_us_p90=210.0,
            used_cuda_graph=True, n_warmups=3, n_iters=10,
        )
        r = vllm_attention.build_record(case, bench,
                                          framework_version="0.19.1",
                                          device_name="X",
                                          kernel_source="vllm_triton_attn")
        assert r.kernel_source == "vllm_triton_attn"


class TestInputValidation:
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
