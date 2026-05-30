"""vllm_gemm runner GPU smoke — 真跑一个 tiny GEMM (skipif no CUDA)。

CPU mock (build_record / input validation) 在
collector/tests/contract/runners/test_vllm_gemm.py。
"""
from __future__ import annotations

import os

import pytest

from collector.runners import vllm_gemm
from collector.schemas import (
    Case,
    Framework,
    OpKind,
    RawRecord,
)


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
    def test_tiny_bf16_gemm(self):
        """跑最小 GEMM (m=1, n=512, k=512), 验证管线端到端."""
        case = Case.make(OpKind.GEMM, {
            "op_subtype": "qkv_proj",
            "m": 1, "n": 512, "k": 512,
            "dtype": "bf16", "tp": 1,
        })
        record = vllm_gemm.run_case(case, device=0)
        assert isinstance(record, RawRecord)
        assert record.op_kind == OpKind.GEMM
        assert record.framework == Framework.VLLM
        assert record.kernel_source == "vllm_row_parallel_linear"
        # latency 必须 > 0 (即使 cudagraph 启动开销也得有数)
        assert record.metrics.latency_us_p50 > 0
        # RTX 4090 BF16 tiny GEMM 单次应该 < 1ms (1000µs)
        assert record.metrics.latency_us_p50 < 1000
        # p10 <= p50 <= p90
        assert record.metrics.latency_us_p10 <= record.metrics.latency_us_p50
        assert record.metrics.latency_us_p50 <= record.metrics.latency_us_p90
        # device 字段填了
        assert record.device != ""
