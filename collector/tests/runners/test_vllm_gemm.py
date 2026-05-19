"""vllm_gemm runner — unit (mock) + GPU smoke (skipif no CUDA)."""
from __future__ import annotations

import os

import pytest

from collector.harness import BenchResult
from collector.runners import vllm_gemm
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    OpKind,
    RawRecord,
)


# ---------------------------------------------------------------------------
# build_record — 纯函数, 单元测可覆盖
# ---------------------------------------------------------------------------

class TestBuildRecord:
    def _case(self) -> Case:
        return Case.make(OpKind.GEMM, {
            "op_subtype": "qkv_proj",
            "m": 128, "n": 6144, "k": 2560,
            "dtype": "bf16", "tp": 1,
        })

    def _bench(self, used_graph: bool = True) -> BenchResult:
        return BenchResult(
            latency_us_p50=120.0,
            latency_us_p10=118.0,
            latency_us_p90=125.0,
            used_cuda_graph=used_graph,
            n_warmups=3,
            n_iters=10,
            fallback_reason=None if used_graph else "capture_failed",
        )

    def test_basic_fields(self):
        case = self._case()
        bench = self._bench(used_graph=True)
        r = vllm_gemm.build_record(case, bench,
                                    framework_version="0.19.1",
                                    device_name="NVIDIA GeForce RTX 4090")
        assert r.case_id == case.case_id
        assert r.op_kind == OpKind.GEMM
        assert r.framework == Framework.VLLM
        assert r.framework_version == "0.19.1"
        assert r.device == "NVIDIA GeForce RTX 4090"
        assert r.kernel_source == "vllm_row_parallel_linear"

    def test_execution_mode_from_graph_flag(self):
        case = self._case()
        r_graph = vllm_gemm.build_record(case, self._bench(used_graph=True),
                                          framework_version="0.19.1",
                                          device_name="X")
        assert r_graph.execution_mode == ExecutionMode.CUDAGRAPH

        r_eager = vllm_gemm.build_record(case, self._bench(used_graph=False),
                                          framework_version="0.19.1",
                                          device_name="X")
        assert r_eager.execution_mode == ExecutionMode.EAGER

    def test_metrics_round_trip(self):
        case = self._case()
        bench = self._bench(used_graph=True)
        r = vllm_gemm.build_record(case, bench,
                                    framework_version="0.19.1", device_name="X")
        assert r.metrics.latency_us_p50 == 120.0
        assert r.metrics.latency_us_p10 == 118.0
        assert r.metrics.latency_us_p90 == 125.0
        assert r.metrics.used_cuda_graph is True
        assert r.metrics.n_warmups == 3
        assert r.metrics.n_iters == 10

    def test_fallback_reason_into_metadata(self):
        case = self._case()
        bench = self._bench(used_graph=False)
        r = vllm_gemm.build_record(case, bench,
                                    framework_version="0.19.1", device_name="X")
        assert r.metadata["fallback_reason"] == "capture_failed"

    def test_params_dict_is_copied(self):
        """build_record copies params dict, mutating original 不影响 record."""
        case = self._case()
        r = vllm_gemm.build_record(case, self._bench(),
                                    framework_version="0.19.1", device_name="X")
        case.params["m"] = 999
        assert r.params["m"] == 128

    def test_kernel_source_override(self):
        case = self._case()
        r = vllm_gemm.build_record(case, self._bench(),
                                    framework_version="0.19.1", device_name="X",
                                    kernel_source="vllm_fused_fp8_gemm")
        assert r.kernel_source == "vllm_fused_fp8_gemm"


# ---------------------------------------------------------------------------
# Input validation (CPU-only, no GPU launched)
# ---------------------------------------------------------------------------

class TestRunCaseInputValidation:
    def test_tp_gt_1_not_implemented(self):
        case = Case.make(OpKind.GEMM, {
            "op_subtype": "qkv_proj", "m": 1, "n": 100, "k": 100,
            "dtype": "bf16", "tp": 2,
        })
        with pytest.raises(NotImplementedError, match="distributed"):
            vllm_gemm.run_case(case, 0)

    def test_unsupported_dtype(self):
        case = Case.make(OpKind.GEMM, {
            "op_subtype": "qkv_proj", "m": 1, "n": 100, "k": 100,
            "dtype": "fp8", "tp": 1,
        })
        with pytest.raises(NotImplementedError, match="BF16"):
            vllm_gemm.run_case(case, 0)


# ---------------------------------------------------------------------------
# GPU smoke test — 真跑一个 tiny GEMM, 验证 latency 非 0 + RawRecord 完整
# ---------------------------------------------------------------------------

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
