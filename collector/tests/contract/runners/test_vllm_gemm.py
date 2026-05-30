"""vllm_gemm runner build_record + input validation (CPU mock, no GPU).

GPU smoke 拆到 collector/tests/gpu/test_vllm_gemm_smoke.py。
"""
from __future__ import annotations

import pytest

from collector.harness import BenchResult
from collector.runners import vllm_gemm
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    OpKind,
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
    def test_unsupported_dtype(self):
        case = Case.make(OpKind.GEMM, {
            "op_subtype": "qkv_proj", "m": 1, "n": 100, "k": 100,
            "dtype": "fp8", "tp": 1,
        })
        with pytest.raises(NotImplementedError, match="BF16"):
            vllm_gemm.run_case(case, 0)
