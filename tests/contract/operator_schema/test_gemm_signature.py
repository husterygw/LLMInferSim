"""GEMM canonicalizer 单测 — Step 2.2.

锁住:
  - collector Case.params + runtime GEMM 生成相同 signature
  - eager vs cudagraph signature 不同
  - framework_version 进 key (不同版本不互相命中)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_infer_sim.core.operator_schema.gemm import (
    gemm_case_params_to_signature,
    gemm_operator_to_signature,
)
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.operators import GEMM
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from tests.helpers.support import make_model_config


_COLLECTOR_CTX = dict(
    framework="vllm",
    framework_version="0.20.1",
    kernel_source="vllm_default",
)


def _gemm_case(m=128, n=6144, k=2560, dtype="bf16", tp=1, mode="eager",
               subtype="qkv_proj"):
    return {
        "op_subtype": subtype,
        "m": m, "n": n, "k": k,
        "dtype": dtype, "tp": tp,
        "execution_mode": mode,
    }


def _ctx(tp=1, mode="eager", framework_version="0.20.1"):
    return build_operator_context(
        make_model_config(),
        DeploymentProfile.flat(tp=tp),
        RuntimeProfile.flat(
            execution_mode=mode, backend="vllm", backend_version=framework_version,
        ),
        get_hardware_profile("RTX_4090"),
    )


def _gemm_op(m=128, n=6144, k=2560, dtype="bf16", tp=1, mode="eager",
             subtype="qkv_proj", framework_version="0.20.1",
             kernel_source="vllm_default") -> GEMM:
    return GEMM(
        name=subtype,
        op_subtype=subtype,
        phase="prefill", layer_idx=0,
        m=m, n=n, k=k,
        ctx=_ctx(tp=tp, mode=mode, framework_version=framework_version),
        dtype_override=dtype,
        kernel_source=kernel_source,
    )


def test_collector_and_runtime_signature_match():
    """同 shape 同 runtime context: case 和 op 必须 hash 到同一个 signature."""
    case = _gemm_case()
    op = _gemm_op()
    sig_c = gemm_case_params_to_signature(case, **_COLLECTOR_CTX)
    sig_r = gemm_operator_to_signature(op)
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_signature_contains_shape_and_tp():
    sig = gemm_case_params_to_signature(_gemm_case(), **_COLLECTOR_CTX)
    assert sig.op_kind == "gemm"
    assert sig.op_subtype == "qkv_proj"
    assert sig.dtype == "bf16"
    shape_dict = dict(sig.shape)
    assert shape_dict["m"] == 128
    assert shape_dict["n"] == 6144
    assert shape_dict["k"] == 2560
    parallel_dict = dict(sig.parallel)
    assert parallel_dict["tp"] == 1


def test_eager_and_cudagraph_signatures_differ():
    """execution_mode 必入 runtime key, eager/cudagraph 不互相命中 (IMPL_PLAN §2.4 测试点 3)."""
    sig_e = gemm_case_params_to_signature(_gemm_case(mode="eager"), **_COLLECTOR_CTX)
    sig_g = gemm_case_params_to_signature(_gemm_case(mode="cudagraph"), **_COLLECTOR_CTX)
    assert sig_e != sig_g
    assert sig_e.stable_hash() != sig_g.stable_hash()


def test_framework_version_in_runtime_key():
    sig_a = gemm_case_params_to_signature(
        _gemm_case(),
        framework="vllm", framework_version="0.20.1", kernel_source="vllm_default",
    )
    sig_b = gemm_case_params_to_signature(
        _gemm_case(),
        framework="vllm", framework_version="0.21.0", kernel_source="vllm_default",
    )
    assert sig_a != sig_b


def test_kernel_source_in_runtime_key():
    sig_a = gemm_case_params_to_signature(
        _gemm_case(),
        framework="vllm", framework_version="0.20.1", kernel_source="vllm_default",
    )
    sig_b = gemm_case_params_to_signature(
        _gemm_case(),
        framework="vllm", framework_version="0.20.1", kernel_source="vllm_fused_qkv",
    )
    assert sig_a != sig_b


def test_subtype_separates_qkv_vs_oproj():
    sig_q = gemm_case_params_to_signature(_gemm_case(subtype="qkv_proj"), **_COLLECTOR_CTX)
    sig_o = gemm_case_params_to_signature(_gemm_case(subtype="o_proj"), **_COLLECTOR_CTX)
    assert sig_q != sig_o


def test_dtype_in_signature():
    sig_bf16 = gemm_case_params_to_signature(_gemm_case(dtype="bf16"), **_COLLECTOR_CTX)
    sig_fp8 = gemm_case_params_to_signature(_gemm_case(dtype="fp8"), **_COLLECTOR_CTX)
    assert sig_bf16 != sig_fp8


def test_tp_in_parallel_key():
    sig_tp1 = gemm_case_params_to_signature(_gemm_case(tp=1), **_COLLECTOR_CTX)
    sig_tp2 = gemm_case_params_to_signature(_gemm_case(tp=2), **_COLLECTOR_CTX)
    assert sig_tp1 != sig_tp2


def test_operator_wrong_kind_raises():
    """传一个 op_kind=attention 的通用容器 op, 走 GEMM canonicalizer 应该拒."""
    bogus = SimpleNamespace(op_kind="attention")
    with pytest.raises(ValueError, match="op_kind=gemm"):
        gemm_operator_to_signature(bogus)
