"""OperatorDBBackend 单测 — Step 3.6."""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.operator_db import OperatorDBBackend
from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_db.stores.memory import MemoryOperatorStore
from llm_infer_sim.core.operator_schema import operator_to_signature
from llm_infer_sim.core.operators import GEMM, Norm
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile


def _gemm_op(m=128, mode="eager", ks="vllm_default", fwv="0.20.1") -> GEMM:
    from llm_infer_sim.core.operators.context import build_operator_context
    from tests.helpers.support import make_model_config
    ctx = build_operator_context(
        make_model_config(),
        DeploymentProfile.flat(),
        RuntimeProfile.flat(execution_mode=mode, backend="vllm", backend_version=fwv),
        get_hardware_profile("RTX_4090"),
    )
    return GEMM(
        name="qkv_proj", op_subtype="qkv_proj",
        phase="prefill", layer_idx=0,
        m=m, n=6144, k=2560,
        ctx=ctx,
        kernel_source=ks,
    )


def _record_for(op: GEMM, latency_us: float = 100.0) -> OperatorRecord:
    sig = operator_to_signature(op)
    return OperatorRecord(
        signature=sig, hardware="RTX_4090",
        framework="vllm", framework_version="0.20.1",
        execution_mode="eager", kernel_source="vllm_default",
        latency_us_p50=latency_us, latency_us_p10=latency_us * 0.9,
        latency_us_p90=latency_us * 1.1, n_iters=10, n_warmups=3,
        source={"case_id": "test-case-id", "source_profiles": ["qwen3_4b"]},
    )


def test_hit_returns_operator_db_entry():
    store = MemoryOperatorStore()
    op = _gemm_op()
    store.add(_record_for(op, latency_us=234.5))

    backend = OperatorDBBackend(store)
    entry = backend.estimate(op)
    assert entry is not None
    assert entry.source == "operator_db"
    assert entry.match_type == "exact"
    assert entry.latency_s == pytest.approx(234.5e-6)
    assert entry.metadata["case_id"] == "test-case-id"
    assert entry.metadata["kernel_source"] == "vllm_default"


def test_miss_returns_none():
    store = MemoryOperatorStore()
    backend = OperatorDBBackend(store)
    assert backend.estimate(_gemm_op()) is None


def test_eager_record_does_not_hit_cudagraph_op():
    """execution_mode 进 signature, eager vs cudagraph 不互相命中."""
    store = MemoryOperatorStore()
    op_eager = _gemm_op(mode="eager")
    store.add(_record_for(op_eager))

    backend = OperatorDBBackend(store)
    op_graph = _gemm_op(mode="cudagraph")
    assert backend.estimate(op_graph) is None


def test_unsupported_op_kind_returns_none():
    from llm_infer_sim.core.operators.context import build_operator_context
    from tests.helpers.support import make_model_config
    store = MemoryOperatorStore()
    backend = OperatorDBBackend(store)
    ctx = build_operator_context(
        make_model_config(), DeploymentProfile.flat(), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
    )
    norm_op = Norm(
        name="x", op_subtype="rmsnorm",
        phase="prefill", layer_idx=0,
        tokens=1, hidden=128, ctx=ctx,
    )
    assert backend.estimate(norm_op) is None


def test_hit_with_roofline_compare():
    """passing roofline backend should fill roofline_s + roofline_gap (V3 §4.5)."""
    store = MemoryOperatorStore()
    op = _gemm_op(m=128)
    # Put a record with much higher latency than roofline lower-bound
    store.add(_record_for(op, latency_us=1000.0))

    hw = get_hardware_profile("RTX_4090")
    rl = RooflineBackend(hw, "eager")
    backend = OperatorDBBackend(store, roofline=rl)
    entry = backend.estimate(op)
    assert entry.source == "operator_db"
    assert entry.roofline_s is not None
    assert entry.roofline_gap is not None
    # real (1000us) vs roofline; gap = real / roofline. Both > 0, gap should be > 0.
    assert entry.roofline_gap > 0
    assert entry.latency_s == pytest.approx(1000e-6)
