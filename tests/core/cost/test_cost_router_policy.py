"""CostRouter policy 单测 — Step 3.7.

锁住:
  - roofline_only: 不查 DB, 全 source=roofline
  - operator_db_first: hit → source=operator_db, miss → source=roofline
  - require_operator_db: miss raises LookupError
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.operator_db import OperatorDBBackend
from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostPolicy, CostRouter
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_db.stores.memory import MemoryOperatorStore
from llm_infer_sim.core.operator_schema import virtual_op_to_signature
from llm_infer_sim.core.operators.ops import GemmOp
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile


def _gemm_op(m=128, subtype="qkv_proj") -> GemmOp:
    return GemmOp(
        name=f"layer0_{subtype}", op_subtype=subtype,
        phase="prefill", layer_idx=0, dtype="bf16",
        m=m, n=6144, k=2560, tp=1,
        framework="vllm", framework_version="0.20.1",
        execution_mode="eager", kernel_source="vllm_default",
    )


def _record(op: GemmOp, latency_us=234.5) -> OperatorRecord:
    sig = virtual_op_to_signature(op)
    return OperatorRecord(
        signature=sig, hardware="RTX_4090",
        framework="vllm", framework_version="0.20.1",
        execution_mode="eager", kernel_source="vllm_default",
        latency_us_p50=latency_us, latency_us_p10=latency_us * 0.9,
        latency_us_p90=latency_us * 1.1, n_iters=10, n_warmups=3,
    )


@pytest.fixture
def hw():
    return get_hardware_profile("RTX_4090")


@pytest.fixture
def deploy():
    return DeployConfig()


def test_router_without_db_defaults_to_roofline_only(hw, deploy):
    """没传 operator_db 时, policy.mode 自动 roofline_only."""
    rl = RooflineBackend(hw, deploy)
    router = CostRouter(rl)
    assert router.policy.mode == "roofline_only"
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(_gemm_op(),))
    trace = router.estimate(plan)
    assert trace.entries[0].source == "roofline"


def test_operator_db_first_hit_then_miss(hw, deploy):
    """两个 op, 一个 DB 有, 一个没有: 第一个 operator_db, 第二个 roofline."""
    rl = RooflineBackend(hw, deploy)
    store = MemoryOperatorStore()
    op_hit = _gemm_op(m=128, subtype="qkv_proj")
    store.add(_record(op_hit, latency_us=234.5))
    op_miss = _gemm_op(m=128, subtype="o_proj")    # not in store

    backend = OperatorDBBackend(store, roofline=rl)
    router = CostRouter(rl, operator_db=backend)
    assert router.policy.mode == "operator_db_first"
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(op_hit, op_miss))
    trace = router.estimate(plan)

    sources = [e.source for e in trace.entries]
    match_types = [e.match_type for e in trace.entries]
    assert sources == ["operator_db", "roofline"]
    assert match_types == ["exact", "fallback"]
    assert trace.entries[0].latency_s == pytest.approx(234.5e-6)
    # second one is roofline-derived
    assert trace.entries[1].metadata.get("kernel_overhead") is not None


def test_require_operator_db_miss_raises(hw, deploy):
    rl = RooflineBackend(hw, deploy)
    backend = OperatorDBBackend(MemoryOperatorStore())
    router = CostRouter(
        rl, operator_db=backend,
        policy=CostPolicy(mode="require_operator_db"),
    )
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(_gemm_op(),))
    with pytest.raises(LookupError, match="no DB hit"):
        router.estimate(plan)


def test_require_operator_db_hit_works(hw, deploy):
    rl = RooflineBackend(hw, deploy)
    store = MemoryOperatorStore()
    op = _gemm_op()
    store.add(_record(op))
    backend = OperatorDBBackend(store, roofline=rl)
    router = CostRouter(
        rl, operator_db=backend,
        policy=CostPolicy(mode="require_operator_db"),
    )
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(op,))
    trace = router.estimate(plan)
    assert trace.entries[0].source == "operator_db"


def test_roofline_only_skips_db(hw, deploy):
    """显式 roofline_only 即使有 DB 也不查."""
    rl = RooflineBackend(hw, deploy)
    store = MemoryOperatorStore()
    op = _gemm_op()
    store.add(_record(op, latency_us=234.5))
    backend = OperatorDBBackend(store)
    router = CostRouter(
        rl, operator_db=backend,
        policy=CostPolicy(mode="roofline_only"),
    )
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(op,))
    trace = router.estimate(plan)
    assert trace.entries[0].source == "roofline"
    # Not the DB value
    assert trace.entries[0].latency_s != pytest.approx(234.5e-6)


def test_disabled_roofline_fallback_raises_on_miss(hw, deploy):
    rl = RooflineBackend(hw, deploy)
    backend = OperatorDBBackend(MemoryOperatorStore())
    router = CostRouter(
        rl, operator_db=backend,
        policy=CostPolicy(
            mode="operator_db_first",
            enable_roofline_fallback=False,
        ),
    )
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(_gemm_op(),))
    with pytest.raises(LookupError, match="roofline fallback disabled"):
        router.estimate(plan)
