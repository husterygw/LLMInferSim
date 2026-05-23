"""V3 §7.3 RooflineBackend + Step 1.6 CostRouter 单测.

锁住:
  - Operator.formula -> RooflineSpec 转换正确
  - estimate() 返 CostTraceEntry, source=roofline, match_type=fallback
  - 大 GEMM (compute-bound) / 小 GEMM (memory-bound) bottleneck 区分
  - CostRouter aggregate StepOpPlan -> StepCostTrace
  - collective op 在阶段 1 被跳过
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.cost.trace import CostTraceEntry, StepCostTrace
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.operators import Collective, GEMM
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile


@pytest.fixture
def hw():
    return get_hardware_profile("RTX_4090")


@pytest.fixture
def deploy_eager():
    return DeployConfig(execution_mode="eager")


@pytest.fixture
def deploy_graph():
    return DeployConfig(execution_mode="cudagraph")


@pytest.fixture
def backend_eager(hw, deploy_eager):
    return RooflineBackend(hw, deploy_eager)


def _gemm_op(m: int, n: int, k: int, name: str = "qkv_proj") -> GEMM:
    """构 GEMM op: GEMM.roofline_spec() 算出 flops=2*m*n*k + 2-byte loads (bf16)."""
    from llm_infer_sim.core.operators.context import build_operator_context
    from llm_infer_sim.core.profiles.model_config import ModelConfig
    ctx = build_operator_context(
        ModelConfig(),
        DeployConfig(backend="vllm", backend_version="0.20.1"),
        get_hardware_profile("RTX_4090"),
    )
    return GEMM(
        name=name, op_subtype=name,
        phase="prefill", layer_idx=0,
        m=m, n=n, k=k,
        ctx=ctx,
    )


def test_estimate_returns_cost_trace_entry(backend_eager):
    op = _gemm_op(m=128, n=6144, k=2560)
    entry = backend_eager.estimate(op)
    assert isinstance(entry, CostTraceEntry)
    assert entry.op_name == "qkv_proj"
    assert entry.display_name == "layer0.qkv_proj"
    assert entry.layer_idx == 0
    assert entry.op_kind == "gemm"
    assert entry.source == "roofline"
    assert entry.match_type == "fallback"
    assert entry.latency_s > 0
    assert entry.roofline_s == entry.latency_s
    assert entry.roofline_gap is None


def test_metadata_includes_bottleneck_breakdown(backend_eager):
    op = _gemm_op(m=128, n=6144, k=2560)
    entry = backend_eager.estimate(op)
    md = entry.metadata
    assert "bottleneck" in md
    assert md["bottleneck"] in ("compute", "memory")
    assert md["t_compute"] > 0
    assert md["t_memory"] > 0
    assert md["arithmetic_intensity"] > 0


def test_large_gemm_is_compute_bound(backend_eager):
    """大 M (M=8192) 高 arithmetic intensity, 应 compute-bound."""
    op = _gemm_op(m=8192, n=6144, k=2560)
    entry = backend_eager.estimate(op)
    assert entry.metadata["bottleneck"] == "compute"


def test_small_gemm_is_memory_bound(backend_eager):
    """小 M (M=1, decode 单 token) 低 arithmetic intensity, 应 memory-bound."""
    op = _gemm_op(m=1, n=6144, k=2560)
    entry = backend_eager.estimate(op)
    assert entry.metadata["bottleneck"] == "memory"


def test_execution_mode_affects_kernel_overhead(hw, deploy_eager, deploy_graph):
    """V3 §7.3 + Phase 5: cudagraph 模式 kernel_overhead = 0, eager 有 dispatch 开销.

    现有 HardwareConfig 默认 kernel_overhead={} (calibrate=True 也清空),
    测试为隔离 execution_mode 行为, 手动注入 default 2us overhead.
    """
    hw.kernel_overhead = {"default": 2e-6}   # 2 µs per op (typical eager dispatch)
    op = _gemm_op(m=128, n=6144, k=2560)
    eager_be = RooflineBackend(hw, deploy_eager)
    graph_be = RooflineBackend(hw, deploy_graph)
    e_eager = eager_be.estimate(op)
    e_graph = graph_be.estimate(op)
    assert e_eager.metadata["kernel_overhead"] == pytest.approx(2e-6)
    assert e_graph.metadata["kernel_overhead"] == 0


def test_router_aggregates_step_plan(backend_eager):
    ops = (
        _gemm_op(m=128, n=6144, k=2560, name="qkv_proj"),
        _gemm_op(m=128, n=2560, k=2560, name="o_proj"),
    )
    plan = StepOpPlan(step_id=0, phase="prefill", ops=ops)
    router = CostRouter(backend_eager)
    trace = router.estimate(plan)
    assert isinstance(trace, StepCostTrace)
    assert trace.step_id == 0
    assert trace.phase == "prefill"
    assert len(trace.entries) == 2
    assert trace.total_latency_s == pytest.approx(sum(e.latency_s for e in trace.entries))
    assert trace.compute_time_s > 0
    assert trace.memory_time_s > 0
    assert trace.comm_time_s == 0.0
    assert trace.runtime_time_s == 0.0
    assert trace.bottleneck in ("compute", "memory")


def test_router_skips_collective_in_stage_1(backend_eager):
    """阶段 5 才接 communication backend, 阶段 1 collective op 直接跳过不入 trace."""
    from llm_infer_sim.core.operators.context import build_operator_context
    from llm_infer_sim.core.profiles.model_config import ModelConfig
    gemm = _gemm_op(m=128, n=6144, k=2560)
    coll_ctx = build_operator_context(
        ModelConfig(),
        DeployConfig(tp_size=2, backend="vllm", backend_version="0.20.1"),
        get_hardware_profile("RTX_4090"),
    )
    coll = Collective(
        name="attn_allreduce", op_subtype="allreduce",
        phase="prefill", layer_idx=0,
        message_bytes=128 * 2560 * 2, world_size=2,
        ctx=coll_ctx, comm_backend="nccl",
        roofline_spec_value=RooflineSpec(
            comm_bytes=128 * 2560 * 2,
            comm_type="allreduce",
            op_category="communication",
        ),
    )
    plan = StepOpPlan(step_id=0, phase="prefill", ops=(gemm, coll))
    router = CostRouter(backend_eager)
    trace = router.estimate(plan)
    assert len(trace.entries) == 1
    assert trace.entries[0].op_name == "qkv_proj"


def test_to_report_dict_round_trip(backend_eager):
    op = _gemm_op(m=128, n=6144, k=2560)
    plan = StepOpPlan(step_id=7, phase="prefill", ops=(op,))
    router = CostRouter(backend_eager)
    trace = router.estimate(plan)
    d = trace.to_report_dict()
    assert d["step_id"] == 7
    assert d["phase"] == "prefill"
    assert len(d["entries"]) == 1
    assert d["entries"][0]["source"] == "roofline"
    assert d["entries"][0]["match_type"] == "fallback"
    assert d["entries"][0]["op_name"] == "qkv_proj"
    assert d["entries"][0]["display_name"] == "layer0.qkv_proj"
    assert d["entries"][0]["layer_idx"] == 0


def test_formula_translation_preserves_mem_breakdown(backend_eager):
    """RooflineBackend._to_operator_profile 保 load_weight/load_act/store_act 5-way breakdown."""
    op = _gemm_op(m=128, n=6144, k=2560)
    entry = backend_eager.estimate(op)
    # mem_bytes = load_weight + load_act + store_act (KV 路径未启用)
    expected_mem = 6144 * 2560 * 2 + 128 * 2560 * 2 + 128 * 6144 * 2
    assert entry.metadata["mem_bytes"] == expected_mem
    assert entry.metadata["flops"] == 2 * 128 * 6144 * 2560
