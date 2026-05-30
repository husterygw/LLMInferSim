"""Phase 1 unified router path (op_plan §结合实施): StepOpPlan.runtime →
op.forward(runtime) dispatch + static `count` scaling, exercised in isolation
with fakes (real ops migrate to this contract in Phase 2+)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.cost.trace import CostTraceEntry
from llm_infer_sim.core.graph.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.graph.step_plan import StepOpPlan


class _FakeRoofline:
    """Returns a fixed 10us entry per op, recording (op, op_runtime) calls."""

    def __init__(self):
        self.calls: list[tuple[Any, Any]] = []

    def estimate(self, op, op_runtime=None) -> CostTraceEntry:
        self.calls.append((op, op_runtime))
        return CostTraceEntry(
            op_name=op.name, op_kind=op.op_kind, op_subtype=op.op_subtype,
            latency_s=10e-6, source="roofline", match_type="formula",
            metadata={"t_compute": 10e-6, "t_memory": 0.0, "t_comm": 0.0},
        )


@dataclass(frozen=True)
class _StaticOp:
    name: str
    count: int = 1
    active: bool = True
    op_kind: str = "gemm"
    op_subtype: str = "qkv_proj"
    seen: list = field(default_factory=list)

    def forward(self, step: StepRuntime) -> OpRuntime | None:
        self.seen.append(step)
        if not self.active:
            return None
        return OpRuntime(phase=step.phase, op_subtype=None,
                         shape={}, parallel={}, runtime={})


def _router():
    return CostRouter(_FakeRoofline())  # operator_db=None → roofline_only


def _runtime():
    return StepRuntime(phase="prefill", total_tokens=2048, num_prefill_requests=1)


def test_forward_called_with_step_runtime_and_count_scaled():
    rt = _router()
    op = _StaticOp(name="qkv_proj", count=36)
    trace = rt.estimate(StepOpPlan(step_id=0, phase="prefill", ops=(op,), runtime=_runtime()))
    # forward() got the StepRuntime
    assert op.seen and op.seen[0].total_tokens == 2048
    # single entry, scaled by count=36
    assert len(trace.entries) == 1
    e = trace.entries[0]
    assert e.metadata["count"] == 36
    assert e.latency_s == 36 * 10e-6
    assert e.metadata["t_compute"] == 36 * 10e-6
    assert trace.total_latency_s == 36 * 10e-6
    assert e.display_name == "qkv_proj[count=36]"


def test_forward_none_skips_inactive_op():
    rt = _router()
    active = _StaticOp(name="attn_prefill", count=1, active=True)
    inactive = _StaticOp(name="attn_decode", count=1, active=False)
    trace = rt.estimate(StepOpPlan(step_id=0, phase="prefill",
                                   ops=(active, inactive), runtime=_runtime()))
    names = [e.op_name for e in trace.entries]
    assert names == ["attn_prefill"]  # inactive op (forward→None) skipped


def test_no_runtime_keeps_legacy_path_no_forward():
    """StepOpPlan without runtime → forward() not called, count from attr (=1)."""
    rt = _router()
    op = _StaticOp(name="qkv_proj", count=5)
    trace = rt.estimate(StepOpPlan(step_id=0, phase="prefill", ops=(op,)))  # runtime=None
    assert op.seen == []  # forward NOT called when plan.runtime is None
    # count still applies from op.count (legacy ops just have count=1 → no-op)
    assert trace.entries[0].metadata["count"] == 5
