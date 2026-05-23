"""V3 §4.5 CostTraceEntry / StepCostTrace — cost 层输出.

per-op trace + step-level aggregation. 字段语义:
    source       : module_profile / operator_db / comm_roofline / roofline
    match_type   : exact / nearest / fallback
    roofline_s   : 同样 shape 走 roofline 的下限 latency (用于 DB hit vs lower bound 对比)
    roofline_gap : 当 source != roofline 时, latency_s / roofline_s; 否则 None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CostTraceEntry:
    op_name: str
    op_kind: str
    op_subtype: str
    latency_s: float
    source: str
    match_type: str
    layer_idx: int | None = None
    display_name: str | None = None
    roofline_s: float | None = None
    roofline_gap: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepCostTrace:
    step_id: int
    phase: str
    total_latency_s: float
    compute_time_s: float
    memory_time_s: float
    comm_time_s: float
    runtime_time_s: float
    entries: tuple[CostTraceEntry, ...]
    bottleneck: str

    def to_report_dict(self) -> dict[str, Any]:
        """转 dict 供当前 reporter/tests 消费. 不承诺兼容旧 dict 结构 (IMPL_PLAN §1.4 Step 1.5)."""
        return {
            "step_id": self.step_id,
            "phase": self.phase,
            "total_latency_s": self.total_latency_s,
            "compute_time_s": self.compute_time_s,
            "memory_time_s": self.memory_time_s,
            "comm_time_s": self.comm_time_s,
            "runtime_time_s": self.runtime_time_s,
            "bottleneck": self.bottleneck,
            "entries": [
                {
                    "op_name": e.op_name,
                    "display_name": (
                        e.display_name
                        if e.display_name is not None
                        else format_display_name(e.op_name, e.layer_idx)
                    ),
                    "layer_idx": e.layer_idx,
                    "op_kind": e.op_kind,
                    "op_subtype": e.op_subtype,
                    "latency_s": e.latency_s,
                    "source": e.source,
                    "match_type": e.match_type,
                    "roofline_s": e.roofline_s,
                    "roofline_gap": e.roofline_gap,
                    "metadata": dict(e.metadata),
                }
                for e in self.entries
            ],
        }


def format_display_name(op_name: str, layer_idx: int | None) -> str:
    """Human-readable unique-ish op label for traces/reports."""
    if layer_idx is None:
        return op_name
    return f"layer{layer_idx}.{op_name}"
