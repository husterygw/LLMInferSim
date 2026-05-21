"""V3 §4.4 StepOpPlan — 一个 step 的 op graph.

当前新 runtime: flat tuple of Operator-compatible objects.
后续扩展为 StagePlan / RankPlan 嵌套 (TP/EP/DP, collective, overlap, critical path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operators.specs import Operator


@dataclass(frozen=True)
class StepOpPlan:
    step_id: int
    phase: str
    ops: tuple[Operator | Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
