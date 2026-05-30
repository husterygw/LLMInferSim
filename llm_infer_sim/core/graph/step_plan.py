"""V3 §4.4 StepOpPlan — 一个 step 的 op graph.

当前新 runtime: flat tuple of Operator-compatible objects.
后续扩展为 StagePlan / RankPlan 嵌套 (TP/EP/DP, collective, overlap, critical path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_infer_sim.core.operators.base import Operator

if TYPE_CHECKING:
    from llm_infer_sim.core.graph.runtime import StepRuntime


@dataclass(frozen=True)
class StepOpPlan:
    step_id: int
    phase: str
    ops: tuple[Operator | Any, ...]
    #: Phase 1 (op_plan §结合实施): step-level dynamic input. When set, CostRouter
    #: feeds it to each op's ``forward(runtime)``; ``None`` keeps the legacy path
    #: (ops carry their dynamic params from construction). Default None so existing
    #: callers are unaffected during migration.
    runtime: "StepRuntime | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)
