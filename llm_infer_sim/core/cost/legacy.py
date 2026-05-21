"""Legacy GlobalStepCost / PerRankCost data classes.

Step 4 migration: relocated from `core/cost_model/cost_result.py` so that
metrics / breakdown / VirtualModelRunner stop depending on the to-be-deleted
`core.cost_model` package.

Still used as the public lingua franca between cost engine and metrics during
the transition. Eventually (post Step 4.5) metrics will consume StepCostTrace
directly and this module can be deleted.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerRankCost:
    """单个 rank 的 cost (详设 §3.3)."""
    rank_id: int = 0
    model_core_time: float = 0.0
    runtime_ops_time: float = 0.0
    communication_time: float = 0.0
    total_time: float = 0.0

    attention_time: float = 0.0
    linear_time: float = 0.0
    moe_time: float = 0.0
    norm_time: float = 0.0

    @property
    def compute_time(self) -> float:
        return self.model_core_time + self.runtime_ops_time


@dataclass
class GlobalStepCost:
    step_id: int
    phase: str = "decode"
    total_latency: float = 0.0

    compute_time: float = 0.0
    memory_time: float = 0.0
    comm_time: float = 0.0

    per_layer: list[dict] = field(default_factory=list)

    per_rank_costs: list[PerRankCost] = field(default_factory=list)
    critical_rank: int = 0
    rank_imbalance: float = 0.0

    @property
    def bottleneck(self) -> str:
        if self.comm_time > max(self.compute_time, self.memory_time):
            return "communication"
        if self.compute_time > self.memory_time:
            return "compute"
        if self.memory_time > 0.0:
            return "memory"
        return "unknown"
