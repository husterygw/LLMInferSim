"""MoE / future routing profiles."""
from llm_infer_sim.core.operators.routing.moe import (
    MoERoutingProfile,
    estimate_distinct_experts,
)

__all__ = ["MoERoutingProfile", "estimate_distinct_experts"]
