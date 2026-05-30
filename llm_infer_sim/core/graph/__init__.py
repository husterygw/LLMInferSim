"""Graph 层 — V3 §4.2: StepShape / StepOpPlan.

Operator 协议在 llm_infer_sim.core.operators (V3 §4.3).
"""
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape

__all__ = ["StepShape", "StepOpPlan"]
