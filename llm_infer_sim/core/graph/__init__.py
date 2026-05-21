"""Graph 层 — V3 §4.2 + §4.4: StepShape / StepOpPlan.

Operator 协议已迁移到 llm_infer_sim.core.operators (V3 §4.3 替换 VirtualOp).
"""
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.graph.step_shape import StepShape

__all__ = ["StepShape", "StepOpPlan"]
