"""CostRouter — V3 §7.1.

阶段 1 简化逻辑 (IMPL_PLAN §1.4 Step 1.6):
    for op in plan.ops:
        if op.op_kind == "collective":
            skip   # 阶段 5 接 CommunicationFormulaBackend
        else:
            RooflineBackend.estimate(op)
    aggregate -> StepCostTrace

阶段 3+ 引入 OperatorDBBackend, 阶段 5 引入 CommunicationFormulaBackend,
阶段 6 引入 ModuleProfileBackend, 优先级见 V3 §7.1.
"""
from __future__ import annotations

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.trace import CostTraceEntry, StepCostTrace
from llm_infer_sim.core.graph.step_plan import StepOpPlan


class CostRouter:
    """阶段 1: 只走 RooflineBackend, 跳过 collective."""

    def __init__(self, roofline: RooflineBackend):
        self.roofline = roofline

    def estimate(self, plan: StepOpPlan) -> StepCostTrace:
        entries: list[CostTraceEntry] = []
        skipped_collective = 0
        for op in plan.ops:
            if op.op_kind == "collective":
                skipped_collective += 1
                continue
            entries.append(self.roofline.estimate(op))

        total = sum(e.latency_s for e in entries)
        compute = sum(float(e.metadata.get("t_compute", 0.0)) for e in entries)
        memory = sum(float(e.metadata.get("t_memory", 0.0)) for e in entries)
        bottleneck = "compute" if compute >= memory else "memory"

        return StepCostTrace(
            step_id=plan.step_id,
            phase=plan.phase,
            total_latency_s=total,
            compute_time_s=compute,
            memory_time_s=memory,
            comm_time_s=0.0,
            runtime_time_s=0.0,
            entries=tuple(entries),
            bottleneck=bottleneck,
        )
