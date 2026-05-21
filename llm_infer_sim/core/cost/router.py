"""CostRouter — V3 §7.1 / IMPL_PLAN §3.7.

阶段 3 优先级 (CostPolicy.mode):
    operator_db_first  -> try OperatorDBBackend, miss → RooflineBackend
    roofline_only      -> 不查 DB, 全走 roofline (阶段 1 行为)
    require_operator_db-> 必须 hit, miss → 抛 LookupError

阶段 5 引入 CommunicationFormulaBackend, 阶段 6 引入 ModuleProfileBackend,
按 V3 §7.1 完整优先级排列.

collective op 阶段 1-3 内默认跳过, 直到阶段 5 接通信 backend.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.cost.backends.operator_db import OperatorDBBackend
from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.trace import CostTraceEntry, StepCostTrace
from llm_infer_sim.core.graph.step_plan import StepOpPlan


@dataclass(frozen=True)
class CostPolicy:
    """V3 §7.2 CostPolicy. Stage 3 用到 mode + 两个 backend toggle."""
    mode: str = "operator_db_first"
    enable_operator_db: bool = True
    enable_roofline_fallback: bool = True


class CostRouter:
    """V3 §7.1 router.

    Stage 3 接入 OperatorDBBackend, 但 roofline 仍是 fallback 出口.
    """

    def __init__(
        self,
        roofline: RooflineBackend,
        *,
        operator_db: OperatorDBBackend | None = None,
        policy: CostPolicy | None = None,
    ):
        self.roofline = roofline
        self.operator_db = operator_db
        self.policy = policy or CostPolicy(
            mode="roofline_only" if operator_db is None else "operator_db_first",
            enable_operator_db=operator_db is not None,
        )

    def estimate(self, plan: StepOpPlan) -> StepCostTrace:
        entries: list[CostTraceEntry] = []
        for op in plan.ops:
            if op.op_kind == "collective":
                continue   # 阶段 5 接 CommunicationFormulaBackend
            entry = self._estimate_op(op)
            entries.append(entry)

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

    def _estimate_op(self, op) -> CostTraceEntry:
        mode = self.policy.mode

        if mode == "roofline_only":
            return self.roofline.estimate(op)

        if mode in ("operator_db_first", "require_operator_db"):
            if self.operator_db is not None and self.policy.enable_operator_db:
                hit = self.operator_db.estimate(op)
                if hit is not None:
                    return hit
            if mode == "require_operator_db":
                raise LookupError(
                    f"require_operator_db: no DB hit for op={op.name!r} "
                    f"({op.op_kind}/{op.op_subtype})"
                )
            if not self.policy.enable_roofline_fallback:
                raise LookupError(
                    f"DB miss and roofline fallback disabled: op={op.name!r}"
                )
            return self.roofline.estimate(op)

        raise ValueError(f"unknown CostPolicy.mode: {mode!r}")
