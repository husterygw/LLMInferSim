"""CostRouter — V3 §7.1 / IMPL_PLAN §3.7.

阶段 3 优先级 (CostPolicy.mode):
    operator_db_first  -> try OperatorDBBackend, miss → RooflineBackend
    roofline_only      -> 不查 DB, 全走 roofline (阶段 1 行为)
    require_operator_db-> 必须 hit, miss → 抛 LookupError

comm_plan Step 3: collective dispatch 收回 RooflineBackend 内部, CostRouter 不再
按 op_kind 特判. collective op 走 OperatorDBBackend (当前 always miss for collective)
→ fallback RooflineBackend._estimate_collective() 跟其它 op 同一条 policy 流水.
"""
from __future__ import annotations

import dataclasses
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
            # 每个 op 定义 forward(runtime): 用 step runtime 绑定本步动态参数。
            # forward() 返回 None 表示该 op 本步不激活 (例如 attention_decode 撞上
            # 纯 prefill step) → 跳过。plan.runtime 为 None 时 (直接构造
            # StepOpPlan(ops=...) 的单元测试) 不绑定, op_runtime=None, backend 用
            # op 构造时 baked 的参数估算。
            op_runtime = None
            fwd = getattr(op, "forward", None)
            plan_runtime = plan.runtime
            if plan_runtime is not None and callable(fwd):
                op_runtime = fwd(plan_runtime)
                if op_runtime is None:
                    continue
            base = self._estimate_op(op, op_runtime)
            count = int(getattr(op, "count", 1))
            entries.append(self._scale_entry(base, count))

        total = sum(e.latency_s for e in entries)
        compute = sum(float(e.metadata.get("t_compute", 0.0)) for e in entries)
        memory = sum(float(e.metadata.get("t_memory", 0.0)) for e in entries)
        comm = sum(float(e.metadata.get("t_comm", 0.0)) for e in entries)
        bottleneck = self._bottleneck(compute, memory, comm)

        return StepCostTrace(
            step_id=plan.step_id,
            phase=plan.phase,
            total_latency_s=total,
            compute_time_s=compute,
            memory_time_s=memory,
            comm_time_s=comm,
            runtime_time_s=0.0,
            entries=tuple(entries),
            bottleneck=bottleneck,
        )

    def _scale_entry(self, base_entry, count):
        """Scale a per-op cost entry by static layer multiplicity ``count``.

        ``count`` 来自 ``op.count`` (模型 _build_ops 把同构 layer 折叠成一个 op +
        重复次数)。count==1 时原样返回, 否则按 count 线性放大 latency / 各时间分量。
        """
        if count == 1:
            return base_entry
        scaled_latency = base_entry.latency_s * count
        scaled_compute = float(base_entry.metadata.get("t_compute", 0.0)) * count
        scaled_memory = float(base_entry.metadata.get("t_memory", 0.0)) * count
        scaled_comm = float(base_entry.metadata.get("t_comm", 0.0)) * count
        new_md = dict(base_entry.metadata)
        new_md["count"] = count
        if "t_compute" in new_md:
            new_md["t_compute"] = scaled_compute
        if "t_memory" in new_md:
            new_md["t_memory"] = scaled_memory
        if "t_comm" in new_md:
            new_md["t_comm"] = scaled_comm
        display = f"{base_entry.op_name}[count={count}]"
        roofline_scaled = (
            base_entry.roofline_s * count
            if base_entry.roofline_s is not None else None
        )
        return dataclasses.replace(
            base_entry,
            latency_s=scaled_latency,
            display_name=display,
            roofline_s=roofline_scaled,
            metadata=new_md,
        )

    def _estimate_op(self, op, op_runtime=None) -> CostTraceEntry:
        # op_runtime: OpRuntime | None. forward() 已绑定本步动态参数时传非 None;
        # 直接构造 StepOpPlan (无 runtime) 时为 None, backend 退回 op 构造时 baked
        # 的参数。policy.mode 决定 OperatorDB / roofline 的查表顺序。
        mode = self.policy.mode

        if mode == "roofline_only":
            return self.roofline.estimate(op, op_runtime)

        if mode in ("operator_db_first", "require_operator_db"):
            if self.operator_db is not None and self.policy.enable_operator_db:
                hit = self.operator_db.estimate(op, op_runtime)
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
            return self.roofline.estimate(op, op_runtime)

        raise ValueError(f"unknown CostPolicy.mode: {mode!r}")

    @staticmethod
    def _bottleneck(compute: float, memory: float, comm: float) -> str:
        values = {
            "compute": compute,
            "memory": memory,
            "communication": comm,
        }
        return max(values, key=values.get)
