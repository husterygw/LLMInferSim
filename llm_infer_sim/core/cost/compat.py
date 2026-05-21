"""Compat shim — Step 4 translation between new StepCostTrace and legacy GlobalStepCost.

Step 4 期间: VirtualModelRunner / MetricsCollector / breakdown 仍消费 GlobalStepCost,
但底层 cost engine 改成 StepCostEngine → StepCostTrace. 本模块负责 trace → cost 转换,
保留 per_layer / per_op 信息供 reporter / metrics 继续工作.

最终 Step 5 之后 GlobalStepCost 可以删除, 由 StepCostTrace 直接驱动 metrics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from llm_infer_sim.core.cost.legacy import GlobalStepCost
from llm_infer_sim.core.cost.trace import StepCostTrace


def step_cost_trace_to_global(
    trace: StepCostTrace,
    *,
    pd_extra_time: float = 0.0,
) -> GlobalStepCost:
    """StepCostTrace → GlobalStepCost (兼容旧 VirtualModelRunner / metrics)."""
    # per_layer breakdown: 把 entries 按 layer_idx 聚合
    per_layer_buckets: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"layer_idx": -1, "ops": [], "t_compute": 0.0,
                 "t_memory": 0.0, "t_comm": 0.0, "t_total": 0.0}
    )
    for entry in trace.entries:
        # entry.metadata 来自 RooflineBackend (V3 §4.5), 含 t_compute / t_memory
        md = entry.metadata
        layer_idx = -1
        # 从 entry.op_name 或 metadata 推 layer_idx; 这里 stage 4 简化:
        # CostTraceEntry 不直接带 layer_idx, 用 -1 作为 model-level bucket.
        b = per_layer_buckets[layer_idx]
        b["layer_idx"] = layer_idx
        b["t_compute"] += float(md.get("t_compute", 0.0))
        b["t_memory"] += float(md.get("t_memory", 0.0))
        b["t_comm"] += float(md.get("t_comm", 0.0))
        b["t_total"] += entry.latency_s
        b["ops"].append({
            "name": entry.op_name,
            "op_kind": entry.op_kind,
            "op_subtype": entry.op_subtype,
            "latency_s": entry.latency_s,
            "source": entry.source,
        })
    per_layer = [per_layer_buckets[i] for i in sorted(per_layer_buckets)]

    return GlobalStepCost(
        step_id=trace.step_id,
        phase=trace.phase,
        total_latency=trace.total_latency_s + pd_extra_time,
        compute_time=trace.compute_time_s,
        memory_time=trace.memory_time_s,
        comm_time=trace.comm_time_s + pd_extra_time,
        per_layer=per_layer,
    )
