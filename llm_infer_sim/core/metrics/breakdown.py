"""Breakdown reporter — compute / memory / comm 三栏 (绝对值, 不打百分比).

注: roofline 模型下 latency = max(compute, memory) + comm 不是相加. compute 与
memory 是"并行关系" (计算单元 vs 存储单元谁先卡谁决定 latency), 强行报百分比
会让 memory-bound 算子的 memory_time 数字"超过 latency"造成误解. 所以这里只
打绝对值 + 显式标 bottleneck.

Step 4 严格迁移后直接消费 StepCostTrace.
"""
from __future__ import annotations

from llm_infer_sim.core.cost.trace import StepCostTrace


def format_step_breakdown(trace: StepCostTrace) -> str:
    """把 StepCostTrace 渲染成可读字符串 (绝对值, 标 bottleneck)."""
    return (
        f"step={trace.step_id} phase={trace.phase} "
        f"latency={trace.total_latency_s * 1e6:8.1f}us | "
        f"compute={trace.compute_time_s * 1e6:7.1f}us "
        f"memory={trace.memory_time_s * 1e6:7.1f}us "
        f"comm={trace.comm_time_s * 1e6:7.1f}us "
        f"| bottleneck={trace.bottleneck}"
    )
