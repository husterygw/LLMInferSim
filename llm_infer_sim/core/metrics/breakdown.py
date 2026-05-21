"""Breakdown reporter — compute / memory / comm 三栏 (绝对值, 不打百分比)。

注: roofline 模型下 latency = max(compute, memory)+comm 不是相加。compute 与
memory 是"并行关系"(计算单元 vs 存储单元谁先卡谁决定 latency), 强行报百分比
会让 memory-bound 算子的 memory_time 数字"超过 latency"造成误解。所以这里只
打绝对值 + 显式标 bottleneck。

阶段 1 范围: 单 step format-and-print。阶段 2+ 起会扩展到 per-request /
TTFT / TPOT / throughput 完整 metrics。
"""
from __future__ import annotations

from llm_infer_sim.core.cost.legacy import GlobalStepCost


def format_step_breakdown(cost: GlobalStepCost) -> str:
    """把 GlobalStepCost 渲染成可读字符串 (绝对值, 标 bottleneck)。"""
    return (
        f"step={cost.step_id} phase={cost.phase} "
        f"latency={cost.total_latency * 1e6:8.1f}us | "
        f"compute={cost.compute_time * 1e6:7.1f}us "
        f"memory={cost.memory_time * 1e6:7.1f}us "
        f"comm={cost.comm_time * 1e6:7.1f}us "
        f"| bottleneck={cost.bottleneck}"
    )
