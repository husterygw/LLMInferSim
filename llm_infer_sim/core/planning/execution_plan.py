"""DistributedExecutionPlan / RankPlan — 详设 §3.2 + §4.6.

阶段 3 范围:
  - 单 rank 占位 (TP=1), per-rank 拆分推到阶段 4
  - mixed step 关键字段: layer_results (dense 部分) + attention_override (mixed attention)
  - 阶段 4 起接 §4.7.1 estimate(workload, plan) 接口签名

详设引用:
  §3.2  DistributedExecutionPlan / RankPlan dataclass 雏形
  §4.6.2 plan_builder 输出本结构
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.cost_model.layer_builder import LayerResult


@dataclass
class RankPlan:
    """单 rank 的逻辑执行计划占位 (阶段 4 才落地真实 per-rank op 列表)。"""
    rank_id: int = 0
    layer_results: list[LayerResult] = field(default_factory=list)


@dataclass
class DistributedExecutionPlan:
    """全局执行计划 (阶段 3 单 rank, 阶段 4 起 per-rank)。

    阶段 3 字段含义:
      layer_results        — dense GEMM/FFN/comm 部分按 LayerResult 聚合
      attention_override   — mixed step 用 MixedAttentionEstimator 输出覆盖 attention
                              {"per_layer_time": float, "strategy": str, "breakdown": dict}
                              None 表示沿用 layer_results 中的 attention (PREFILL/DECODE 走这条)
      extra_runtime_time   — 额外 kernel launch / sync overhead (秒)
    """
    step_id: int = 0
    world_size: int = 1
    mixed_mode: str = ""             # 仅 mixed step 非空, 取 backend.mixed_attention.mode
    layer_results: list[LayerResult] = field(default_factory=list)
    attention_override: dict[str, Any] | None = None
    extra_runtime_time: float = 0.0
