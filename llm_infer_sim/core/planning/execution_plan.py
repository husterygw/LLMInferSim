"""DistributedExecutionPlan / RankPlan / StageExecution — 详设 §3.2 + §4.6。

阶段 3 范围:
  - 单 rank 占位 (TP=1), per-rank 拆分推到阶段 4
  - mixed step 关键字段: layer_results (dense 部分) + attention_override
  - 阶段 4 起接 §4.7.1 estimate(workload, plan) 接口签名

阶段 4 起 (additive, 不破坏阶段 3 路径):
  - 加 StageExecution dataclass (一个执行阶段, 串行 stages 组成完整 plan)
  - DistributedExecutionPlan.stages list (阶段 3 fallback: 空列表, 走 layer_results 路径)
  - RankPlan 扩展: 加 model_ops / runtime_ops / comm_ops 列表 (symmetric ranks 下所有
    rank 共享同一份, 阶段 6 才真实 per-rank 拆分)

详设引用:
  §3.2  DistributedExecutionPlan / RankPlan / StageExecution dataclass
  §4.6.2 plan_builder 输出本结构
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.cost_model.layer_builder import LayerResult
from llm_infer_sim.core.ops.base import OperatorProfile


@dataclass
class RankPlan:
    """单 rank 的逻辑执行计划 (详设 §3.2)。

    阶段 3: 仅 rank_id + layer_results 占位 (单 rank 走 layer_results 路径)。
    阶段 4: 加 model_ops / runtime_ops / comm_ops 三类 op 列表 (symmetric 起步)。
    阶段 6: per-rank 真实差异化 (asymmetric, 不同 rank 拿不同 expert / shard)。
    """
    rank_id: int = 0
    layer_results: list[LayerResult] = field(default_factory=list)
    # 阶段 4 起 (symmetric 假设下与 layer_results 等价, 留位给阶段 6)
    model_ops: list[OperatorProfile] = field(default_factory=list)
    runtime_ops: list[OperatorProfile] = field(default_factory=list)
    comm_ops: list[OperatorProfile] = field(default_factory=list)


@dataclass
class StageExecution:
    """一个执行阶段 (串行 stages 组成完整 plan, 详设 §3.2)。

    阶段 4 起作为 plan 的二级结构使用; 阶段 3 路径 stages=[] fallback 到
    旧 layer_results 字段。

    is_parallel = True: 所有 rank 并行执行该 stage (取 max(rank_times))
    is_parallel = False: 串行执行 (取 sum(rank_times))
    has_collective = True: 包含 allreduce / alltoall 等 collective comm op
    """
    stage_name: str                              # "attention" / "mlp" / "comm" / ...
    rank_plans: list[RankPlan] = field(default_factory=list)
    is_parallel: bool = True
    has_collective: bool = False
    collective_type: str = ""                    # allreduce / alltoall / allgather / ""


@dataclass
class DistributedExecutionPlan:
    """全局执行计划 (详设 §3.2)。

    阶段 3 字段含义 (mixed step path):
      layer_results        — dense GEMM/FFN/comm 部分按 LayerResult 聚合
      attention_override   — mixed step 用 MixedAttentionEstimator 输出覆盖 attention
                              {"per_layer_time": float, "strategy": str, "breakdown": dict}
                              None 表示沿用 layer_results 中的 attention (PREFILL/DECODE 走这条)
      extra_runtime_time   — 额外 kernel launch / sync overhead (秒)

    阶段 4 起 (additive):
      stages              — 串行 stages 列表 (空 = fallback 到 layer_results 路径)
      execution_context   — 并行拓扑信息 (None = 不感知, 阶段 4 起赋值)
    """
    step_id: int = 0
    world_size: int = 1
    mixed_mode: str = ""             # 仅 mixed step 非空, 取 backend.mixed_attention.mode
    layer_results: list[LayerResult] = field(default_factory=list)
    attention_override: dict[str, Any] | None = None
    extra_runtime_time: float = 0.0

    # ---- 阶段 4 起 (additive) ----
    stages: list[StageExecution] = field(default_factory=list)
    execution_context: Any = None    # DistributedExecutionContext, 避免循环 import 用 Any
