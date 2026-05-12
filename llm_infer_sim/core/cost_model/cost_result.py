"""GlobalStepCost — step 级 cost 结构。

阶段 1 起带 compute/memory/comm 三栏;
阶段 4 起加 per_rank_costs (symmetric ranks 假设 → 所有 rank 一致);
阶段 6 起 per_rank_costs 真实 asymmetric (系方 §3.5 Stage B 触发)。

详设引用: §3.3 GlobalStepCost / PerRankCost。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerRankCost:
    """单个 rank 的 cost (详设 §3.3)。

    阶段 4 (symmetric ranks): 所有 rank 这份字段一致, 由 aggregator 复制填充;
    阶段 6 (asymmetric): 每个 rank 拿到不同 expert 子集, 字段真实有差异。
    """
    rank_id: int = 0
    # 阶段 4 主体字段
    model_core_time: float = 0.0     # 模型主体计算时间 (s)
    runtime_ops_time: float = 0.0    # runtime ops (阶段 X)
    communication_time: float = 0.0  # 通信时间 (s)
    total_time: float = 0.0          # = model_core + runtime_ops + communication

    # 细分 (供 reporter / debug 用; 阶段 4 起填)
    attention_time: float = 0.0
    linear_time: float = 0.0
    moe_time: float = 0.0
    norm_time: float = 0.0

    @property
    def compute_time(self) -> float:
        return self.model_core_time + self.runtime_ops_time


@dataclass
class GlobalStepCost:
    step_id: int
    phase: str = "decode"
    total_latency: float = 0.0       # 实际 sleep 用的 step 端到端时间 (s)

    # breakdown 三栏 (阶段 1 起填; 阶段 0 全 0)
    compute_time: float = 0.0        # roofline compute-bound 部分 (s)
    memory_time: float = 0.0         # roofline memory-bound 部分 (s)
    comm_time: float = 0.0           # 通信部分 (阶段 1 = 0; 阶段 4 起非 0)

    # 可选 per-layer breakdown (供详设 §4.10 Reporter 消费)
    per_layer: list[dict] = field(default_factory=list)

    # ---- 阶段 4 起: per-rank 维度 (symmetric ranks 假设, 所有 rank 一致) ----
    per_rank_costs: list[PerRankCost] = field(default_factory=list)
    critical_rank: int = 0           # 最慢 rank id (symmetric 下任意, 阶段 6 起非平凡)
    rank_imbalance: float = 0.0      # max(t) / avg(t) - 1 (阶段 4 = 0, 阶段 6 非零)

    @property
    def bottleneck(self) -> str:
        if self.comm_time > max(self.compute_time, self.memory_time):
            return "communication"
        if self.compute_time > self.memory_time:
            return "compute"
        if self.memory_time > 0.0:
            return "memory"
        return "unknown"
