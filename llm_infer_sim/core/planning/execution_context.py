"""DistributedExecutionContext — 并行拓扑上下文 (详设 §4.6.1)。

阶段 4 范围:
  - 描述 TP/DP/EP 并行配置 + intra/inter-node 拓扑
  - 通过 properties 暴露 tp_size / ep_size / dp_size 方便上层 dispatch
  - 阶段 4 起从 vllm_config 经 adapter 推导 (与 ProfileBundle 平级)

阶段 6 / 7 起扩展:
  - expert_placement 在 EP 拓扑下生效 (uniform → locality-aware)
  - intra_node_size / inter_node_count 给跨节点通信修正用 (阶段 7)
  - comm_topology dispatch nvlink / pcie / custom (阶段 7)

不在本阶段实现 (placeholder 字段已留):
  - PP / sequence parallel
  - dynamic re-grouping
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.profiles.deploy import ParallelConfig


@dataclass
class DistributedExecutionContext:
    """并行执行拓扑上下文 (详设 §4.6.1)。

    阶段 4: 仅 tp / dp 字段消费; 其他字段留 placeholder 给后续阶段。
    """
    parallel_config: ParallelConfig

    # 阶段 4 主体字段
    world_size: int = 1
    pp_size: int = 1                    # Pipeline Parallelism (§10.5 7.5 子阶段)

    # 通信拓扑 (阶段 7 跨节点修正用)
    intra_node_size: int = 8            # 单节点卡数
    inter_node_count: int = 1           # 节点数
    comm_topology: str = "nvlink"       # nvlink / pcie / custom

    # MoE 专家分布 (阶段 6/7)
    expert_placement: str = "uniform"   # uniform / locality-aware

    # ---- 便利 properties (dispatch on 这些) ----

    @property
    def tp_size(self) -> int:
        return self.parallel_config.tp_size

    @property
    def ep_size(self) -> int:
        return self.parallel_config.ep_size

    @property
    def dp_size(self) -> int:
        return self.parallel_config.dp_size

    @property
    def is_distributed(self) -> bool:
        """是否真实分布式 (world_size > 1)。"""
        return self.world_size > 1


def context_from_parallel_config(
    parallel: ParallelConfig,
    intra_node_size: int = 8,
) -> DistributedExecutionContext:
    """Helper: 从 ParallelConfig 直接构造 context (阶段 4 单节点默认)。"""
    world = parallel.total_devices
    return DistributedExecutionContext(
        parallel_config=parallel,
        world_size=world,
        intra_node_size=min(intra_node_size, world) if world > 0 else intra_node_size,
        inter_node_count=max(1, (world + intra_node_size - 1) // intra_node_size),
    )
