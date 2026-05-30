"""DeploymentProfile — 部署聚合 (config_plan §4.3)。

部署只描述并行 / 调度容量 / KV cache 容量 / PD 分离。
execution_mode、framework/backend、prefill overhead 属于 RuntimeProfile, 不在此。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.deployment.kv_cache import KVCacheConfig
from llm_infer_sim.core.deployment.parallelism import ParallelismConfig
from llm_infer_sim.core.deployment.pd_disagg import PDDisaggConfig
from llm_infer_sim.core.deployment.scheduler import SchedulerConfig


@dataclass(frozen=True)
class DeploymentProfile:
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    pd: PDDisaggConfig | None = None

    @classmethod
    def flat(
        cls,
        *,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        ep: int = 1,
        moe_tp: int = 1,
        moe_ep: int = 1,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        block_size: int = 16,
        num_gpu_blocks: int | None = None,
        pd: PDDisaggConfig | None = None,
    ) -> "DeploymentProfile":
        """扁平 kwargs 构造 (测试 / adapter 便捷入口)。"""
        return cls(
            parallelism=ParallelismConfig(
                tp=tp, pp=pp, dp=dp, ep=ep, moe_tp=moe_tp, moe_ep=moe_ep,
            ),
            scheduler=SchedulerConfig(
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
            ),
            kv_cache=KVCacheConfig(
                block_size=block_size, num_gpu_blocks=num_gpu_blocks,
            ),
            pd=pd,
        )
