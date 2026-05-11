"""GlobalStepCost — step 级 cost 结构, 阶段 1 起带 compute/memory/comm 三栏。"""
from __future__ import annotations

from dataclasses import dataclass, field


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

    @property
    def bottleneck(self) -> str:
        if self.comm_time > max(self.compute_time, self.memory_time):
            return "communication"
        if self.compute_time > self.memory_time:
            return "compute"
        if self.memory_time > 0.0:
            return "memory"
        return "unknown"
