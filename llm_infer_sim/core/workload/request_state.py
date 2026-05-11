"""RequestMetrics / SystemMetrics — 详设 §3.4。

阶段 3 落地: 用于 MetricsCollector 跨步状态追踪 + reporter 聚合输出。
所有时间字段一律用 simulator time (秒), 不用 wall-clock —— 这样 instant
TimeEmulator mode 下指标依然有意义。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """单个请求的全生命周期指标 (simulator-time)。"""

    request_id: str
    arrival_time: float = 0.0          # 第一次出现在 step 时的 sim-time (秒)
    first_token_time: float = 0.0      # 首个 output token 产出 sim-time
    completion_time: float = 0.0       # 进入 finished_req_ids 的 sim-time
    output_tokens: int = 0             # 已产出的 output token 数
    target_output_len: int = 0
    per_token_latencies: list[float] = field(default_factory=list)
    completed: bool = False
    saw_first_token: bool = False

    @property
    def ttft(self) -> float:
        """Time To First Token = first_token_time - arrival_time。"""
        if not self.saw_first_token:
            return 0.0
        return max(0.0, self.first_token_time - self.arrival_time)

    @property
    def tpot(self) -> float:
        """Time Per Output Token = mean(per_token_latencies[1:])。"""
        if len(self.per_token_latencies) <= 1:
            return 0.0
        decode_latencies = self.per_token_latencies[1:]
        return sum(decode_latencies) / len(decode_latencies)

    @property
    def e2e_latency(self) -> float:
        if not self.completed:
            return 0.0
        return max(0.0, self.completion_time - self.arrival_time)


@dataclass
class SystemMetrics:
    """系统级聚合指标 — reporter 输出的目标结构。"""

    total_requests: int = 0
    completed_requests: int = 0
    total_output_tokens: int = 0
    total_steps: int = 0
    elapsed_sim_time: float = 0.0

    requests_per_second: float = 0.0
    output_tokens_per_second: float = 0.0

    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0

    avg_tpot: float = 0.0
    p50_tpot: float = 0.0
    p90_tpot: float = 0.0
    p99_tpot: float = 0.0

    avg_e2e: float = 0.0
    p50_e2e: float = 0.0
    p90_e2e: float = 0.0
    p99_e2e: float = 0.0

    avg_step_breakdown: dict[str, float] = field(default_factory=dict)
