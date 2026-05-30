"""ExecutionProfile — 执行模式 + worker overhead (config_plan §4.5)。

execution_mode: "eager" / "cudagraph" (单一来源, cost 经此读)。
prefill_worker_overhead_s: 框架在 prefill step 上的常数开销 (roofline 不建模)。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProfile:
    execution_mode: str = "eager"
    prefill_worker_overhead_s: float = 0.005
