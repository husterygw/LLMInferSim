"""CudaGraphProfile — CUDA graph 捕获策略 (config_plan §4.5)。

当前无 legacy 数据源; execution_mode=="cudagraph" 由 ExecutionProfile 表达。
保留占位供后续 capture-size / piecewise 策略落地。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaGraphProfile:
    enabled: bool = False
