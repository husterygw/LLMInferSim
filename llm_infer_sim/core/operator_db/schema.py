"""V3 §8.1 OperatorRecord — DB 一条标准化记录.

字段语义:
    signature       — V3 §5 OperatorSignature; partition 内主键
    hardware        — partition key, e.g. "RTX_4090"
    framework + version + execution_mode + kernel_source 已在 signature.runtime 里;
    顶层重复保留方便快速 filter / debug.

来源:
    collector RawRecord -> collector_v2 importer -> OperatorRecord -> OperatorStore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.signature import OperatorSignature


@dataclass(frozen=True)
class OperatorRecord:
    signature: OperatorSignature
    hardware: str
    framework: str
    framework_version: str
    execution_mode: str
    kernel_source: str

    latency_us_p50: float
    latency_us_p10: float
    latency_us_p90: float
    n_iters: int
    n_warmups: int
    confidence: float = 1.0
    roofline_us: float | None = None
    roofline_gap: float | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_s(self) -> float:
        return self.latency_us_p50 * 1e-6
