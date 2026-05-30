"""QuantizationProfile — 量化字节宽度 (w/a/kv 每元素字节数)。

从 EfficiencyProfile 拆出 (config_plan §5): EfficiencyProfile 原本同时承载 roofline
efficiency 查表 和 量化字节, 职责混杂。字节是 roofline 公式的纯输入 (经 OperatorContext
喂给各 op), 与 efficiency 校准无关, 故独立成 profile。

adapter (profile_extractor) 从 vLLM quantization_config 推导 w_byte/a_byte/kv_byte;
默认 bf16 (2.0) = 未量化上界。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationProfile:
    """量化层 / activation / KV cache 的 bytes-per-element (bf16=2.0, fp8=1.0, fp4=0.5)。"""

    w_byte: float = 2.0
    a_byte: float = 2.0
    kv_byte: float = 2.0

    @property
    def w_bit(self) -> int:
        return int(self.w_byte * 8)

    @property
    def a_bit(self) -> int:
        return int(self.a_byte * 8)

    @property
    def kv_bit(self) -> int:
        return int(self.kv_byte * 8)

    @classmethod
    def placeholder(cls) -> "QuantizationProfile":
        """默认 bf16 (全 2.0) = 未量化。"""
        return cls()


__all__ = ["QuantizationProfile"]
