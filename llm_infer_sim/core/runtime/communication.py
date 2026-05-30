"""RuntimeCommunicationPolicy — runtime 的通信实现选择 (config_plan §4.5)。

区别于 HardwareProfile.communication (物理链路/协议参数): 这里是 runtime 框架
对 collective 实现的选择 (custom allreduce / flashinfer / symm-mem 等)。当前无
legacy 数据源, 默认 vLLM 行为占位。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeCommunicationPolicy:
    allreduce_implementation_order: tuple[str, ...] = field(default_factory=tuple)
    enable_custom_allreduce: bool = False
    enable_flashinfer_allreduce: bool = False
    enable_symm_mem: bool = False
