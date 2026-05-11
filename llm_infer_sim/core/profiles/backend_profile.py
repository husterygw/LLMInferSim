"""BackendExecutionProfile — 后端执行策略配置 (详设 §4.8.1)。

阶段 3 范围:
  - mixed_attention.mode: 仅实现 split_kernels (其他 3 种推到 §10.5 子阶段)
  - dense_gemm.mode: merged 默认 (split 极少用, 同样推后)
  - 不实现 YAML 加载, 仅默认 dataclass; 阶段 6/X 才接 YAML

详设引用:
  §4.7.1b MixedAttentionEstimator dispatch on backend.mixed_attention.mode
  §4.6.2  plan_builder dispatch on backend.dense_gemm.mode
  §10 阶段 3 列出 mixed_mode/dense_gemm/mixed_attention 字段必须落地
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MixedAttentionPolicy:
    """Mixed batch 中 attention 的执行策略 (详设 §4.8.1)。

    阶段 3 仅 split_kernels 可用; 其他 mode 在 MixedAttentionEstimator 里
    raise NotImplementedError 提示推到 §10.5 子阶段。
    """
    mode: str = "split_kernels"
    # split_kernels 模式参数
    prefill_decode_overlap: bool = False
    inter_kernel_sync_us: float = 5.0
    # 其他 mode 的参数预留 (阶段 X)
    ragged_efficiency: float = 0.85
    chunk_size: int = 512
    chunk_decode_interleave: bool = True
    append_overhead_factor: float = 1.1


@dataclass
class DenseGemmPolicy:
    """Dense GEMM 层对 mixed batch 的处理策略 (详设 §4.8.1)。

    merged: 所有 token 合并成一个大 GEMM (M = total_tokens), 主流 backend 默认。
    split:  prefill / decode 分别 GEMM (极少见, 阶段 X)。
    """
    mode: str = "merged"


@dataclass
class BackendExecutionProfile:
    """目标 attention backend 的完整执行策略。

    阶段 3 默认值对齐 vLLM default backend (FlashAttention prefill +
    PagedAttention decode = split_kernels)。
    """
    name: str = "flash_attn_paged"
    mixed_attention: MixedAttentionPolicy = field(default_factory=MixedAttentionPolicy)
    dense_gemm: DenseGemmPolicy = field(default_factory=DenseGemmPolicy)


def default_backend_profile() -> BackendExecutionProfile:
    """阶段 3 占位: 硬编码默认 profile。

    阶段 6/X 起从 YAML 加载, 替换本函数。
    """
    return BackendExecutionProfile()
