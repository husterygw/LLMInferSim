"""BackendExecutionProfile — 后端执行策略配置 (详设 §4.8.1 / §4.8.1.1)。

阶段 3 范围:
  - mixed_attention.mode: 仅实现 split_kernels (其他 3 种推到 §10.5 子阶段)
  - dense_gemm.mode: merged 默认 (split 极少用, 同样推后)
  - 不实现 YAML 加载, 仅默认 dataclass; 阶段 6/X 才接 YAML

阶段 3.5 范围:
  - 新增 infer_backend_profile_from_vllm(): 从 vllm_config.attention_config.backend
    推导 mixed_attention.mode (详设 §4.8.1.1)
  - default_backend_profile() 保留供 standalone / 测试使用

详设引用:
  §4.7.1b   MixedAttentionEstimator dispatch on backend.mixed_attention.mode
  §4.6.2    plan_builder dispatch on backend.dense_gemm.mode
  §4.8.1.1  vLLM Backend → mixed_mode 映射表
  §10 阶段 3 / 3.5 列出 mixed_mode/dense_gemm/mixed_attention 字段必须落地
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MixedAttentionPolicy:
    """Mixed batch 中 attention 的执行策略 (详设 §4.8.1)。

    阶段 3 仅 split_kernels 可用;
    阶段 3.5 新增 unified_ragged (vLLM 默认 FA varlen / FlashInfer);
    chunked_prefill_interleaved / decode_priority_prefill_append 仍 raise
    NotImplementedError, 推到 §10.5 子阶段。
    """
    mode: str = "split_kernels"
    # split_kernels 模式参数
    prefill_decode_overlap: bool = False
    inter_kernel_sync_us: float = 0.0   # 阶段 0-9 = 0 (跟 EfficiencyProfile.placeholder=1.0
                                        # / hw.kernel_overhead={} 同一哲学: 不在 cost model 里
                                        # 写未校准折扣). 阶段 X §9.4.2 calibration 触发时
                                        # 跟 kernel_overhead 字典统一进, 不在这里硬编码。
    # unified_ragged 模式参数 (阶段 3.5)
    ragged_efficiency: float = 1.0      # 阶段 0-9 = 1.0 (placeholder 哲学一致).
                                        # SM 利用率函数 efficiency(ragged_skew) 推到
                                        # 阶段 X §9.4.2 Layer 1 microbench 拟合。
    # 其他 mode 的参数预留 (阶段 X)
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

    name 用于报告标识 (flash_attn / flashinfer / ...) , 由 §4.8.1.1
    infer_backend_profile_from_vllm 推导;default 沿用阶段 3 标签。
    """
    name: str = "flash_attn_paged"
    mixed_attention: MixedAttentionPolicy = field(default_factory=MixedAttentionPolicy)
    dense_gemm: DenseGemmPolicy = field(default_factory=DenseGemmPolicy)


def default_backend_profile() -> BackendExecutionProfile:
    """阶段 3 占位: 硬编码默认 profile, 供 standalone 模式 / 测试使用。

    vLLM 路径在阶段 3.5 起改走 infer_backend_profile_from_vllm()。
    阶段 6/X 起从 YAML 加载补全其他字段。
    """
    return BackendExecutionProfile()


# ============================================================================
# §4.8.1.1 vLLM Backend → mixed_mode 映射 (阶段 3.5)
# ============================================================================

# 主流 NVIDIA backend → (name, mixed_mode) 映射。
# 阶段 3.5 全部映射到 unified_ragged (vLLM 0.20+ 主流 backend 在 mixed batch
# 下都走单 kernel ragged varlen); MLA 系列先占位, 阶段 8 DeepSeek-V3 真实验证。
_BACKEND_MODE_MAP: dict[str, tuple[str, str]] = {
    "FLASH_ATTN":     ("flash_attn",     "unified_ragged"),
    "FLASHINFER":     ("flashinfer",     "unified_ragged"),
    "TRITON_ATTN":    ("triton_attn",    "unified_ragged"),
    "FLEX_ATTENTION": ("flex_attention", "unified_ragged"),
    "FLASH_ATTN_MLA": ("flash_attn_mla", "unified_ragged"),  # 阶段 8 占位
    "FLASHMLA":       ("flashmla",       "unified_ragged"),  # 阶段 8 占位
    "FLASHINFER_MLA": ("flashinfer_mla", "unified_ragged"),  # 阶段 8 占位
    "TRITON_MLA":     ("triton_mla",     "unified_ragged"),  # 阶段 8 占位
}

# 非 NVIDIA / 特殊 backend, fail-fast。
_UNSUPPORTED_BACKENDS: set[str] = {
    "ROCM_ATTN", "ROCM_AITER_MLA", "ROCM_AITER_TRITON_MLA",
    "ROCM_AITER_FA", "ROCM_AITER_MLA_SPARSE", "ROCM_AITER_UNIFIED_ATTN",
    "XPU_MLA_SPARSE", "CPU_ATTN",
    "NO_ATTENTION", "CUSTOM", "TORCH_SDPA",
}


def _backend_to_mode(backend, vllm_config) -> tuple[str, str]:
    """Backend enum → (name, mode) 映射, 含 None 默认推导 + fail-fast。

    Returns:
        (backend_name_for_report, mixed_attention_mode)

    Raises:
        NotImplementedError: backend 在 _UNSUPPORTED_BACKENDS 或不在
            _BACKEND_MODE_MAP 中 (未来新增的 enum)。
    """
    # backend=None: vLLM platform 自动选 (H100 → FLASH_ATTN, B200 → FLASHINFER)
    # 阶段 3.5 简化: 两者形态等价 (都是 unified_ragged), 不复刻 vLLM 启发式。
    if backend is None:
        return ("flash_attn_auto", "unified_ragged")

    name = backend.name
    if name in _UNSUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"Attention backend {name} 暂不支持 (阶段 3.5): "
            f"本系统当前仅支持 NVIDIA CUDA / FlashInfer 系列。"
            f"已支持列表: {sorted(_BACKEND_MODE_MAP.keys())}。"
            f"替代: 设置 VLLM_ATTENTION_BACKEND=FLASH_ATTN 或留空走默认。"
        )
    if name not in _BACKEND_MODE_MAP:
        raise NotImplementedError(
            f"Unknown attention backend {name} (新 enum, 未在 §4.8.1.1 "
            f"_BACKEND_MODE_MAP 中映射)。请在 backend_profile.py 加映射, "
            f"或临时设置 VLLM_ATTENTION_BACKEND=FLASH_ATTN 绕过。"
        )
    return _BACKEND_MODE_MAP[name]


def infer_backend_profile_from_vllm(vllm_config) -> BackendExecutionProfile:
    """从 vllm_config.attention_config 推断 backend profile (详设 §4.8.1.1)。

    阶段 3.5 范围: 仅推断 mixed_attention.mode 与 name;
    其他字段 (flash_attn_version / use_cudnn_prefill 等) 沿用默认, 推到阶段 X。

    Raises:
        NotImplementedError: 命中 _UNSUPPORTED_BACKENDS 或未列出的 enum。
    """
    attn_cfg = getattr(vllm_config, "attention_config", None)
    backend = getattr(attn_cfg, "backend", None) if attn_cfg is not None else None
    name, mode = _backend_to_mode(backend, vllm_config)
    return BackendExecutionProfile(
        name=name,
        mixed_attention=MixedAttentionPolicy(mode=mode),
    )
