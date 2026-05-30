"""KernelBackendProfile — attention/gemm kernel policy (config_plan §4.5)。

框架无关. 任何"从 framework_config 推导 kernel policy"的逻辑由 adapter 提供:
  - adapters/vllm/profile_extractor.py:_extract_kernel_profile()
  - (将来) adapters/sglang/profile_extractor.py:_extract_kernel_profile()

字段:
  backend_name   报告标识 (flash_attn / flashinfer / ...)
  mixed_attention / dense_gemm  → mixed batch 执行策略
  moe_routing    → MoE 路由建模假设
  topology_hint  → PCIe 拓扑缩放提示 (长期由 placement resolver 生成, 暂留)

详设引用:
  §4.7.1b   MixedAttentionEstimator dispatch on kernels.mixed_attention.mode
  §4.6.2    plan_builder dispatch on kernels.dense_gemm.mode
  §4.8.1.1  vLLM Backend → mixed_mode 映射表 (实现在 adapters/vllm/)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.operators import MoERoutingProfile


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
    inter_kernel_sync_us: float = 0.0   # 阶段 0-9 = 0 (跟 hw.kernel_overhead={} 同一哲学:
                                        # 不在 cost model 里写未校准折扣). 阶段 X §9.4.2
                                        # calibration 触发时跟 kernel_overhead 字典统一进,
                                        # 不在这里硬编码。
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


@dataclass(frozen=True)
class KernelBackendProfile:
    backend_name: str = "flash_attn_paged"
    mixed_attention: MixedAttentionPolicy = field(default_factory=MixedAttentionPolicy)
    dense_gemm: DenseGemmPolicy = field(default_factory=DenseGemmPolicy)
    moe_routing: MoERoutingProfile = field(default_factory=MoERoutingProfile.balanced)
    topology_hint: str = "concentrated"
