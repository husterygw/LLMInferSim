"""BackendExecutionProfile — 后端执行策略数据类 (详设 §4.8.1)。

框架无关. 任何"从 framework_config 推导 backend"的逻辑由 adapter 提供:
  - adapters/vllm/profile_extractor.py:_extract_backend_profile()
  - (将来) adapters/sglang/profile_extractor.py:_extract_backend_profile()

阶段 3 范围:
  - mixed_attention.mode: 仅实现 split_kernels (其他 3 种推到 §10.5 子阶段)
  - dense_gemm.mode: merged 默认 (split 极少用, 同样推后)
  - 不实现 YAML 加载, 仅默认 dataclass; 阶段 6/X 才接 YAML

阶段 3.5 重构后:
  - vLLM enum 映射搬到 adapters/vllm/profile_extractor.py (详设 §1.1 架构分层)
  - 本文件只保留框架无关数据类 + default_backend_profile()

详设引用:
  §4.7.1b   MixedAttentionEstimator dispatch on backend.mixed_attention.mode
  §4.6.2    plan_builder dispatch on backend.dense_gemm.mode
  §4.8.1.1  vLLM Backend → mixed_mode 映射表 (实现在 adapters/vllm/)
  §10 阶段 3 / 3.5 列出 mixed_mode/dense_gemm/mixed_attention 字段必须落地
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer_sim.core.cost_model.moe_routing import MoERoutingPolicy


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

    name 用于报告标识 (flash_attn / flashinfer / ...) , 由 adapter 推导;
    default 沿用阶段 3 标签。
    """
    name: str = "flash_attn_paged"
    mixed_attention: MixedAttentionPolicy = field(default_factory=MixedAttentionPolicy)
    dense_gemm: DenseGemmPolicy = field(default_factory=DenseGemmPolicy)
    # 阶段 5-δ: MoE 路由建模 (默认 skew=0.0 = uniform, 阶段 X calibrate)
    moe_routing: MoERoutingPolicy = field(default_factory=MoERoutingPolicy)
    # Phase 5 (通信建模):
    # execution_mode 控制是否加 framework_call_overhead (通信) / kernel_overhead (计算).
    #   "eager"     — 加 per-op dispatch overhead
    #   "cudagraph" — 整图一次 launch, 加 0
    # 从 vllm_config.enforce_eager / compilation_config.cudagraph_mode 推断.
    execution_mode: str = "eager"
    # topology_hint 控制 PCIe 拓扑下 effective_intra_bw 怎么缩.
    #   "concentrated" — n 张卡都在同一 root (默认 / 保守)
    #   "balanced"     — n 张卡均匀分布跨 root
    # 从 LLM_INFER_SIM_NUMA_HINT env 或 CUDA_VISIBLE_DEVICES + gpu_to_root 推断.
    topology_hint: str = "concentrated"


def default_backend_profile() -> BackendExecutionProfile:
    """阶段 3 占位: 硬编码默认 profile, 供 standalone 模式 / 测试 / fallback 使用。

    vLLM 路径在阶段 3.5 起改走 adapters/vllm/profile_extractor.extract_profile_bundle()。
    阶段 6/X 起从 YAML 加载补全其他字段。
    """
    return BackendExecutionProfile()
