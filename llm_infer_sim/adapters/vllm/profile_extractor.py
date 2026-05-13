"""vLLM → ProfileBundle 提取 (详设 §4.8.1.1 + §4.8.3)。

阶段 3.5 重构: 把"读 vllm_config 形状"的全部代码搬到 adapter 层, 与 core/profiles
解耦 (详设 §1.1 架构分层: "core 完全框架无关")。

职责:
  1. extract_profile_bundle(vllm_config): 从 vllm.config.VllmConfig 抽取
     ModelConfig + DeployConfig + HardwareConfig + EfficiencyProfile +
     BackendExecutionProfile, 打包成框架无关的 ProfileBundle 返回。
  2. vLLM AttentionBackendEnum → BackendExecutionProfile 映射表
     (_VLLM_BACKEND_MODE_MAP / _VLLM_UNSUPPORTED_BACKENDS, 详设 §4.8.1.1)。
  3. vLLM hf_config → 框架无关 ModelConfig 字段抽取。

对应详设引用:
  §1.1   架构分层: core 不 import vllm, vllm 形状只在 adapter
  §4.8.1 BackendExecutionProfile (core 数据类)
  §4.8.1.1 vLLM Backend → mixed_mode 映射 (本文件实现)
  §4.8.3 ProfileManager (重命名为 extract_profile_bundle, 实现在 adapter)
"""
from __future__ import annotations

import os

from llm_infer_sim.core.profiles.backend_profile import (
    BackendExecutionProfile,
    MixedAttentionPolicy,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig, ParallelConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_adapters import get_adapter
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.profiles.profile_manager import ProfileBundle


def extract_profile_bundle(vllm_config) -> ProfileBundle:
    """从 vllm.config.VllmConfig 构造框架无关的 ProfileBundle。

    Args:
        vllm_config: vllm.config.VllmConfig (含 model_config / parallel_config /
            cache_config / attention_config 等子配置)。

    Returns:
        ProfileBundle —— 完全脱离 vllm 类型, 后续 cost model / planning 都不再
        感知 vllm 形状。
    """
    # ---- 1. ModelConfig (从 hf_config + model_adapter 提取) ----
    mc = vllm_config.model_config
    hf = mc.hf_config
    model_type = getattr(hf, "model_type", "")
    adapter = get_adapter(model_type)
    model_id = mc.model
    model_config = _extract_model_config(model_id, adapter, hf)

    # ---- 2. ParallelConfig (阶段 4 起 tp>1, 阶段 6 起 ep>1) ----
    pc = vllm_config.parallel_config
    parallel = ParallelConfig(
        tp_size=pc.tensor_parallel_size,
        dp_size=getattr(pc, "data_parallel_size", 1) or 1,
        # vLLM ParallelConfig.enable_expert_parallel: bool, 默认 False
        # 当 True 时, EP group = TP × DP (单节点下 = TP)
        enable_ep=bool(getattr(pc, "enable_expert_parallel", False)),
    )

    # ---- 3. EfficiencyProfile (placeholder 全 1.0) ----
    efficiency = EfficiencyProfile.placeholder()

    # ---- 3.5. Quantization 切 w_byte/a_byte ----
    # 从 hf_config.quantization_config 读 quant_method, 决定全局 weight/activation byte:
    #   fp8 (V3/V4): w=1.0, activation_scheme="dynamic" → a=1.0; "static" → a 也是 1.0
    #   fp4 / nvfp4 / mxfp4: w=0.5, a=0.5 (假设 activation 同精度)
    #   其他 (gptq/awq/None): 不动, 沿用默认 fp16 (w=a=2.0)
    # 注意: kv_byte 由 cache_config (KVCacheSpec) 单独控制, 这里不动.
    qcfg = getattr(hf, "quantization_config", None) or {}
    if isinstance(qcfg, dict):
        quant_method = (qcfg.get("quant_method") or "").lower()
        activation_scheme = (qcfg.get("activation_scheme") or "").lower()
    else:
        quant_method = (getattr(qcfg, "quant_method", "") or "").lower()
        activation_scheme = (getattr(qcfg, "activation_scheme", "") or "").lower()
    # vLLM 会把 model-family-specific 名字写回 (e.g. "deepseek_v4_fp8"),
    # 不只是裸 "fp8" / "fp4"。用子串匹配兜底; "fp4" 优先匹配以免被 "fp8" 撞上.
    if "fp4" in quant_method:
        efficiency.w_byte = 0.5
        efficiency.a_byte = 0.5
    elif "fp8" in quant_method:
        efficiency.w_byte = 1.0
        # activation_scheme="dynamic"/"static" 都是 per-token/tensor fp8 量化
        if activation_scheme in ("dynamic", "static"):
            efficiency.a_byte = 1.0

    # ---- 4. HardwareConfig (默认 H100, env 可覆盖) ----
    hw_name = os.environ.get("LLM_INFER_SIM_HW", "H100")
    hw = get_hardware_profile(hw_name)
    efficiency.apply_to(hw)

    # ---- 5. DeployConfig (跨 step 不变的部分; estimate() 时按 workload 覆盖) ----
    # V4 indexer K cache dtype: 从 attention_config.use_fp4_indexer_cache 推
    # (option C, 阶段 9-β). use_fp4_indexer_cache=True → 0.5B (MXFP4), 默认 1.0B (FP8).
    attn_cfg = getattr(vllm_config, "attention_config", None)
    use_fp4_indexer = bool(getattr(attn_cfg, "use_fp4_indexer_cache", False))
    indexer_kv_byte = 0.5 if use_fp4_indexer else 1.0

    deploy = DeployConfig(
        batch_size=1,                            # 占位
        input_len=1,                             # 占位
        output_len=1,                            # 占位
        w_byte=efficiency.w_byte,
        a_byte=efficiency.a_byte,
        kv_byte=efficiency.kv_byte,
        indexer_kv_byte=indexer_kv_byte,
        parallel=parallel,
        use_flash_attention=True,                # 现代 vLLM 默认 flash
    )

    # ---- 6. BackendExecutionProfile (阶段 3.5: 从 attention_config 推导) ----
    backend = _extract_backend_profile(vllm_config)

    return ProfileBundle(
        model=model_config,
        deploy=deploy,
        hw=hw,
        efficiency=efficiency,
        backend=backend,
    )


def _extract_model_config(model_id, adapter, hf) -> ModelConfig:
    """复制自 llm-viewer get_model_graph._build_model_config (精简版)。

    阶段 2 不支持 V4 sparse / hyper-connections, 那些字段全 0; 阶段 8/9 再开。
    MoE / MLA 字段已经透传 (阶段 5/8 会用)。
    """
    hidden_dim = adapter.get_hidden_size(hf)
    num_heads = adapter.get_num_attention_heads(hf)
    num_kv_heads_raw = adapter.get_num_key_value_heads(hf)
    num_kv_heads = int(round(num_kv_heads_raw)) if num_kv_heads_raw else num_heads
    head_dim_default = hidden_dim // num_heads
    ffn_dim = adapter.get_intermediate_size(hf)
    num_layers = adapter.get_num_hidden_layers(hf)
    vocab_size = adapter.get_vocab_size(hf)

    # 显式 head_dim (Qwen3 的 head_dim ≠ hidden / num_heads)
    explicit_head_dim = getattr(hf, "head_dim", 0) or 0
    head_dim = explicit_head_dim if explicit_head_dim > 0 else head_dim_default

    # MoE 字段 (阶段 5+): 兼容 DeepSeek 与 Qwen 两种命名
    #   DeepSeek-V2/V3: n_routed_experts / n_shared_experts / first_k_dense_replace
    #   Qwen2-MoE / Qwen3-MoE: num_experts / shared_expert_intermediate_size / mlp_only_layers
    n_routed = (
        getattr(hf, "n_routed_experts", 0)
        or getattr(hf, "num_experts", 0)
        or 0
    )
    is_moe = n_routed > 0
    num_activated = getattr(hf, "num_experts_per_tok", 0) or 0
    expert_dim = getattr(hf, "moe_intermediate_size", 0) or 0
    # n_shared: DeepSeek 显式给数, Qwen 用 shared_expert_intermediate_size / expert_dim 推算
    n_shared_explicit = getattr(hf, "n_shared_experts", 0) or 0
    shared_intermediate = getattr(hf, "shared_expert_intermediate_size", 0) or 0
    if n_shared_explicit:
        n_shared = n_shared_explicit
    elif shared_intermediate > 0 and expert_dim > 0:
        n_shared = shared_intermediate // expert_dim
    else:
        n_shared = 0
    # first_moe_layer: DeepSeek 用 first_k_dense_replace, Qwen 用 mlp_only_layers (列表)
    first_k_dense = getattr(hf, "first_k_dense_replace", 0) or 0
    if first_k_dense == 0:
        mlp_only = getattr(hf, "mlp_only_layers", None) or []
        first_k_dense = (max(mlp_only) + 1) if mlp_only else 0

    # MLA 字段 (阶段 8+): DeepSeek-V3 真实激活
    kv_lora_rank = getattr(hf, "kv_lora_rank", 0) or 0
    qk_rope_head_dim = getattr(hf, "qk_rope_head_dim", 0) or 0
    qk_nope_head_dim = getattr(hf, "qk_nope_head_dim", 0) or 0
    kv_latent_dim = (kv_lora_rank + qk_rope_head_dim) if kv_lora_rank > 0 else 0
    # v_head_dim: DeepSeek-V3 config.json 无显式字段, fallback 到 qk_nope_head_dim
    # (V3 modeling.py 中 v_head_dim ≡ qk_nope_head_dim = 128); 仅在两者都不存在时
    # 才用 head_dim (hidden/num_heads). 这条 fallback 链是阶段 8-β 修正:
    # 旧代码默认 head_dim=56(7168/128), 但 V3 真实 v_head_dim = 128.
    v_head_dim_explicit = getattr(hf, "v_head_dim", 0) or 0
    if v_head_dim_explicit > 0:
        v_head_dim = v_head_dim_explicit
    elif qk_nope_head_dim > 0:
        v_head_dim = qk_nope_head_dim   # MLA 默认: v_head_dim ≡ qk_nope_head_dim
    else:
        v_head_dim = 0                  # 非 MLA 模型, layer_builder 退到 head_dim
    # q_lora_rank: DeepSeek-V3 Q 投影也走 LoRA 分解 (1536 in V3); 0 = 直接 hidden→Q proj
    q_lora_rank = getattr(hf, "q_lora_rank", 0) or 0

    # V4 字段 (阶段 9+): sparse attention + HC + grouped O proj.
    # V4 path 触发条件: window_size > 0 AND o_groups > 0.
    # V4-Flash hf_config 真实字段名:
    #   sliding_window=128 (注意映射到 ModelConfig.window_size)
    #   o_groups=8, o_lora_rank=1024
    #   compress_ratios=[0,0,4,128,...] (list, 优先级高于 a/b 派生)
    #   index_topk=512, index_n_heads=64, index_head_dim=128
    #   hc_mult=4, hc_sinkhorn_iters=20
    #   expert_dtype="fp4" → expert_fp4=True
    window_size = getattr(hf, "sliding_window", 0) or 0
    o_lora_rank = getattr(hf, "o_lora_rank", 0) or 0
    o_groups = getattr(hf, "o_groups", 0) or 0
    compress_ratios_list = list(getattr(hf, "compress_ratios", None) or [])
    index_topk = getattr(hf, "index_topk", 0) or 0
    index_n_heads = getattr(hf, "index_n_heads", 0) or 0
    index_head_dim = getattr(hf, "index_head_dim", 0) or 0
    hc_mult = getattr(hf, "hc_mult", 0) or 0
    hc_sinkhorn_iters = getattr(hf, "hc_sinkhorn_iters", 0) or 0
    # expert_fp4: 从 expert_dtype="fp4" 推导 (V4 用); 默认 False
    expert_dtype = getattr(hf, "expert_dtype", "") or ""
    expert_fp4 = (expert_dtype.lower() == "fp4")
    # num_hash_layers: V4 前 N 层用 hash routing (tid2eid lookup, FLOPs≈0)
    num_hash_layers = getattr(hf, "num_hash_layers", 0) or 0

    return ModelConfig(
        name=model_id.split("/")[-1] if isinstance(model_id, str) else "model",
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        is_moe=is_moe,
        num_experts=n_routed,
        num_activated_experts=num_activated,
        expert_dim=expert_dim,
        num_shared_experts=n_shared,
        moe_layer_freq=1,
        first_moe_layer=first_k_dense,
        kv_latent_dim=kv_latent_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        rope_head_dim=qk_rope_head_dim,
        q_lora_rank=q_lora_rank,
        # V4 字段
        window_size=window_size,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        compress_ratios=compress_ratios_list,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        expert_fp4=expert_fp4,
        num_hash_layers=num_hash_layers,
    )


# ============================================================================
# §4.8.1.1 vLLM Backend → mixed_mode 映射 (阶段 3.5)
# ============================================================================

# 主流 NVIDIA backend → (name, mixed_mode) 映射。
# 阶段 3.5 全部映射到 unified_ragged (vLLM 0.20+ 主流 backend 在 mixed batch
# 下都走单 kernel ragged varlen); MLA 系列先占位, 阶段 8 DeepSeek-V3 真实验证。
_VLLM_BACKEND_MODE_MAP: dict[str, tuple[str, str]] = {
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
_VLLM_UNSUPPORTED_BACKENDS: set[str] = {
    "ROCM_ATTN", "ROCM_AITER_MLA", "ROCM_AITER_TRITON_MLA",
    "ROCM_AITER_FA", "ROCM_AITER_MLA_SPARSE", "ROCM_AITER_UNIFIED_ATTN",
    "XPU_MLA_SPARSE", "CPU_ATTN",
    "NO_ATTENTION", "CUSTOM", "TORCH_SDPA",
}


def _vllm_backend_to_mode(backend) -> tuple[str, str]:
    """vLLM AttentionBackendEnum → (name, mode) 映射, 含 None 默认 + fail-fast。

    Returns:
        (backend_name_for_report, mixed_attention_mode)

    Raises:
        NotImplementedError: backend 在 _VLLM_UNSUPPORTED_BACKENDS 或不在
            _VLLM_BACKEND_MODE_MAP 中 (未来新增的 enum)。
    """
    # backend=None: vLLM platform 自动选 (H100 → FLASH_ATTN, B200 → FLASHINFER)
    # 阶段 3.5 简化: 两者形态等价 (都是 unified_ragged), 不复刻 vLLM 启发式。
    if backend is None:
        return ("flash_attn_auto", "unified_ragged")

    name = backend.name
    if name in _VLLM_UNSUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"Attention backend {name} 暂不支持 (阶段 3.5): "
            f"本系统当前仅支持 NVIDIA CUDA / FlashInfer 系列。"
            f"已支持列表: {sorted(_VLLM_BACKEND_MODE_MAP.keys())}。"
            f"替代: 设置 VLLM_ATTENTION_BACKEND=FLASH_ATTN 或留空走默认。"
        )
    if name not in _VLLM_BACKEND_MODE_MAP:
        raise NotImplementedError(
            f"Unknown attention backend {name} (新 enum, 未在 §4.8.1.1 "
            f"_VLLM_BACKEND_MODE_MAP 中映射)。请在 adapters/vllm/profile_extractor.py "
            f"加映射, 或临时设置 VLLM_ATTENTION_BACKEND=FLASH_ATTN 绕过。"
        )
    return _VLLM_BACKEND_MODE_MAP[name]


def _extract_backend_profile(vllm_config) -> BackendExecutionProfile:
    """从 vllm_config.attention_config 推断 BackendExecutionProfile。

    阶段 3.5 范围: 仅推断 mixed_attention.mode 与 name;
    其他字段 (flash_attn_version / use_cudnn_prefill 等) 沿用默认, 推到阶段 X。

    Raises:
        NotImplementedError: 命中 _VLLM_UNSUPPORTED_BACKENDS 或未列出的 enum。
    """
    attn_cfg = getattr(vllm_config, "attention_config", None)
    backend = getattr(attn_cfg, "backend", None) if attn_cfg is not None else None
    name, mode = _vllm_backend_to_mode(backend)
    return BackendExecutionProfile(
        name=name,
        mixed_attention=MixedAttentionPolicy(mode=mode),
    )


__all__ = [
    "extract_profile_bundle",
    # 以下导出供 feature gate (virtual_platform._check_unsupported_features) 用
    "_VLLM_BACKEND_MODE_MAP",
    "_VLLM_UNSUPPORTED_BACKENDS",
]
