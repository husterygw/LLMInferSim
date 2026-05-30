"""vLLM → SimulationScenario 提取 (V3 §4.8.1.1 + §4.8.3).

把"读 vllm_config 形状"的全部代码集中在 adapter 层, 与 core 解耦
(V3 §1.1 架构分层: core 完全框架无关).

职责:
  1. extract_scenario(vllm_config): 从 vllm.config.VllmConfig 抽取
     ModelProfile + DeploymentProfile + HardwareProfile + RuntimeProfile +
     CalibrationProfile, 组装成框架无关的 SimulationScenario 返回.
  2. vLLM AttentionBackendEnum → KernelBackendProfile 映射表
     (_VLLM_BACKEND_MODE_MAP / _VLLM_UNSUPPORTED_BACKENDS, V3 §4.8.1.1).
  3. vLLM hf_config → 框架无关 ModelConfig 字段抽取.
"""
from __future__ import annotations

import dataclasses
import os

from llm_infer_sim.adapters.vllm.sim_overlay import PDDisaggOverlay, load_sim_overlay
from llm_infer_sim.core.runtime.kernels import (
    KernelBackendProfile,
    MixedAttentionPolicy,
)
from llm_infer_sim.core.calibration import get_calibration_profile
from llm_infer_sim.core.deployment.pd_disagg import PDDisaggConfig
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.hardware import get_hardware_config
from llm_infer_sim.core.hardware.device import HardwareProfile
from llm_infer_sim.core.models.adapters import get_adapter
from llm_infer_sim.core.models.config import ModelConfig, ModelProfile
from llm_infer_sim.core.models.quantization import QuantizationProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.scenario import SimulationScenario


def _parse_profile_parts(vllm_config) -> dict:
    """从 vllm.config.VllmConfig 抽取框架无关的扁平域对象 (生产 + 测试共用解析核心)。

    Args:
        vllm_config: vllm.config.VllmConfig (含 model_config / parallel_config /
            cache_config / attention_config 等子配置)。

    Returns:
        dict(model, deployment, hardware, runtime, calibration) —— 完全脱离 vllm
        类型; extract_scenario 直接 SimulationScenario(**parts)。
    """
    # 可选 YAML overlay (vLLM 推导 < config.yaml < env)。缺文件 → 空 overlay → 行为不变。
    overlay = load_sim_overlay()

    # ---- 1. ModelConfig (从 hf_config + model_adapter 提取) ----
    mc = vllm_config.model_config
    hf = mc.hf_config
    model_type = getattr(hf, "model_type", "")
    adapter = get_adapter(model_type)
    model_id = mc.model
    model_config = _extract_model_config(model_id, adapter, hf)

    # ---- 2. Parallelism (tp / dp / ep) ----
    pc = vllm_config.parallel_config
    tp = int(pc.tensor_parallel_size)
    dp = int(getattr(pc, "data_parallel_size", 1) or 1)
    # vLLM: enable_expert_parallel=True 时 ep = tp × dp; 否则 ep = 1
    ep = (tp * dp) if bool(getattr(pc, "enable_expert_parallel", False)) else 1

    # ---- 3. Quantization 切 w_byte/a_byte (默认 bf16=2.0) ----
    w_byte = a_byte = kv_byte = 2.0
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
        w_byte = 0.5
        a_byte = 0.5
    elif "fp8" in quant_method:
        w_byte = 1.0
        # activation_scheme="dynamic"/"static" 都是 per-token/tensor fp8 量化
        if activation_scheme in ("dynamic", "static"):
            a_byte = 1.0

    # ---- 3.5b. Non-quantized modules 解析 ----
    # compressed-tensors: `ignore` (list of patterns / regex)
    # awq / gptq / bitsandbytes: `modules_to_not_convert` (list of module names)
    covered_non_quantized, unhandled_non_quantized = _classify_non_quantized_modules(qcfg)
    if unhandled_non_quantized:
        # log 让用户知道有 ignore pattern 没被 base 路径覆盖, sizing 会偏差
        import warnings
        warnings.warn(
            f"profile_extractor: 检测到 {len(unhandled_non_quantized)} 个 ignore "
            f"pattern 不在已知 base 集合 (lm_head/embed/norm), sizing 不为它们做 "
            f"bytes correction. patterns: {unhandled_non_quantized}",
            stacklevel=2,
        )

    # ---- 3.6. KV cache dtype 切 kv_byte ----
    # vLLM `cache_config.cache_dtype`:
    #   "auto"        → 跟 model dtype 走 (默认; 这里保留 kv_byte=2.0 不变)
    #   "fp8" / "fp8_e4m3" / "fp8_e5m2"  → 1 byte
    #   "fp16" / "bfloat16"              → 2 bytes
    #   "int8"                           → 1 byte
    cc = getattr(vllm_config, "cache_config", None)
    cache_dtype = (getattr(cc, "cache_dtype", "") or "").lower()
    if "fp8" in cache_dtype or cache_dtype == "int8":
        kv_byte = 1.0
    elif "fp4" in cache_dtype:
        kv_byte = 0.5
    elif cache_dtype in ("fp16", "bfloat16", "float16"):
        kv_byte = 2.0
    # "auto" 或空: 保留默认 kv_byte = 2.0 (fp16)

    # YAML overlay 覆盖量化字节 (overlay 里 auto → None → 不覆盖, 沿用 vLLM 推导)。
    if overlay.quantization.w_byte is not None:
        w_byte = overlay.quantization.w_byte
    if overlay.quantization.a_byte is not None:
        a_byte = overlay.quantization.a_byte
    if overlay.quantization.kv_byte is not None:
        kv_byte = overlay.quantization.kv_byte
    quantization = QuantizationProfile(w_byte=w_byte, a_byte=a_byte, kv_byte=kv_byte)

    # ---- 4. HardwareConfig (默认 H100; vLLM < config.yaml < env) ----
    # env presence detection: env 设了才赢, 否则用 YAML, 再否则默认。
    _env_hw = os.environ.get("LLM_INFER_SIM_HW")
    hw_name = _env_hw if _env_hw is not None else (overlay.hardware.name or "H100")
    hw = get_hardware_config(hw_name)

    # ---- 4b. hw efficiency (preset 默认 1.0; YAML 覆盖, env 最高) ----
    # 临时 sweep knob: LLM_INFER_SIM_MEM_EFFICIENCY 覆盖 mem_efficiency
    # (对标 AIConfigurator mem_bw_empirical_scaling_factor, 0.8 是 H100/A100 经验值)。
    _eff: dict[str, float] = {}
    if overlay.hardware.compute_efficiency is not None:
        _eff["compute_efficiency"] = overlay.hardware.compute_efficiency
    if overlay.hardware.mem_efficiency is not None:
        _eff["mem_efficiency"] = overlay.hardware.mem_efficiency
    if overlay.hardware.comm_efficiency is not None:
        _eff["comm_efficiency"] = overlay.hardware.comm_efficiency
    _mem_eff = os.environ.get("LLM_INFER_SIM_MEM_EFFICIENCY")
    if _mem_eff is not None:
        _eff["mem_efficiency"] = float(_mem_eff)
    if _eff:
        hw = dataclasses.replace(hw, **_eff)

    # ---- 5. PD 分离 (详设 §7.6) ----
    pd_cfg = _extract_pd_config(vllm_config, overlay.pd_disagg)

    # ---- 6. KernelBackendProfile (阶段 3.5: 从 attention_config 推导) ----
    kernel_profile = _extract_kernel_profile(vllm_config, overlay.hardware.topology_hint)

    # ---- 7. DeploymentProfile + RuntimeProfile (config_plan §2) ----
    cache_cfg = getattr(vllm_config, "cache_config", None)
    block_size = int(getattr(cache_cfg, "block_size", 16)) if cache_cfg else 16
    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = (
        int(sched_cfg.max_num_batched_tokens)
        if sched_cfg and getattr(sched_cfg, "max_num_batched_tokens", None) else None
    )
    max_num_seqs = (
        int(sched_cfg.max_num_seqs)
        if sched_cfg and getattr(sched_cfg, "max_num_seqs", None) else None
    )
    backend_version = None
    try:
        import vllm as _vllm
        backend_version = str(_vllm.__version__)
    except Exception:
        pass

    deployment = DeploymentProfile.flat(
        tp=tp,
        pp=1,
        dp=dp,
        ep=ep,
        moe_tp=tp,
        moe_ep=ep,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        num_gpu_blocks=None,
        pd=pd_cfg,
    )
    runtime = RuntimeProfile.flat(
        execution_mode=_infer_execution_mode(vllm_config),  # 单一来源 (cost 读它)
        backend="vllm",
        backend_version=backend_version,
        kernel_profile=kernel_profile,
    )

    # calibration.enabled: YAML 可关标定 (默认开)。无 env 控制。
    _calibrated = True if overlay.calibration.enabled is None else overlay.calibration.enabled

    return dict(
        model=ModelProfile.from_legacy(model_config, quantization),
        deployment=deployment,
        hardware=HardwareProfile.from_legacy(hw),
        runtime=runtime,
        calibration=get_calibration_profile(hw_name, calibrated=_calibrated),
    )


def extract_scenario(vllm_config) -> SimulationScenario:
    """从 vllm.config.VllmConfig 构造结构化 SimulationScenario (config_plan Step 3/8)。

    生产路径入口: 直接从解析出的结构化域对象组装 scenario。
    """
    return SimulationScenario(**_parse_profile_parts(vllm_config))


# 已知的 "永远 base dtype" 模块 (我们 sizing 已经把它们算成 base_w_byte).
# pattern 是 substring match (case-insensitive). 真实 ignore 列表通常含 regex 比如
# "re:.*lm_head", 我们简化成 substring 匹配 (够覆盖 99% 场景).
_KNOWN_BASE_PATTERNS = (
    "lm_head", "embed_tokens", "embedding",
    "layernorm", "rms_norm", "rmsnorm", "norm",
)


def _classify_non_quantized_modules(qcfg) -> tuple[list[str], list[str]]:
    """解析 quantization_config 里的 ignore / modules_to_not_convert.

    返回 (covered, unhandled):
      covered: pattern 命中 _KNOWN_BASE_PATTERNS, sizing 已经把这些层算成 base dtype, no-op
      unhandled: 其他 pattern (例: 某层 q_proj, 某 Linear), 当前 sizing 没建模
                 — 用 list 记录, extract 时 log warn, 让用户感知精度 gap

    覆盖的 quant config schema:
      - compressed-tensors: {"ignore": [str, ...]}    (可能含 "re:..." regex 前缀)
      - awq / gptq / bitsandbytes: {"modules_to_not_convert": [str, ...]}
      - 其他: 不动
    """
    if qcfg is None:
        return [], []
    if isinstance(qcfg, dict):
        ignore = qcfg.get("ignore") or []
        not_convert = qcfg.get("modules_to_not_convert") or []
    else:
        ignore = getattr(qcfg, "ignore", None) or []
        not_convert = getattr(qcfg, "modules_to_not_convert", None) or []
    raw = list(ignore) + list(not_convert)
    if not raw:
        return [], []
    covered: list[str] = []
    unhandled: list[str] = []
    for pat in raw:
        if not isinstance(pat, str):
            continue
        # 剥 regex 前缀 "re:"
        norm = pat[3:] if pat.startswith("re:") else pat
        norm_lower = norm.lower()
        # 任意 known base substring 命中 → covered
        if any(kp in norm_lower for kp in _KNOWN_BASE_PATTERNS):
            covered.append(pat)
        else:
            unhandled.append(pat)
    return covered, unhandled


def _torch_dtype_to_byte(dtype, default: float = 2.0) -> float:
    """torch.dtype → 字节宽度. 不 import torch, 用字符串匹配 (适配 mock 场景)."""
    if dtype is None:
        return default
    name = str(dtype).lower()  # e.g. "torch.bfloat16"
    if any(k in name for k in ("float32", "float64")):
        return 4.0
    if any(k in name for k in ("float16", "bfloat16", "half")):
        return 2.0
    if "float8" in name or "fp8" in name or "int8" in name:
        return 1.0
    if "float4" in name or "fp4" in name:
        return 0.5
    return default


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

    # V4 model (sliding_window>0 + o_groups>0) 已在 #157 删除支持. 检测到 V4 hf_config
    # 字段时显式抛错, 避免静默 fallback 到 V3 path.
    if getattr(hf, "sliding_window", 0) and getattr(hf, "o_groups", 0):
        raise NotImplementedError(
            f"DeepSeek V4 (sliding_window + o_groups) support removed in #157. "
            f"hf_config has sliding_window={hf.sliding_window!r}, o_groups={hf.o_groups!r}. "
            f"V4 will be reimplemented on new operator class architecture."
        )

    architectures = getattr(hf, "architectures", None) or []
    arch = architectures[0] if architectures else ""

    return ModelConfig(
        name=model_id.split("/")[-1] if isinstance(model_id, str) else "model",
        arch=arch,
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


def _extract_kernel_profile(
    vllm_config, topology_hint_overlay: str | None = None
) -> KernelBackendProfile:
    """从 vllm_config 推断 KernelBackendProfile (含 Phase 5 通信建模字段)。

    阶段 3.5: mixed_attention.mode + backend_name
    Phase 5: topology_hint (concentrated/balanced)

    Raises:
        NotImplementedError: 命中 _VLLM_UNSUPPORTED_BACKENDS 或未列出的 enum。
    """
    attn_cfg = getattr(vllm_config, "attention_config", None)
    backend = getattr(attn_cfg, "backend", None) if attn_cfg is not None else None
    name, mode = _vllm_backend_to_mode(backend)
    return KernelBackendProfile(
        backend_name=name,
        mixed_attention=MixedAttentionPolicy(mode=mode),
        topology_hint=_infer_topology_hint(topology_hint_overlay),
    )


def _infer_execution_mode(vllm_config) -> str:
    """从 vllm_config 推 execution_mode (Phase 5)。

    enforce_eager=True → "eager"
    compilation_config.cudagraph_mode in {None, NONE} → "eager"
    其他 → "cudagraph"
    """
    if getattr(vllm_config, "enforce_eager", False):
        return "eager"
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is not None:
        cgm = getattr(cc, "cudagraph_mode", None)
        if cgm is None or str(cgm).endswith("NONE"):
            return "eager"
    return "cudagraph"


def _infer_topology_hint(overlay_hint: str | None = None) -> str:
    """推 topology_hint (Phase 5): vLLM(暂无) < config.yaml < env。

    env presence detection: LLM_INFER_SIM_NUMA_HINT 设了才赢, 否则 YAML, 再否则
    default "concentrated"。暂不解析 CUDA_VISIBLE_DEVICES + gpu_to_root (留 Phase 6)。
    """
    env = os.environ.get("LLM_INFER_SIM_NUMA_HINT")
    if env is not None:
        return env
    if overlay_hint is not None:
        return overlay_hint
    return "concentrated"


def _extract_pd_config(
    vllm_config, overlay: PDDisaggOverlay | None = None
) -> PDDisaggConfig:
    """PD 分离 config — 优先级 vLLM 推导 < config.yaml < env (详设 §7.6)。

    1. vLLM 推导: vllm_config.kv_transfer_config (用户已手动起 real connector,
       multi-proc PD)。**会同时触发 vLLM PD 真路径**, 我们叠加 cost; 但 vLLM 真路径需
       msgpack + connector class 可加载 + 多进程协调, 单进程 demo 通常不工作。
    2. config.yaml: pd_disagg.* 覆盖 (纯 cost path)。
    3. env: `LLM_INFER_SIM_PD_ROLE=kv_producer|kv_consumer|kv_both` (+ CONNECTOR /
       BANDWIDTH_GBPS / LATENCY_US / PARALLEL_SIZE) 最高优先, **不触发 vLLM 真 connector**,
       只走 cost path。推荐用此路径做 cost 评估。
    """
    # 1. vLLM 推导基线
    base = PDDisaggConfig()
    kvt = getattr(vllm_config, "kv_transfer_config", None)
    if kvt is not None:
        role = getattr(kvt, "kv_role", None)
        if role is not None:
            base = PDDisaggConfig(
                role=role,
                connector_name=getattr(kvt, "kv_connector", None),
                kv_parallel_size=int(getattr(kvt, "kv_parallel_size", 1) or 1),
            )

    # 2. config.yaml overlay (非 None 才覆盖)
    if overlay is not None:
        _repl: dict = {}
        if overlay.role is not None:
            _repl["role"] = overlay.role
        if overlay.connector_name is not None:
            _repl["connector_name"] = overlay.connector_name
        if overlay.kv_parallel_size is not None:
            _repl["kv_parallel_size"] = overlay.kv_parallel_size
        if overlay.connector_bandwidth_gbps is not None:
            _repl["connector_bandwidth_gbps"] = overlay.connector_bandwidth_gbps
        if overlay.connector_latency_us is not None:
            _repl["connector_latency_us"] = overlay.connector_latency_us
        if _repl:
            base = dataclasses.replace(base, **_repl)

    # 3. env 最高优先 (整体替换为 sim-only cost path 配置)
    env_role = os.environ.get("LLM_INFER_SIM_PD_ROLE", "").strip()
    if env_role in ("kv_producer", "kv_consumer", "kv_both"):
        env_conn = os.environ.get(
            "LLM_INFER_SIM_PD_CONNECTOR", "P2pNcclConnector"
        ).strip()
        env_bw = os.environ.get("LLM_INFER_SIM_PD_BANDWIDTH_GBPS")
        env_lat = os.environ.get("LLM_INFER_SIM_PD_LATENCY_US")
        base = PDDisaggConfig(
            role=env_role,
            connector_name=env_conn,
            kv_parallel_size=int(os.environ.get("LLM_INFER_SIM_PD_PARALLEL_SIZE", "1")),
            connector_bandwidth_gbps=float(env_bw) if env_bw else None,
            connector_latency_us=float(env_lat) if env_lat else None,
        )

    return base


__all__ = [
    "extract_scenario",
    # 以下导出供 feature gate (virtual_platform._check_unsupported_features) 用
    "_VLLM_BACKEND_MODE_MAP",
    "_VLLM_UNSUPPORTED_BACKENDS",
]
