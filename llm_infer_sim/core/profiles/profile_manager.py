"""ProfileManager — 三件套打包: ModelConfig + DeployConfig + HardwareConfig。

职责 (按详设 §4.7 / §4.8 简化版):
  1. 从 vLLM VllmConfig 抽出 ModelConfig (复制 llm-viewer get_model_graph
     ._build_model_config 的核心逻辑)
  2. 从 vLLM ParallelConfig 抽出 ParallelConfig (llm-viewer 风格)
  3. 组装 DeployConfig (含 EfficiencyProfile 转出的 w_byte/a_byte/kv_byte
     + parallel + use_flash_attention 等)
  4. 加载 HardwareConfig (env LLM_INFER_SIM_HW 控制, 默认 H100)
  5. 应用 EfficiencyProfile 系数到 hw

阶段 2 范围:
  - dense + GQA + MHA 路径 (Qwen3-4B / opt-125m 都覆盖)
  - 单卡 (TP=1, no DP/EP) —— TP 推到阶段 4
  - chunked prefill 推到阶段 3 (此处 DeployConfig.use_flash_attention 默认 True)

阶段 4+ 起:
  - ParallelConfig 真实从 vllm 解析 tp/dp/ep size
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from llm_infer_sim.core.profiles.backend_profile import (
    BackendExecutionProfile,
    default_backend_profile,
    infer_backend_profile_from_vllm,
)
from llm_infer_sim.core.profiles.deploy import DeployConfig, ParallelConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig, get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.model_adapters import (
    UnsupportedModelError,
    get_adapter,
)


@dataclass
class ProfileBundle:
    """三件套打包供 cost model 直接使用。"""

    model: ModelConfig
    deploy: DeployConfig
    hw: HardwareConfig
    efficiency: EfficiencyProfile
    backend: BackendExecutionProfile = field(default_factory=default_backend_profile)


class ProfileManager:
    """阶段 2: 从 vLLM 配置三件套构造 (ModelConfig, DeployConfig, hw)。"""

    @staticmethod
    def from_vllm_config(vllm_config) -> ProfileBundle:
        """从 vllm VllmConfig 构造 ProfileBundle。

        Args:
            vllm_config: vllm.config.VllmConfig (含 model_config / parallel_config /
                cache_config / load_config 等子配置)
        """
        # ---- 1. ModelConfig (复制 llm-viewer _build_model_config 核心逻辑) ----
        mc = vllm_config.model_config
        hf = mc.hf_config
        model_type = getattr(hf, "model_type", "")
        adapter = get_adapter(model_type)
        model_id = mc.model

        model_config = _build_model_config(model_id, adapter, hf)

        # ---- 2. ParallelConfig (阶段 2: TP=1) ----
        pc = vllm_config.parallel_config
        parallel = ParallelConfig(
            tp_size=pc.tensor_parallel_size,
            dp_size=getattr(pc, "data_parallel_size", 1) or 1,
            enable_ep=False,
        )

        # ---- 3. EfficiencyProfile (placeholder 全 1.0) ----
        efficiency = EfficiencyProfile.placeholder()

        # ---- 4. HardwareConfig (默认 H100, env 可覆盖) ----
        hw_name = os.environ.get("LLM_INFER_SIM_HW", "H100")
        hw = get_hardware_profile(hw_name)
        efficiency.apply_to(hw)

        # ---- 5. DeployConfig (DeployConfig 是 llm-viewer 复用的运行时部署配置) ----
        # 阶段 2: input_len/output_len 由 cost model 在 estimate() 时根据 workload 注入
        # 这里只放跨 step 不变的部分 (parallel + dtype)
        deploy = DeployConfig(
            batch_size=1,                            # 占位, estimate 时按 workload 覆盖
            input_len=1,                             # 占位
            output_len=1,                            # 占位
            w_byte=efficiency.w_byte,
            a_byte=efficiency.a_byte,
            kv_byte=efficiency.kv_byte,
            parallel=parallel,
            use_flash_attention=True,                # 现代 vLLM 默认 flash
        )

        # 阶段 3.5: 从 vllm_config.attention_config.backend 推导 mixed_mode
        # (详设 §4.8.1.1); 老 default_backend_profile() 保留供 standalone 模式。
        return ProfileBundle(
            model=model_config,
            deploy=deploy,
            hw=hw,
            efficiency=efficiency,
            backend=infer_backend_profile_from_vllm(vllm_config),
        )


def _build_model_config(model_id, adapter, hf) -> ModelConfig:
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

    # MoE 字段 (阶段 5+)
    n_routed = getattr(hf, "n_routed_experts", 0) or 0
    is_moe = n_routed > 0
    num_activated = getattr(hf, "num_experts_per_tok", 0) or 0
    expert_dim = getattr(hf, "moe_intermediate_size", 0) or 0
    n_shared = getattr(hf, "n_shared_experts", 0) or 0
    first_k_dense = getattr(hf, "first_k_dense_replace", 0) or 0

    # MLA 字段 (阶段 8+)
    kv_lora_rank = getattr(hf, "kv_lora_rank", 0) or 0
    qk_rope_head_dim = getattr(hf, "qk_rope_head_dim", 0) or 0
    qk_nope_head_dim = getattr(hf, "qk_nope_head_dim", 0) or 0
    kv_latent_dim = (kv_lora_rank + qk_rope_head_dim) if kv_lora_rank > 0 else 0
    v_head_dim = getattr(hf, "v_head_dim", 0) or 0

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
    )


__all__ = ["ProfileBundle", "ProfileManager", "UnsupportedModelError"]
