"""OperatorContext / build_operator_context unit tests (#158)."""
from __future__ import annotations

import dataclasses

import pytest

from llm_infer_sim.core.operators.context import (
    OperatorContext,
    build_operator_context,
)
from llm_infer_sim.core.operators import MoERoutingProfile
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.models.quantization import QuantizationProfile


def _qwen3_4b() -> ModelConfig:
    return make_model_config(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def test_build_operator_context_defaults_placeholder():
    """quantization=None → placeholder (bf16 byte=2.0) 全部 default."""
    ctx = build_operator_context(
        _qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
    )
    assert ctx.w_byte == 2.0
    assert ctx.a_byte == 2.0
    assert ctx.kv_byte == 2.0
    assert ctx.dtype == "bf16"


def test_build_operator_context_picks_up_quantization_bytes():
    """quantization (fp8 全 1.0) → ctx byte=1.0."""
    quant = QuantizationProfile(w_byte=1.0, a_byte=1.0, kv_byte=1.0)
    ctx = build_operator_context(
        _qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
        quantization=quant,
    )
    assert ctx.w_byte == 1.0
    assert ctx.a_byte == 1.0
    assert ctx.kv_byte == 1.0


def test_operator_context_runtime_passthrough_from_deploy():
    deployment = DeploymentProfile.flat(tp=4, ep=2, block_size=16)
    runtime = RuntimeProfile.flat(
        execution_mode="cudagraph", backend="vllm", backend_version="0.19.1",
    )
    ctx = build_operator_context(_qwen3_4b(), deployment, runtime, get_hardware_profile("RTX_4090"))
    assert ctx.framework == "vllm"
    assert ctx.framework_version == "0.19.1"
    assert ctx.execution_mode == "cudagraph"
    assert ctx.tp_size == 4
    assert ctx.ep_size == 2
    assert ctx.block_size == 16


def test_operator_context_is_frozen():
    ctx = build_operator_context(
        _qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.w_byte = 0.5


def test_build_operator_context_carries_routing():
    """routing 作为部署级成本假设随 ctx 传递; 默认 None (MoE 模型退回 balanced())."""
    routing = MoERoutingProfile.balanced()
    ctx = build_operator_context(
        _qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
        routing=routing,
    )
    assert ctx.routing is routing
    # 默认 None
    ctx_default = build_operator_context(
        _qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(),
        get_hardware_profile("RTX_4090"),
    )
    assert ctx_default.routing is None


def test_operator_context_routing_excluded_from_eq():
    """routing 是成本假设非身份键 (compare=False): 仅 routing 不同的 ctx 相等."""
    hw = get_hardware_profile("RTX_4090")
    base = build_operator_context(_qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(), hw)
    skewed = build_operator_context(
        _qwen3_4b(), DeploymentProfile.flat(), RuntimeProfile.flat(), hw,
        routing=MoERoutingProfile(distribution="balanced", skew=1.0),
    )
    assert base == skewed


def test_framework_version_falls_back_to_unknown():
    """deploy.backend_version=None → ctx.framework_version='unknown'."""
    runtime = RuntimeProfile.flat(backend_version=None)
    ctx = build_operator_context(_qwen3_4b(), DeploymentProfile.flat(), runtime, get_hardware_profile("RTX_4090"))
    assert ctx.framework_version == "unknown"
