"""OperatorContext / ModelBuildContext / build_operator_context unit tests (#158)."""
from __future__ import annotations

import dataclasses

import pytest

from llm_infer_sim.core.operators.context import (
    ModelBuildContext,
    OperatorContext,
    build_model_build_context,
    build_operator_context,
)
from llm_infer_sim.core.operators import MoERoutingProfile
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.efficiency_profile import EfficiencyProfile
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig


def _qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def test_build_operator_context_defaults_placeholder():
    """efficiency=None → placeholder (全 1.0 byte=2.0 bf16) 全部 deafult."""
    ctx = build_operator_context(
        _qwen3_4b(), DeployConfig(), get_hardware_profile("RTX_4090"),
    )
    assert ctx.w_byte == 2.0
    assert ctx.a_byte == 2.0
    assert ctx.kv_byte == 2.0
    assert ctx.dtype == "bf16"


def test_build_operator_context_picks_up_efficiency_bytes():
    """efficiency.w_byte=1.0 (fp8) → ctx.w_byte=1.0."""
    eff = EfficiencyProfile.placeholder()
    eff.w_byte = 1.0
    eff.a_byte = 1.0
    eff.kv_byte = 1.0
    ctx = build_operator_context(
        _qwen3_4b(), DeployConfig(), get_hardware_profile("RTX_4090"),
        efficiency=eff,
    )
    assert ctx.w_byte == 1.0
    assert ctx.a_byte == 1.0
    assert ctx.kv_byte == 1.0


def test_operator_context_runtime_passthrough_from_deploy():
    deploy = DeployConfig(
        tp_size=4, ep_size=2, execution_mode="cudagraph",
        backend="vllm", backend_version="0.19.1", block_size=16,
    )
    ctx = build_operator_context(_qwen3_4b(), deploy, get_hardware_profile("RTX_4090"))
    assert ctx.framework == "vllm"
    assert ctx.framework_version == "0.19.1"
    assert ctx.execution_mode == "cudagraph"
    assert ctx.tp_size == 4
    assert ctx.ep_size == 2
    assert ctx.block_size == 16


def test_operator_context_is_frozen():
    ctx = build_operator_context(
        _qwen3_4b(), DeployConfig(), get_hardware_profile("RTX_4090"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.w_byte = 0.5


def test_model_build_context_wraps_op_ctx_with_routing():
    mbc = build_model_build_context(
        _qwen3_4b(), DeployConfig(), get_hardware_profile("RTX_4090"),
        routing=MoERoutingProfile.balanced(),
        indexer_kv_byte=1.0,
    )
    assert isinstance(mbc, ModelBuildContext)
    assert isinstance(mbc.op, OperatorContext)
    assert mbc.routing is not None
    assert mbc.indexer_kv_byte == 1.0


def test_model_build_context_routing_optional():
    """dense model 不需要 routing, None 即可."""
    mbc = build_model_build_context(
        _qwen3_4b(), DeployConfig(), get_hardware_profile("RTX_4090"),
    )
    assert mbc.routing is None


def test_framework_version_falls_back_to_unknown():
    """deploy.backend_version=None → ctx.framework_version='unknown'."""
    deploy = DeployConfig(backend_version=None)
    ctx = build_operator_context(_qwen3_4b(), deploy, get_hardware_profile("RTX_4090"))
    assert ctx.framework_version == "unknown"
