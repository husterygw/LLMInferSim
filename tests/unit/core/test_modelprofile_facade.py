"""config_plan Step F: ModelProfile 扁平 property facade 与 ModelConfig duck-type 等价。

锁住: 模型图 (get_model) / sizing / kv_block_allocator 用 ModelProfile 直读, 结果与
等价 flat ModelConfig byte-identical。facade property 必须逐字段 == to_legacy() 重建值。
"""
from __future__ import annotations

import pytest

# models.config first: 初始化 operators/models/graph 链, 规避 step_plan ↔ models.base
# 的潜在 import cycle (本文件按字母序最先被 collect, 成为链的入口)。
from llm_infer_sim.core.models.config import ModelConfig, ModelProfile
from tests.helpers.support import make_model_config
from llm_infer_sim.core.graph.step_shape import StepShape
from llm_infer_sim.core.models.registry import get_model
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.sizing import (
    estimate_param_bytes,
    estimate_param_count,
    per_rank_param_bytes,
)
from llm_infer_sim.core.simulation.kv_block_allocator import compute_block_bytes
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)

_FLAT_FIELDS = (
    "name", "arch", "hidden_dim", "num_heads", "num_kv_heads", "head_dim",
    "ffn_dim", "num_layers", "vocab_size", "is_moe", "num_experts",
    "num_activated_experts", "expert_dim", "num_shared_experts",
    "moe_layer_freq", "first_moe_layer", "kv_latent_dim", "kv_lora_rank",
    "v_head_dim", "qk_nope_head_dim", "rope_head_dim", "q_lora_rank",
)


def _dense() -> ModelConfig:
    return make_model_config(
        name="Qwen3-4B", arch="Qwen3ForCausalLM",
        hidden_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def _moe() -> ModelConfig:
    return make_model_config(
        name="Qwen3-30B-A3B", arch="Qwen3MoeForCausalLM",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=6144, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0, moe_layer_freq=1,
    )


def _mla() -> ModelConfig:
    return make_model_config(
        name="DeepSeek-V3", arch="DeepseekV3ForCausalLM",
        hidden_dim=7168, num_heads=128, num_kv_heads=128, head_dim=128,
        ffn_dim=18432, num_layers=8, vocab_size=129280,
        is_moe=True, num_experts=256, num_activated_experts=8,
        expert_dim=2048, num_shared_experts=1, moe_layer_freq=1,
        first_moe_layer=3,
        kv_lora_rank=512, v_head_dim=128, qk_nope_head_dim=128,
        rope_head_dim=64, q_lora_rank=1536,
    )


_CASES = pytest.mark.parametrize(
    "make", [_dense, _moe, _mla], ids=["dense", "moe", "mla"]
)


@_CASES
def test_facade_fields_match_flat(make):
    mc = make()
    mp = ModelProfile.from_legacy(mc)
    for f in _FLAT_FIELDS:
        assert getattr(mp, f) == getattr(mc, f), f
    for i in range(mc.num_layers):
        assert mp.is_moe_layer(i) == mc.is_moe_layer(i)


@_CASES
def test_facade_roundtrip_byte_identical(make):
    mc = make()
    assert ModelProfile.from_legacy(mc).to_legacy() == mc


@_CASES
def test_sizing_identical(make):
    mc = make()
    mp = ModelProfile.from_legacy(mc)
    assert estimate_param_count(mp) == estimate_param_count(mc)
    assert estimate_param_bytes(mp) == estimate_param_bytes(mc)
    assert per_rank_param_bytes(mp, 2.0, tp_size=4, ep_size=4) == (
        per_rank_param_bytes(mc, 2.0, tp_size=4, ep_size=4)
    )


@_CASES
def test_block_bytes_identical(make):
    mc = make()
    mp = ModelProfile.from_legacy(mc)
    assert compute_block_bytes(mp, 16, 2.0) == compute_block_bytes(mc, 16, 2.0)


def _prefill_step(isl: int = 128) -> StepShape:
    wl = GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, num_decode_tokens=0,
        total_scheduled_tokens=isl,
        num_prefill_requests=1, num_decode_requests=0,
    )
    return StepShape.from_workload(wl, "eager")


def _op_signature(plan):
    return [(op.__class__.__name__, op.count) for op in plan.ops]


@_CASES
def test_model_graph_op_list_identical(make):
    mc = make()
    mp = ModelProfile.from_legacy(mc)
    deployment = DeploymentProfile.flat(tp=4, ep=4)
    runtime = RuntimeProfile.flat()
    hw = get_hardware_profile("RTX_4090")

    def graph(model):
        ctx = build_operator_context(model, deployment, runtime, hw)
        return get_model(model, ctx).forward(_prefill_step())

    assert _op_signature(graph(mp)) == _op_signature(graph(mc))
