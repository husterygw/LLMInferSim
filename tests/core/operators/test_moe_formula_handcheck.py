"""moe_plan Phase 4 数值验收: MoE.routed_experts 公式 1:1 手算对照.

锁住 plan §4 Phase 4 数字 (Qwen3-30B-A3B, TP=4, EP=1):
  decode token=1:
    distinct_experts == 8
    expert_dim_per_device == 192
    expert_flops == 1 × 8 × 3 × 2 × 2048 × 192 == 18,874,368
    expert_weight_read == 8 × 3 × 2048 × 192 × 2 == 18,874,368 (bf16)
  prefill token=128:
    expert_dim_per_device == 192
    expert_flops == 128 × 8 × 3 × 2 × 2048 × 192 == 2,415,919,104
    distinct_experts ≈ 128 × (1 - (1 - 8/128)^128) ≈ 127.97

外加: RooflineBackend 走全链路时 metadata 含 distinct_experts/expert_flops/...
breakdown (Phase 4.2 透传)

数据 dump 到 docs/baselines/moe_formula_handcheck_RTX_4090.json 供 CI lock.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.cost.router import CostRouter
from llm_infer_sim.core.graph.step_plan import StepOpPlan
from llm_infer_sim.core.operators import MoE, MoERoutingProfile
from llm_infer_sim.core.operators.context import build_operator_context
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig


HANDCHECK_DUMP = Path(__file__).resolve().parents[3] / "docs" / "baselines" / "moe_formula_handcheck_RTX_4090.json"


def _qwen3_30b_a3b_ctx(tp: int, ep: int):
    mc = ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8, expert_dim=768,
        num_shared_experts=0, moe_layer_freq=1, first_moe_layer=0,
    )
    deploy = DeployConfig(
        tp_size=tp, ep_size=ep,
        backend="vllm", backend_version="0.20.1",
        execution_mode="cudagraph",
    )
    return build_operator_context(mc, deploy, get_hardware_profile("RTX_4090"))


# ---------------------------------------------------------------------------
# Decode handcheck (plan §4 Phase 4)
# ---------------------------------------------------------------------------

def test_decode_token1_tp4_ep1_handcheck():
    ctx = _qwen3_30b_a3b_ctx(tp=4, ep=1)
    moe = MoE.routed_experts(
        layer_idx=0, tokens=1, ctx=ctx,
        routing=MoERoutingProfile.balanced(), phase="decode",
    )
    br = moe.formula_breakdown()
    assert br["distinct_experts"] == 8.0, br
    assert br["expert_dim_per_device"] == 192, br        # = 768 // 4
    assert br["tokens_per_device"] == 1, br
    assert br["expert_flops"] == 1 * 8 * 3 * 2 * 2048 * 192, br  # = 18_874_368
    assert br["expert_flops"] == 18_874_368, br
    # weight_read: distinct(8) × num_gemms(3) × hidden × eddim_per_dev × w_byte(2) / ep(1)
    assert br["expert_weight_read"] == 8 * 3 * 2048 * 192 * 2, br  # = 18_874_368
    assert br["routing_skew"] == 0.0, br


# ---------------------------------------------------------------------------
# Prefill handcheck (plan §4 Phase 4)
# ---------------------------------------------------------------------------

def test_prefill_token128_tp4_ep1_handcheck():
    ctx = _qwen3_30b_a3b_ctx(tp=4, ep=1)
    moe = MoE.routed_experts(
        layer_idx=0, tokens=128, ctx=ctx,
        routing=MoERoutingProfile.balanced(), phase="prefill",
    )
    br = moe.formula_breakdown()
    assert br["expert_dim_per_device"] == 192, br
    assert br["tokens_per_device"] == 128, br
    assert br["expert_flops"] == 128 * 8 * 3 * 2 * 2048 * 192, br  # = 2_415_919_104
    assert br["expert_flops"] == 2_415_919_104, br
    # balanced coupon-collector ≈ 127.97 (近乎全 expert hit)
    assert 127.0 < br["distinct_experts"] < 128.0, br


# ---------------------------------------------------------------------------
# EP=4 path (per-rank shrink 验证)
# ---------------------------------------------------------------------------

def test_ep4_token16_tp1_handcheck():
    """EP=4: expert_dim 不切 TP, flops/weight 按 ep 分 1/4 给 rank."""
    ctx = _qwen3_30b_a3b_ctx(tp=1, ep=4)
    moe = MoE.routed_experts(
        layer_idx=0, tokens=16, ctx=ctx,
        routing=MoERoutingProfile.balanced(), phase="prefill",
    )
    br = moe.formula_breakdown()
    # ep>1 时 expert_dim_per_device 不切 (整 expert_dim)
    assert br["expert_dim_per_device"] == 768, br
    # tokens_per_device = tokens × top_k / ep = 16 × 8 / 4 = 32
    assert br["tokens_per_device"] == 32, br
    # flops = tokens × top_k × 3 × 2 × hidden × expert_dim / ep
    # = 16 × 8 × 3 × 2 × 2048 × 768 / 4
    assert br["expert_flops"] == 16 * 8 * 3 * 2 * 2048 * 768 // 4, br


# ---------------------------------------------------------------------------
# RooflineBackend metadata 透传 (Phase 4.2)
# ---------------------------------------------------------------------------

def test_metadata_breakdown_carried_through_roofline_backend():
    """走 RooflineBackend.estimate() 全链路, metadata 含 formula breakdown 字段."""
    ctx = _qwen3_30b_a3b_ctx(tp=4, ep=1)
    moe = MoE.routed_experts(
        layer_idx=0, tokens=1, ctx=ctx,
        routing=MoERoutingProfile.balanced(), phase="decode",
    )
    deploy = DeployConfig(
        tp_size=4, ep_size=1,
        backend="vllm", backend_version="0.20.1",
        execution_mode="cudagraph",
    )
    backend = RooflineBackend(get_hardware_profile("RTX_4090"), deploy)
    router = CostRouter(backend)
    trace = router.estimate(StepOpPlan(step_id=0, phase="decode", ops=(moe,)))
    md = trace.entries[0].metadata
    # plan Phase 4 metadata fields
    assert "distinct_experts" in md
    assert "expert_dim_per_device" in md
    assert "tokens_per_device" in md
    assert "expert_flops" in md
    assert "expert_weight_read" in md
    assert "routing_skew" in md
    # 数值跟 op.formula_breakdown() 一致
    assert md["distinct_experts"] == 8.0
    assert md["expert_flops"] == 18_874_368
    assert md["expert_weight_read"] == 18_874_368


# ---------------------------------------------------------------------------
# Dump baseline JSON (供 CI lock + 离线对照)
# ---------------------------------------------------------------------------

def test_dump_handcheck_baseline_json():
    """Dump 当前 handcheck 数字到 docs/baselines/moe_formula_handcheck_RTX_4090.json.

    任何后续公式改动只要影响这些字段, 此 test 会因 JSON byte-for-byte 不一致而 fail;
    需要明确意图后更新 baseline JSON.
    """
    ctx_decode = _qwen3_30b_a3b_ctx(tp=4, ep=1)
    decode = MoE.routed_experts(
        layer_idx=0, tokens=1, ctx=ctx_decode,
        routing=MoERoutingProfile.balanced(), phase="decode",
    )
    ctx_prefill = _qwen3_30b_a3b_ctx(tp=4, ep=1)
    prefill = MoE.routed_experts(
        layer_idx=0, tokens=128, ctx=ctx_prefill,
        routing=MoERoutingProfile.balanced(), phase="prefill",
    )
    ctx_ep = _qwen3_30b_a3b_ctx(tp=1, ep=4)
    ep_case = MoE.routed_experts(
        layer_idx=0, tokens=16, ctx=ctx_ep,
        routing=MoERoutingProfile.balanced(), phase="prefill",
    )

    expected = {
        "model": "Qwen3-30B-A3B",
        "hardware": "RTX_4090",
        "moe_plan_phase": "Phase 4",
        "cases": {
            "decode_token1_tp4_ep1": decode.formula_breakdown(),
            "prefill_token128_tp4_ep1": prefill.formula_breakdown(),
            "ep4_token16_tp1": ep_case.formula_breakdown(),
        },
    }

    HANDCHECK_DUMP.parent.mkdir(exist_ok=True, parents=True)
    if HANDCHECK_DUMP.exists():
        on_disk = json.loads(HANDCHECK_DUMP.read_text())
        assert on_disk == expected, (
            f"baseline drift! {HANDCHECK_DUMP} 跟当前 formula 不匹配.\n"
            f"on_disk:\n{json.dumps(on_disk, indent=2)}\n"
            f"current:\n{json.dumps(expected, indent=2)}\n"
            f"如果是 intended 改公式, 删除 baseline json 让 test 重新生成."
        )
    else:
        HANDCHECK_DUMP.write_text(json.dumps(expected, indent=2) + "\n")
        # 首次生成不算 fail, 但提示用户 commit
        pytest.skip(f"baseline json {HANDCHECK_DUMP} 首次生成, 请 commit 锁住")
