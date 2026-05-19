"""RooflineAnalyzer per-op efficiency lookup (B.6).

覆盖:
  1. op.efficiency_key 设了 + efficiency_profile 命中 → raw / efficiency
  2. op.efficiency_key 设了 + miss → fallback hw scalar (raw 不变)
  3. op.efficiency_key=None → 完全不查 (向后兼容)
  4. efficiency_profile=None → 完全不查
  5. _dtype_from_bits 各档
  6. op 构造函数都自动设了 efficiency_key
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.cost_model.roofline import (
    RooflineAnalyzer,
    _dtype_from_bits,
)
from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.ops.linear import (
    fused_gate_up_gemm,
    fused_qkv_gemm,
    linear_layer,
)
from llm_infer_sim.core.ops.normalization import mlp_activation, norm_layer
from llm_infer_sim.core.ops.attention import rope_kernel
from llm_infer_sim.core.profiles.efficiency_profile import (
    EfficiencyEntry,
    EfficiencyProfile,
)
from llm_infer_sim.core.profiles.hardware import get_hardware_profile


# ---- _dtype_from_bits ----

def test_dtype_from_bits_bf16():
    assert _dtype_from_bits(w_bit=16, a_bit=16) == "bfloat16"


def test_dtype_from_bits_fp8():
    assert _dtype_from_bits(w_bit=8, a_bit=8) == "fp8"


def test_dtype_from_bits_fp4():
    assert _dtype_from_bits(w_bit=4, a_bit=4) == "fp4"


def test_dtype_from_bits_uses_max():
    """w=4 a=16 → bf16 (取较大)."""
    assert _dtype_from_bits(w_bit=4, a_bit=16) == "bfloat16"


# ---- op 构造函数都自动设 efficiency_key ----

def test_linear_layer_has_efficiency_key():
    op = linear_layer("q_proj", ic=2560, oc=4096, tokens=128,
                      w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    assert op.efficiency_key == ("dense_gemm", "tokens<=128")


def test_fused_qkv_gemm_has_efficiency_key():
    op = fused_qkv_gemm(
        "qkv_proj", hidden=2560,
        num_q_heads_per_tp=32, num_kv_heads_per_tp=8,
        head_dim=128, tokens=512,
        w_byte=2.0, a_byte=2.0, kv_byte=2.0,
    )
    assert op.efficiency_key == ("dense_gemm", "tokens<=1024")


def test_fused_gate_up_gemm_has_efficiency_key():
    op = fused_gate_up_gemm("gate_up_proj", hidden=2560,
                            intermediate_per_tp=9728, tokens=2048,
                            w_byte=2.0, a_byte=2.0)
    assert op.efficiency_key == ("dense_gemm", "tokens>1024")


def test_norm_layer_has_rmsnorm_key():
    op = norm_layer("attn_norm", tokens=128, hidden_size=2560, a_byte=2.0)
    assert op.efficiency_key == ("rmsnorm", "tokens<=128")


def test_mlp_activation_has_swiglu_key():
    op = mlp_activation("act_fn", tokens=128, hidden_size=2560, a_byte=2.0)
    assert op.efficiency_key == ("swiglu", "tokens<=128")


def test_rope_kernel_has_rope_key():
    op = rope_kernel("rotary_emb", tokens=64,
                     num_q_heads_per_tp=32, num_kv_heads_per_tp=8,
                     head_dim=128, a_byte=2.0)
    assert op.efficiency_key == ("rope", "tokens<=128")  # 64 → tokens<=128


def test_decode_token_lands_in_smallest_bucket():
    """decode (tokens=1): 进 tokens<=16 桶."""
    op = linear_layer("o_proj", ic=4096, oc=2560, tokens=1,
                      w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    assert op.efficiency_key == ("dense_gemm", "tokens<=16")


# ---- RooflineAnalyzer + efficiency_profile ----

def _make_profile_with(entries: list[EfficiencyEntry]) -> EfficiencyProfile:
    p = EfficiencyProfile()
    for e in entries:
        p.add_entry(e)
    return p


def test_no_lookup_when_profile_none():
    """analyzer 没设 efficiency_profile → 走 hw scalar 路径 (raw 不变)."""
    hw = get_hardware_profile("RTX_4090")
    analyzer = RooflineAnalyzer(hw, w_bit=16, a_bit=16, kv_bit=16,
                                 efficiency_profile=None)
    op = linear_layer("q_proj", ic=2560, oc=4096, tokens=128,
                      w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    r = analyzer.analyze(op)
    assert r.total_time > 0


def test_no_lookup_when_op_key_none():
    """op.efficiency_key=None → 即使 profile 有 entry 也不查."""
    hw = get_hardware_profile("RTX_4090")
    profile = _make_profile_with([
        EfficiencyEntry("dense_gemm", "bfloat16", "tokens<=128", efficiency=0.5),
    ])
    analyzer = RooflineAnalyzer(hw, w_bit=16, a_bit=16, kv_bit=16,
                                 efficiency_profile=profile)
    # 手工造一个 op 不带 key
    op = OperatorProfile(name="bare", op_category="matmul", flops=int(1e9),
                         load_weight=int(1e6), load_act=int(1e5))
    r = analyzer.analyze(op)
    # 不应改变 (hw scalar default=1.0)
    raw_compute = 1e9 / hw.effective_peak_flops
    raw_memory = (1e6 + 1e5) / hw.effective_mem_bandwidth
    assert r.t_compute == pytest.approx(raw_compute, rel=1e-3)
    assert r.t_memory == pytest.approx(raw_memory, rel=1e-3)


def test_lookup_refines_with_entry():
    """entry.efficiency=0.5 → t 应放大 2× (vs hw default=1.0)."""
    hw = get_hardware_profile("RTX_4090")
    # 保 hw.compute_efficiency = 1.0
    hw.compute_efficiency = 1.0
    hw.mem_efficiency = 1.0
    profile = _make_profile_with([
        EfficiencyEntry("dense_gemm", "bfloat16", "tokens<=128", efficiency=0.5),
    ])
    analyzer = RooflineAnalyzer(hw, w_bit=16, a_bit=16, kv_bit=16,
                                 efficiency_profile=profile)
    op_raw = linear_layer("test", ic=2560, oc=4096, tokens=128,
                          w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    op_raw.efficiency_key = None  # clear → 不查
    r_raw = analyzer.analyze(op_raw)

    op_with = linear_layer("test", ic=2560, oc=4096, tokens=128,
                           w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    # tokens=128 → tokens<=128 bucket, 命中
    r_with = analyzer.analyze(op_with)
    # ratio = hw.compute_eff / entry.eff = 1.0 / 0.5 = 2.0
    assert r_with.t_compute == pytest.approx(r_raw.t_compute * 2.0, rel=1e-3)
    assert r_with.t_memory == pytest.approx(r_raw.t_memory * 2.0, rel=1e-3)


def test_lookup_miss_falls_back_to_default():
    """entry 不命中 → 用 hw.compute_efficiency 默认, 行为同 no-profile."""
    hw = get_hardware_profile("RTX_4090")
    hw.compute_efficiency = 1.0
    profile = _make_profile_with([
        EfficiencyEntry("dense_gemm", "bfloat16", "tokens<=1024", efficiency=0.5),
    ])
    analyzer = RooflineAnalyzer(hw, w_bit=16, a_bit=16, kv_bit=16,
                                 efficiency_profile=profile)
    # tokens=128 → tokens<=128 bucket, NOT 命中 (entry 是 tokens<=1024)
    op = linear_layer("test", ic=2560, oc=4096, tokens=128,
                      w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    r = analyzer.analyze(op)
    # 也不该 refine, t_compute 是 raw
    raw_compute = op.flops / hw.effective_peak_flops
    assert r.t_compute == pytest.approx(raw_compute, rel=1e-3)


def test_lookup_wildcard_match():
    """entry shape='*' 通配桶 → 任意 token bucket 都命中."""
    hw = get_hardware_profile("RTX_4090")
    hw.compute_efficiency = 1.0
    profile = _make_profile_with([
        EfficiencyEntry("dense_gemm", "bfloat16", "*", efficiency=0.5),
    ])
    analyzer = RooflineAnalyzer(hw, w_bit=16, a_bit=16, kv_bit=16,
                                 efficiency_profile=profile)
    op = linear_layer("test", ic=2560, oc=4096, tokens=128,
                      w_byte=2.0, a_byte=2.0, kv_byte=2.0)
    r = analyzer.analyze(op)
    assert r.total_time > 0
    # 应该被 refine (具体值看 raw × 2.0)
    raw_compute = op.flops / hw.effective_peak_flops
    assert r.t_compute == pytest.approx(raw_compute * 2.0, rel=1e-3)
