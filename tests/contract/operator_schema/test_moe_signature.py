"""MoE canonicalizer 单测 — Step 2.3 + moe_plan Phase 2.

锁住:
  - balanced 与 power_law signature 不同 (IMPL_PLAN §2.4 测试点 2)
  - power_law alpha 不同 → 不同 signature
  - tp / ep 都进 parallel key
  - collector ↔ runtime signature 一致 (Stage 4 后才会真正生成 runtime MoE op)
  - moe_plan Phase 2: AIC-aligned raw params 转 internal canonical 等价于 legacy raw
  - moe_plan Phase 2: AIC distribution 单字段 ("balanced", "power_law_1.01", "power_law_1.2") 解码正确
  - moe_plan Phase 2: latency 单位约定 (metrics.latency_us_*) 不被打破
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operator_schema.moe import (
    _aic_distribution_to_internal,
    _moe_dtype_to_internal,
    moe_case_params_to_signature,
    moe_operator_to_signature,
)
from llm_infer_sim.core.operators import MoE
from llm_infer_sim.core.operators.base import RooflineSpec


_CTX = dict(framework="vllm", framework_version="0.20.1", kernel_source="vllm_fused_moe")


def _moe_case(routing="balanced", alpha=0.0, tp=1, ep=4, mode="eager"):
    """Legacy internal-format raw params (用于 _legacy archive 数据兼容验证)."""
    return {
        "num_tokens": 128, "hidden": 2048,
        "moe_intermediate": 768, "topk": 8, "num_experts": 128,
        "tp": tp, "ep": ep,
        "routing_distribution": routing,
        "power_law_alpha": alpha,
        "dtype": "bf16", "execution_mode": mode,
    }


def _aic_moe_case(distribution="balanced", moe_tp=1, moe_ep=4, mode="eager", moe_dtype="bfloat16"):
    """AIC-aligned raw params (Phase 2 新采集格式)."""
    return {
        "num_tokens": 128, "hidden_size": 2048,
        "inter_size": 768, "topk": 8, "num_experts": 128,
        "moe_tp_size": moe_tp, "moe_ep_size": moe_ep,
        "distribution": distribution,
        "moe_dtype": moe_dtype, "execution_mode": mode,
    }


def _moe_ctx(tp=1, ep=4, mode="eager"):
    from llm_infer_sim.core.operators.context import build_operator_context
    from llm_infer_sim.core.deployment.profile import DeploymentProfile
    from llm_infer_sim.core.runtime.profile import RuntimeProfile
    from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
    from tests.helpers.support import make_model_config
    return build_operator_context(
        make_model_config(),
        DeploymentProfile.flat(tp=tp, ep=ep),
        RuntimeProfile.flat(
            execution_mode=mode, backend="vllm", backend_version="0.20.1",
        ),
        get_hardware_profile("RTX_4090"),
    )


def _moe_op(routing="balanced", alpha=0.0, tp=1, ep=4, mode="eager"):
    return MoE(
        name="fused_moe", op_subtype="fused_moe",
        phase="prefill", layer_idx=0,
        num_tokens=128, hidden=2048,
        moe_intermediate=768, topk=8, num_experts=128,
        routing_distribution=routing,
        power_law_alpha=alpha,
        ctx=_moe_ctx(tp=tp, ep=ep, mode=mode),
        roofline_spec_value=RooflineSpec(flops=1, op_category="matmul"),
    )


def test_collector_and_runtime_signature_match():
    sig_c = moe_case_params_to_signature(_moe_case(), **_CTX)
    sig_r = moe_operator_to_signature(_moe_op())
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def _routed_experts_op(tp, ep, mode="eager"):
    """Production-shaped MoE op: op_subtype='routed_experts' + GLOBAL tp/ep.

    This is what qwen3_moe builds (MoE.routed_experts), as opposed to the
    hand-fed MoE(op_subtype='fused_moe', tp=post-shard) in _moe_op().
    """
    return MoE(
        name="routed_experts", op_subtype="routed_experts",
        phase="prefill", layer_idx=0,
        num_tokens=128, hidden=2048,
        moe_intermediate=768, topk=8, num_experts=128,
        routing_distribution="balanced", power_law_alpha=0.0,
        ctx=_moe_ctx(tp=tp, ep=ep, mode=mode),
        roofline_spec_value=RooflineSpec(flops=1, op_category="matmul"),
    )


def test_routed_experts_signs_as_fused_moe_canonical():
    """Production op (routed_experts subtype) must sign with canonical fused_moe
    subtype — the graph node name must not leak into the signature."""
    sig = moe_operator_to_signature(_routed_experts_op(tp=4, ep=4))
    assert sig.op_subtype == "fused_moe"


def test_ep_path_signs_moe_tp_1_matching_collector():
    """EP path: global tp=4, ep=4. Collector stored moe_tp_size=1, moe_ep_size=4.
    The query signature must use per-kernel sharding (tp=1), not global tp=4."""
    sig_q = moe_operator_to_signature(_routed_experts_op(tp=4, ep=4))
    sig_db = moe_case_params_to_signature(
        _aic_moe_case(distribution="balanced", moe_tp=1, moe_ep=4, mode="eager"),
        **_CTX,
    )
    assert dict(sig_q.parallel) == {"tp": 1, "ep": 4}
    assert sig_q == sig_db
    assert sig_q.stable_hash() == sig_db.stable_hash()


def test_tp_only_moe_signs_moe_tp_equals_tp():
    """TP-only MoE (ep=1): experts TP-sharded → moe_tp = tp. Collector stored
    moe_tp_size=4, moe_ep_size=1."""
    sig_q = moe_operator_to_signature(_routed_experts_op(tp=4, ep=1))
    sig_db = moe_case_params_to_signature(
        _aic_moe_case(distribution="balanced", moe_tp=4, moe_ep=1, mode="eager"),
        **_CTX,
    )
    assert dict(sig_q.parallel) == {"tp": 4, "ep": 1}
    assert sig_q == sig_db


def test_balanced_vs_power_law_differ():
    sig_b = moe_case_params_to_signature(
        _moe_case(routing="balanced", alpha=0.0), **_CTX,
    )
    sig_p = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.2), **_CTX,
    )
    assert sig_b != sig_p


def test_power_law_alpha_difference_changes_signature():
    """power_law alpha=1.01 vs 1.2 应该是不同 signature."""
    sig_a = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.01), **_CTX,
    )
    sig_b = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.2), **_CTX,
    )
    assert sig_a != sig_b


def test_tp_and_ep_in_parallel_key():
    sig_a = moe_case_params_to_signature(_moe_case(tp=1, ep=4), **_CTX)
    sig_b = moe_case_params_to_signature(_moe_case(tp=4, ep=1), **_CTX)
    assert sig_a != sig_b


def test_eager_cudagraph_differ():
    sig_e = moe_case_params_to_signature(_moe_case(mode="eager"), **_CTX)
    sig_g = moe_case_params_to_signature(_moe_case(mode="cudagraph"), **_CTX)
    assert sig_e != sig_g


# ---------------------------------------------------------------------------
# moe_plan Phase 2: AIC-aligned raw params 转 internal canonical
# ---------------------------------------------------------------------------

def test_aic_distribution_balanced_decodes_correctly():
    rd, alpha = _aic_distribution_to_internal("balanced")
    assert rd == "balanced"
    assert alpha == 0.0


def test_aic_distribution_power_law_decodes_alpha():
    rd, alpha = _aic_distribution_to_internal("power_law_1.01")
    assert rd == "power_law"
    assert alpha == pytest.approx(1.01)
    rd2, alpha2 = _aic_distribution_to_internal("power_law_1.2")
    assert rd2 == "power_law"
    assert alpha2 == pytest.approx(1.2)


def test_aic_distribution_invalid_raises():
    with pytest.raises(ValueError):
        _aic_distribution_to_internal("unknown_routing")
    with pytest.raises(ValueError):
        _aic_distribution_to_internal("power_law_abc")    # alpha 非数


def test_moe_dtype_mapping_bf16_fp16():
    assert _moe_dtype_to_internal("bfloat16") == "bf16"
    assert _moe_dtype_to_internal("float16") == "fp16"
    # 未映射的 dtype 原样返 (后续 fp8/fp4 phase 扩展)
    assert _moe_dtype_to_internal("fp8") == "fp8"


def test_aic_params_to_signature_equivalent_to_legacy():
    """AIC raw params 转出的 signature 必须跟 legacy raw params 等价 — 单点 importer 不漂."""
    sig_aic = moe_case_params_to_signature(
        _aic_moe_case(distribution="balanced", moe_tp=4, moe_ep=1, mode="cudagraph"),
        **_CTX,
    )
    sig_legacy = moe_case_params_to_signature(
        _moe_case(routing="balanced", alpha=0.0, tp=4, ep=1, mode="cudagraph"),
        **_CTX,
    )
    assert sig_aic == sig_legacy
    assert sig_aic.stable_hash() == sig_legacy.stable_hash()


def test_aic_power_law_params_to_signature_equivalent_to_legacy():
    sig_aic = moe_case_params_to_signature(
        _aic_moe_case(distribution="power_law_1.2", moe_tp=1, moe_ep=4, mode="cudagraph"),
        **_CTX,
    )
    sig_legacy = moe_case_params_to_signature(
        _moe_case(routing="power_law", alpha=1.2, tp=1, ep=4, mode="cudagraph"),
        **_CTX,
    )
    assert sig_aic == sig_legacy


def test_aic_runtime_signature_match_with_op():
    """走 AIC raw params 产的 signature 必须跟 runtime FusedMoE op 产的 signature 一致."""
    sig_aic = moe_case_params_to_signature(
        _aic_moe_case(distribution="balanced", moe_tp=1, moe_ep=4, mode="eager"),
        **_CTX,
    )
    sig_op = moe_operator_to_signature(_moe_op())
    assert sig_aic == sig_op


# ---------------------------------------------------------------------------
# moe_plan §3.6.3: latency 单位约定 — RawRecord.metrics.latency_us_* 固定 us
# (AIC helper output 是 ms, 但 LLMInferSim 内部 collector 写的就是 us;
#  外部 AIC csv importer 后续接入时必须 ×1000, plan §3.6.3 已锁该规则)
# ---------------------------------------------------------------------------

def test_metrics_latency_unit_is_microseconds():
    """RawRecord.Metrics 字段名约定: latency_us_*, 不能改成 latency_ms_* 或 latency_*.

    这条 lock 防止后续 import AIC ms 数据时单位混淆: 任何 importer 必须显式做
    ms → us 转换 (× 1000) 再填进 latency_us_*, 不允许在字段名上模糊.
    """
    from collector.schemas import Metrics
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(Metrics)}
    # 必须存在 us 命名
    assert "latency_us_p50" in field_names
    assert "latency_us_p10" in field_names
    assert "latency_us_p90" in field_names
    # 不能出现混淆命名
    for f in field_names:
        assert not f.startswith("latency_ms_"), (
            f"{f}: 不允许 latency_ms_* 命名, 会跟 AIC ms 原始数据混淆 "
            f"(see moe_plan §3.6.3)"
        )
