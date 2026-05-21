"""4.2: V3 StepCostEngine 端到端 smoke + 跟 legacy ModelCoreCostModel 对照.

不通过 VirtualModelRunner (需 vLLM 完整初始化), 直接拼装:
    vllm_config → extract_profile_bundle → bundle.deploy_v3
                                         → build_qwen_roofline_engine
                                         → engine.estimate(workload) → StepCostTrace
                                         → step_cost_trace_to_global → GlobalStepCost

对照 legacy:
    bundle.deploy (LegacyDeployConfig) → ModelCoreCostModel(bundle).estimate(workload)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost.compat import step_cost_trace_to_global
from llm_infer_sim.core.cost.engine import (
    build_deepseek_roofline_engine,
    build_qwen_roofline_engine,
)
from llm_infer_sim.core.cost_model.model_core import ModelCoreCostModel
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _qwen3_4b_vllm_config(tp_size: int = 1):
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-4B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192, max_num_seqs=64,
        ),
    )


def _qwen3_30b_a3b_vllm_config(tp_size: int = 2):
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32, num_key_value_heads=4,
        hidden_size=2048, num_hidden_layers=48,
        intermediate_size=6144, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, mlp_only_layers=[],
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-30B-A3B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192, max_num_seqs=64,
        ),
    )


def _prefill_workload(isl=128) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, total_scheduled_tokens=isl,
        num_prefill_requests=1,
    )


def _decode_workload(n=1, ctx=128) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id=f"d{i}", phase=StepPhase.DECODE,
            num_tokens=1, context_len=ctx,
        ) for i in range(n)],
        num_decode_tokens=n, total_scheduled_tokens=n,
        num_decode_requests=n,
    )


# ---- V3 path runs ----

def test_v3_qwen3_4b_prefill_runs():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    trace = engine.estimate(_prefill_workload(isl=128))
    assert trace.total_latency_s > 0
    assert trace.compute_time_s > 0
    assert trace.memory_time_s > 0


def test_v3_qwen3_4b_decode_runs():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    trace = engine.estimate(_decode_workload(n=4, ctx=512))
    assert trace.total_latency_s > 0
    assert trace.bottleneck == "memory"   # decode 1 token 应 memory-bound


def test_v3_qwen3_30b_a3b_decode_runs():
    """MoE 模型 (TP=2 EP=1) 走 Qwen template MoE 分支."""
    bundle = extract_profile_bundle(_qwen3_30b_a3b_vllm_config(tp_size=2))
    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    trace = engine.estimate(_decode_workload(n=1, ctx=128))
    assert trace.total_latency_s > 0
    # routed_experts + routed_expert_allreduce + moe_gate 都应在 entries 里
    names = {e.op_subtype for e in trace.entries}
    assert "router" in names                # moe_gate
    assert "fused_moe" in names             # routed_experts


# ---- Translation: StepCostTrace → GlobalStepCost ----

def test_trace_to_global_keeps_total_latency():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    trace = engine.estimate(_prefill_workload(isl=128))
    cost = step_cost_trace_to_global(trace)
    assert cost.total_latency == trace.total_latency_s
    assert cost.compute_time == trace.compute_time_s
    assert cost.memory_time == trace.memory_time_s
    assert cost.step_id == trace.step_id
    assert cost.phase == trace.phase


def test_trace_to_global_pd_extra_time_added():
    """PD 分离场景: pd_extra_time 加到 comm_time 和 total_latency."""
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    trace = engine.estimate(_decode_workload(n=1, ctx=128))
    base_total = trace.total_latency_s
    pd = 1e-3   # 1 ms KV transfer
    cost = step_cost_trace_to_global(trace, pd_extra_time=pd)
    assert cost.total_latency == pytest.approx(base_total + pd, rel=1e-9)
    assert cost.comm_time == pytest.approx(trace.comm_time_s + pd, rel=1e-9)


# ---- Legacy vs V3 magnitude comparison ----

def test_legacy_vs_v3_qwen_4b_decode_same_magnitude():
    """Qwen3-4B decode: legacy vs V3 latency 量级一致 (≤ 5×)."""
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    wl = _decode_workload(n=1, ctx=128)

    # V3 path
    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    v3_trace = engine.estimate(wl)
    v3_total = v3_trace.total_latency_s

    # Legacy path
    legacy_cost_model = ModelCoreCostModel(bundle)
    legacy = legacy_cost_model.estimate(wl)
    legacy_total = legacy["total_time"]

    # 量级 sanity: 两者都 > 0; 比值不应离谱
    assert v3_total > 0
    assert legacy_total > 0
    ratio = max(v3_total, legacy_total) / min(v3_total, legacy_total)
    assert ratio < 5.0, (
        f"V3 vs legacy 量级差太大: v3={v3_total*1e6:.1f}us, "
        f"legacy={legacy_total*1e6:.1f}us, ratio={ratio:.2f}×"
    )


def test_legacy_vs_v3_qwen_4b_prefill_same_magnitude():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    wl = _prefill_workload(isl=512)

    engine = build_qwen_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    v3_total = engine.estimate(wl).total_latency_s

    legacy = ModelCoreCostModel(bundle).estimate(wl)
    legacy_total = legacy["total_time"]

    assert v3_total > 0 and legacy_total > 0
    ratio = max(v3_total, legacy_total) / min(v3_total, legacy_total)
    assert ratio < 5.0, (
        f"prefill ISL=512: v3={v3_total*1e3:.2f}ms, legacy={legacy_total*1e3:.2f}ms"
    )


# ---- DeepSeek-V3 path ----

def _deepseek_v3_vllm_config(tp_size: int = 8):
    hf = SimpleNamespace(
        model_type="deepseek_v3",
        num_attention_heads=128, num_key_value_heads=128,
        hidden_size=7168, num_hidden_layers=61,
        intermediate_size=18432, vocab_size=129280,
        kv_lora_rank=512, q_lora_rank=1536,
        qk_nope_head_dim=128, qk_rope_head_dim=64,
        n_routed_experts=256, num_experts_per_tok=8,
        moe_intermediate_size=2048, n_shared_experts=1, first_k_dense_replace=3,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="deepseek-ai/DeepSeek-V3"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(block_size=64),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384, max_num_seqs=64,
        ),
    )


def test_v3_deepseek_v3_dense_layer_uses_mla():
    """DeepSeek-V3 走 DeepSeekModelTemplate (MLA attention)."""
    bundle = extract_profile_bundle(_deepseek_v3_vllm_config())
    engine = build_deepseek_roofline_engine(bundle.model, bundle.deploy_v3, bundle.hw)
    trace = engine.estimate(_decode_workload(n=1, ctx=128))
    assert trace.total_latency_s > 0
    # MLA-specific subtype 必出现
    subtypes = {e.op_subtype for e in trace.entries}
    assert "q_a_proj" in subtypes
    assert "kv_a_proj_with_mqa" in subtypes
