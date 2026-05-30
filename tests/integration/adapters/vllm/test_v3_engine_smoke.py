"""V3 StepCostEngine 端到端 smoke (Qwen3-dense / Qwen3-MoE / DeepSeek-V3).

直接拼装:
    vllm_config → extract_profile_bundle → bundle.deploy (V3 DeployConfig)
                                         → build_*_roofline_engine
                                         → engine.estimate(workload) → StepCostTrace
"""
from __future__ import annotations

from types import SimpleNamespace

from llm_infer_sim.adapters.vllm.profile_extractor import extract_scenario
from llm_infer_sim.core.cost.engine import build_roofline_engine_from_scenario
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
    scenario = extract_scenario(_qwen3_4b_vllm_config())
    engine = build_roofline_engine_from_scenario(scenario)
    trace = engine.estimate(_prefill_workload(isl=128))
    assert trace.total_latency_s > 0
    assert trace.compute_time_s > 0
    assert trace.memory_time_s > 0


def test_v3_qwen3_4b_decode_runs():
    scenario = extract_scenario(_qwen3_4b_vllm_config())
    engine = build_roofline_engine_from_scenario(scenario)
    trace = engine.estimate(_decode_workload(n=4, ctx=512))
    assert trace.total_latency_s > 0
    assert trace.bottleneck == "memory"


def test_v3_qwen3_30b_a3b_decode_runs():
    """MoE 模型 (TP=2 EP=1) 走 Qwen template MoE 分支."""
    scenario = extract_scenario(_qwen3_30b_a3b_vllm_config(tp_size=2))
    engine = build_roofline_engine_from_scenario(scenario)
    trace = engine.estimate(_decode_workload(n=1, ctx=128))
    assert trace.total_latency_s > 0
    names = {e.op_subtype for e in trace.entries}
    assert "router" in names                # moe_gate
    # moe_plan Phase 3: routed expert op_subtype is "routed_experts" (was "fused_moe").
    # backend 信息走 kernel_source, 不进 op_subtype.
    assert "routed_experts" in names


# ---- DeepSeek-V3 (MLA) ----

def test_v3_deepseek_v3_dense_layer_uses_mla():
    """DeepSeek-V3 走 DeepSeekModel (MLA attention)."""
    scenario = extract_scenario(_deepseek_v3_vllm_config())
    engine = build_roofline_engine_from_scenario(scenario)
    trace = engine.estimate(_decode_workload(n=1, ctx=128))
    assert trace.total_latency_s > 0
    subtypes = {e.op_subtype for e in trace.entries}
    assert "q_a_proj" in subtypes
    assert "kv_a_proj_with_mqa" in subtypes
