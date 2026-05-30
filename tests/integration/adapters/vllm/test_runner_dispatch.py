"""VirtualModelRunner V3 cost engine dispatch — V3 是唯一路径 (Step 4.5 删除 legacy).

不通过完整 vLLM init (无 GPU). 用 mock VllmConfig 拼装 bundle, 单独测 _estimate_cost.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_scenario
from llm_infer_sim.core.cost.engine import build_roofline_engine_from_scenario
from llm_infer_sim.core.cost.trace import StepCostTrace
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def _qwen3_4b_vc():
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-4B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=64),
    )


def _prefill_wl(isl=128):
    return GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(request_id="r", phase=StepPhase.PREFILL,
                                  num_tokens=isl, context_len=0)],
        num_prefill_tokens=isl, total_scheduled_tokens=isl,
        num_prefill_requests=1,
    )


def _decode_wl(n=1, ctx=512):
    return GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(request_id=f"d{i}", phase=StepPhase.DECODE,
                                  num_tokens=1, context_len=ctx) for i in range(n)],
        num_decode_tokens=n, total_scheduled_tokens=n,
        num_decode_requests=n,
    )


def _mixed_wl():
    return GlobalStepWorkload(
        step_id=2, phase=StepPhase.MIXED,
        requests=[
            RequestWorkload(request_id="p", phase=StepPhase.PREFILL,
                            num_tokens=200, context_len=0),
            RequestWorkload(request_id="d1", phase=StepPhase.DECODE,
                            num_tokens=1, context_len=512),
            RequestWorkload(request_id="d2", phase=StepPhase.DECODE,
                            num_tokens=1, context_len=512),
        ],
        num_prefill_tokens=200, num_decode_tokens=2,
        total_scheduled_tokens=202,
        num_prefill_requests=1, num_decode_requests=2,
    )


def _estimate(scenario, workload) -> StepCostTrace:
    engine = build_roofline_engine_from_scenario(scenario)
    return engine.estimate(workload)


# ---- 产出 StepCostTrace ----

@pytest.mark.parametrize("wl", [_prefill_wl(), _decode_wl(n=4, ctx=512)])
def test_v3_estimate_produces_step_cost_trace(wl):
    scenario = extract_scenario(_qwen3_4b_vc())
    trace = _estimate(scenario, wl)
    assert isinstance(trace, StepCostTrace)
    assert trace.total_latency_s > 0
    assert trace.step_id == wl.step_id
    assert trace.phase == wl.phase.value


# ---- mixed phase ----

def test_v3_handles_mixed_phase():
    scenario = extract_scenario(_qwen3_4b_vc())
    trace = _estimate(scenario, _mixed_wl())
    assert trace.total_latency_s > 0
    assert 1e-3 < trace.total_latency_s < 100e-3


# ---- 边界 ----

def test_v3_empty_workload():
    scenario = extract_scenario(_qwen3_4b_vc())
    wl = GlobalStepWorkload(
        step_id=99, phase=StepPhase.DECODE, requests=[],
        num_decode_tokens=0, total_scheduled_tokens=0, num_decode_requests=0,
    )
    trace = _estimate(scenario, wl)
    assert trace.total_latency_s >= 0


def test_v3_large_decode_batch():
    scenario = extract_scenario(_qwen3_4b_vc())
    trace = _estimate(scenario, _decode_wl(n=64, ctx=4096))
    assert trace.total_latency_s > 0
    assert trace.total_latency_s < 1.0


# ---- 字段填充 ----

def test_v3_trace_entries_populated():
    scenario = extract_scenario(_qwen3_4b_vc())
    trace = _estimate(scenario, _prefill_wl(128))
    assert len(trace.entries) >= 1
    for e in trace.entries:
        assert e.latency_s >= 0
        assert e.source in ("roofline", "operator_db")


def test_v3_decode_bottleneck_is_memory():
    scenario = extract_scenario(_qwen3_4b_vc())
    trace = _estimate(scenario, _decode_wl(n=1, ctx=512))
    assert trace.bottleneck == "memory"


def test_v3_long_prefill_bottleneck_is_compute():
    scenario = extract_scenario(_qwen3_4b_vc())
    trace = _estimate(scenario, _prefill_wl(2048))
    assert trace.bottleneck == "compute"
