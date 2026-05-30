"""GlobalStepWorkload / RequestWorkload / StepPhase 数据结构。"""
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def test_step_phase_values():
    assert StepPhase.PREFILL.value == "prefill"
    assert StepPhase.DECODE.value == "decode"
    assert StepPhase.MIXED.value == "mixed"
    assert StepPhase.CHUNKED_PREFILL.value == "chunked_prefill"


def test_global_step_workload_aggregates():
    requests = [
        RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=7, context_len=7, target_output_len=128,
        ),
        RequestWorkload(
            request_id="r1", phase=StepPhase.DECODE,
            num_tokens=1, context_len=42, target_output_len=128, generated_tokens=10,
        ),
    ]
    w = GlobalStepWorkload(
        step_id=3,
        phase=StepPhase.MIXED,
        requests=requests,
        num_prefill_tokens=7,
        num_decode_tokens=1,
        total_scheduled_tokens=8,
        num_prefill_requests=1,
        num_decode_requests=1,
    )
    assert w.batch_size == 2
    assert w.max_context_len == 42  # max over requests


def test_global_step_workload_empty():
    w = GlobalStepWorkload(step_id=0, phase=StepPhase.DECODE)
    assert w.batch_size == 0
    assert w.max_context_len == 0
