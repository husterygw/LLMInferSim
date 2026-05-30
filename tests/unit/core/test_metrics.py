"""MetricsCollector + ReportGenerator 行为回归 (V3 StepCostTrace)."""
from llm_infer_sim.core.cost.trace import StepCostTrace
from llm_infer_sim.core.metrics.collector import MetricsCollector, _percentile
from llm_infer_sim.core.metrics.reporter import ReportGenerator
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


def _wl(step_id, phase, requests, total_tokens, n_pref_req, n_dec_req,
        n_pref_tok=0, n_dec_tok=0):
    return GlobalStepWorkload(
        step_id=step_id, phase=phase, requests=requests,
        num_prefill_tokens=n_pref_tok, num_decode_tokens=n_dec_tok,
        total_scheduled_tokens=total_tokens,
        num_prefill_requests=n_pref_req, num_decode_requests=n_dec_req,
    )


def _cost(step_id, phase, latency, compute=0.0, memory=0.0, comm=0.0) -> StepCostTrace:
    """手造一个 StepCostTrace 用于 metrics 回归 (entries 空)."""
    if compute >= memory:
        bottleneck = "compute" if compute > 0 else "memory"
    else:
        bottleneck = "memory"
    return StepCostTrace(
        step_id=step_id,
        phase=phase.value if hasattr(phase, "value") else phase,
        total_latency_s=latency,
        compute_time_s=compute, memory_time_s=memory,
        comm_time_s=comm, runtime_time_s=0.0,
        entries=(),
        bottleneck=bottleneck,
    )


def test_percentile_helper():
    assert _percentile([], 50) == 0.0
    assert _percentile([1.0], 99) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    # 50% of [1,2,3,4] interpolates to 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_single_request_full_lifecycle():
    """1 个 request: prefill 1 step (10ms) → decode 3 steps (1ms each) → finish."""
    c = MetricsCollector()
    # Step 1: prefill, 256 tok, no first token yet (generated_tokens=0)
    c.record_step(
        _wl(1, StepPhase.PREFILL, [
            RequestWorkload(request_id="r1", phase=StepPhase.PREFILL,
                            num_tokens=256, context_len=256, generated_tokens=0,
                            target_output_len=4),
        ], total_tokens=256, n_pref_req=1, n_dec_req=0, n_pref_tok=256),
        _cost(1, StepPhase.PREFILL, latency=0.010),
    )
    # Step 2: decode (generated_tokens=1) → first_token recorded at sim_time=0.011
    c.record_step(
        _wl(2, StepPhase.DECODE, [
            RequestWorkload(request_id="r1", phase=StepPhase.DECODE,
                            num_tokens=1, context_len=257, generated_tokens=1,
                            target_output_len=4),
        ], total_tokens=1, n_pref_req=0, n_dec_req=1, n_dec_tok=1),
        _cost(2, StepPhase.DECODE, latency=0.001),
    )
    # Step 3: decode #2
    c.record_step(
        _wl(3, StepPhase.DECODE, [
            RequestWorkload(request_id="r1", phase=StepPhase.DECODE,
                            num_tokens=1, context_len=258, generated_tokens=2,
                            target_output_len=4),
        ], total_tokens=1, n_pref_req=0, n_dec_req=1, n_dec_tok=1),
        _cost(3, StepPhase.DECODE, latency=0.001),
    )
    # Step 4: decode #3 + finish
    c.record_step(
        _wl(4, StepPhase.DECODE, [
            RequestWorkload(request_id="r1", phase=StepPhase.DECODE,
                            num_tokens=1, context_len=259, generated_tokens=3,
                            target_output_len=4),
        ], total_tokens=1, n_pref_req=0, n_dec_req=1, n_dec_tok=1),
        _cost(4, StepPhase.DECODE, latency=0.001),
        finished_req_ids={"r1"},
    )

    r = c.requests["r1"]
    assert r.completed
    # arrival = 0, first_token = 0.010 (end of step 1) + 0.001 = 0.011
    assert r.arrival_time == 0.0
    assert abs(r.first_token_time - 0.011) < 1e-9
    assert abs(r.ttft - 0.011) < 1e-9
    # tpot = mean of latencies after first: [step2=0.001, step3=0.001, step4=0.001][1:]
    # = mean([0.001, 0.001]) = 0.001
    assert abs(r.tpot - 0.001) < 1e-9
    # e2e = completion - arrival = (0.010 + 0.001*3) - 0 = 0.013
    assert abs(r.e2e_latency - 0.013) < 1e-9

    summary = c.get_summary()
    assert summary.total_requests == 1
    assert summary.completed_requests == 1
    assert summary.total_steps == 4
    assert abs(summary.elapsed_sim_time - 0.013) < 1e-9
    assert summary.requests_per_second > 0


def test_finished_req_ids_marks_completion():
    c = MetricsCollector()
    c.record_step(
        _wl(1, StepPhase.PREFILL, [
            RequestWorkload(request_id="r1", phase=StepPhase.PREFILL,
                            num_tokens=8, context_len=8, generated_tokens=0),
        ], total_tokens=8, n_pref_req=1, n_dec_req=0, n_pref_tok=8),
        _cost(1, StepPhase.PREFILL, latency=0.005),
        finished_req_ids={"r1"},
    )
    assert c.requests["r1"].completed
    assert abs(c.requests["r1"].completion_time - 0.005) < 1e-9


def test_multi_request_throughput():
    """2 requests prefill 一起 → decode 一起 → finish 一起。"""
    c = MetricsCollector()
    reqs_pref = [
        RequestWorkload(request_id=f"r{i}", phase=StepPhase.PREFILL,
                        num_tokens=64, context_len=64, generated_tokens=0,
                        target_output_len=2)
        for i in range(2)
    ]
    c.record_step(_wl(1, StepPhase.PREFILL, reqs_pref, total_tokens=128,
                       n_pref_req=2, n_dec_req=0, n_pref_tok=128),
                  _cost(1, StepPhase.PREFILL, latency=0.020))

    reqs_dec1 = [
        RequestWorkload(request_id=f"r{i}", phase=StepPhase.DECODE,
                        num_tokens=1, context_len=65, generated_tokens=1,
                        target_output_len=2)
        for i in range(2)
    ]
    c.record_step(_wl(2, StepPhase.DECODE, reqs_dec1, total_tokens=2,
                       n_pref_req=0, n_dec_req=2, n_dec_tok=2),
                  _cost(2, StepPhase.DECODE, latency=0.002),
                  finished_req_ids={"r0", "r1"})

    s = c.get_summary()
    assert s.total_requests == 2
    assert s.completed_requests == 2
    assert s.requests_per_second == 2 / 0.022


def test_reporter_console_does_not_crash():
    c = MetricsCollector()
    # 空数据也要能出报告
    rep = ReportGenerator(c)
    text = rep.generate_console_report()
    assert "Performance Report" in text
    assert "Total Requests:      0" in text


def test_reporter_save(tmp_path):
    c = MetricsCollector()
    c.record_step(
        _wl(1, StepPhase.PREFILL, [
            RequestWorkload(request_id="r1", phase=StepPhase.PREFILL,
                            num_tokens=8, context_len=8, generated_tokens=0),
        ], total_tokens=8, n_pref_req=1, n_dec_req=0, n_pref_tok=8),
        _cost(1, StepPhase.PREFILL, latency=0.005),
        finished_req_ids={"r1"},
    )
    rep = ReportGenerator(c)
    out = tmp_path / "report.json"
    rep.save_report(out)
    assert out.exists()
    assert out.with_suffix(".txt").exists()
