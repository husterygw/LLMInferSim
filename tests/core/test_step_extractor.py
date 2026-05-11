"""阶段 3 E 块: step_extractor + request_states 跨步状态回归测试。

确认:
  1. avg_decode_context_len / max_prefill_seqlen 计算正确
  2. extract() 接受 request_states 字典且能补回 target_output_len
  3. extract() 对 mixed batch 的 phase 判定 = MIXED
  4. extract() 对 chunked prefill (新到 + cached 续段) 都识别 chunked_prefill
"""
from types import SimpleNamespace

from llm_infer_sim.adapters.vllm.step_extractor import VllmStepExtractor
from llm_infer_sim.core.workload.workload import StepPhase


def _so(new_reqs, cached, num_scheduled, total_tokens, finished=None):
    """构造一个 vLLM SchedulerOutput-shape 的 SimpleNamespace。"""
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=total_tokens,
        finished_req_ids=finished or set(),
        preempted_req_ids=None,
    )


def _new_req(req_id, prompt_len, computed=0, max_tokens=128):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=[0] * prompt_len,
        num_computed_tokens=computed,
        sampling_params=SimpleNamespace(max_tokens=max_tokens),
    )


def _cached(req_ids, num_computed, num_output):
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens=list(num_computed),
        num_output_tokens=list(num_output),
    )


def test_mixed_batch_phase():
    """1 prefill + 2 decode → MIXED."""
    so = _so(
        new_reqs=[_new_req("p1", prompt_len=64)],
        cached=_cached(["d1", "d2"], num_computed=[100, 200], num_output=[5, 10]),
        num_scheduled={"p1": 64, "d1": 1, "d2": 1},
        total_tokens=66,
    )
    wl = VllmStepExtractor.extract(so, step_id=1)
    assert wl.phase == StepPhase.MIXED
    assert wl.num_prefill_tokens == 64
    assert wl.num_decode_tokens == 2
    assert wl.num_prefill_requests == 1
    assert wl.num_decode_requests == 2


def test_chunked_prefill_new_request():
    """新请求 prompt=1000 但本 step 只调度 256 tok → chunked_prefill。"""
    so = _so(
        new_reqs=[_new_req("p1", prompt_len=1000, computed=0)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 256},
        total_tokens=256,
    )
    wl = VllmStepExtractor.extract(so, step_id=1)
    r = wl.requests[0]
    assert r.phase == StepPhase.CHUNKED_PREFILL
    assert r.is_chunked is True
    assert r.chunk_size == 256


def test_chunked_prefill_cached_continuation():
    """cached 请求本 step 调度 ntok>1, 表示还在啃 prompt → chunked_prefill。"""
    so = _so(
        new_reqs=[],
        cached=_cached(["p1"], num_computed=[256], num_output=[0]),
        num_scheduled={"p1": 256},  # 还在 prefill 阶段, 不是 decode
        total_tokens=256,
    )
    wl = VllmStepExtractor.extract(so, step_id=2)
    r = wl.requests[0]
    assert r.phase == StepPhase.CHUNKED_PREFILL
    assert r.is_chunked is True


def test_request_states_target_output_len_propagates():
    """cached step 没 sampling_params, 必须从 request_states 拿 target_output_len。"""
    request_states = {"d1": {"target_output_len": 200}}
    so = _so(
        new_reqs=[],
        cached=_cached(["d1"], num_computed=[42], num_output=[10]),
        num_scheduled={"d1": 1},
        total_tokens=1,
    )
    wl = VllmStepExtractor.extract(so, step_id=3, request_states=request_states)
    assert wl.requests[0].target_output_len == 200


def test_avg_decode_context_len_and_max_prefill_seqlen():
    """阶段 3 新增 properties 用于 MixedAttentionEstimator。"""
    so = _so(
        new_reqs=[_new_req("p1", prompt_len=512), _new_req("p2", prompt_len=256)],
        cached=_cached(["d1", "d2", "d3"],
                       num_computed=[1000, 2000, 3000],
                       num_output=[5, 5, 5]),
        num_scheduled={"p1": 512, "p2": 256, "d1": 1, "d2": 1, "d3": 1},
        total_tokens=771,
    )
    wl = VllmStepExtractor.extract(so, step_id=1)
    # decode ctx_lens: 1001, 2001, 3001 → avg = 2001
    assert wl.avg_decode_context_len == 2001
    # max prefill seqlen = max(512, 256) = 512
    assert wl.max_prefill_seqlen == 512


def test_empty_step_marked_decode():
    """空 step (no requests) 应被标 DECODE 防御性占位。"""
    so = _so(
        new_reqs=[],
        cached=_cached([], [], []),
        num_scheduled={},
        total_tokens=0,
    )
    wl = VllmStepExtractor.extract(so, step_id=99)
    assert wl.phase == StepPhase.DECODE
    assert wl.batch_size == 0
