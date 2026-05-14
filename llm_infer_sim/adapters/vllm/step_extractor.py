"""VllmStepExtractor — vLLM SchedulerOutput → GlobalStepWorkload。

vLLM 0.20.1 的 SchedulerOutput 关键字段:
  scheduled_new_reqs:    list[NewRequestData]
  scheduled_cached_reqs: CachedRequestData            # 单对象 + parallel arrays
  num_scheduled_tokens:  dict[req_id, int]
  total_num_scheduled_tokens: int
  finished_req_ids:      set[str]
  preempted_req_ids:     set[str] | None

CachedRequestData 用 parallel arrays 存批量 cached 请求:
  req_ids: list[str]
  num_computed_tokens: list[int]                       # 已经 forward 过的 token 数
  num_output_tokens:   list[int]                       # 已生成的 output token 数

设计文档 §4.3.2 的写法是 `for req in scheduled_cached_reqs`, 在 0.20.1 上不成立,
本实现按 parallel-array 遍历来修正。
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload,
    RequestWorkload,
    StepPhase,
)


class VllmStepExtractor:
    """vLLM SchedulerOutput → 框架无关 GlobalStepWorkload。"""

    @staticmethod
    def extract(
        scheduler_output: Any,
        step_id: int,
        request_states: dict[str, dict] | None = None,
    ) -> GlobalStepWorkload:
        request_states = request_states or {}
        requests: list[RequestWorkload] = []

        num_prefill_tokens = 0
        num_decode_tokens = 0
        num_prefix_cached_tokens = 0

        # ---- 1. 新到 prefill 请求 ----
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            ntok = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            prompt_len = len(new_req.prompt_token_ids or [])
            already_computed = new_req.num_computed_tokens
            target_max = _max_tokens_or_default(new_req.sampling_params)
            # 新请求首次出现时 num_computed_tokens > 0 = prefix cache 命中
            if already_computed > 0:
                num_prefix_cached_tokens += already_computed
            # chunked prefill: 本 step 调度 token 数 < 剩余 prompt
            remaining_prompt = max(prompt_len - already_computed, 0)
            is_chunked = ntok < remaining_prompt

            requests.append(
                RequestWorkload(
                    request_id=req_id,
                    phase=StepPhase.CHUNKED_PREFILL if is_chunked else StepPhase.PREFILL,
                    num_tokens=ntok,
                    context_len=already_computed + ntok,
                    target_output_len=target_max,
                    generated_tokens=0,
                    is_chunked=is_chunked,
                    chunk_size=ntok if is_chunked else 0,
                )
            )
            num_prefill_tokens += ntok

        # ---- 2. 已缓存请求 (parallel arrays) ----
        cached = scheduler_output.scheduled_cached_reqs
        cached_req_ids: list[str] = list(cached.req_ids)
        cached_num_computed: list[int] = list(cached.num_computed_tokens)
        cached_num_output: list[int] = list(cached.num_output_tokens)

        for i, req_id in enumerate(cached_req_ids):
            ntok = scheduler_output.num_scheduled_tokens.get(req_id, 1)
            num_computed = cached_num_computed[i]
            num_output = cached_num_output[i]

            # decode: 单步只取 1 token, 之前都已 forward 过
            # chunked prefill 第二段+: ntok > 1 表示还在啃 prompt
            is_chunked_continuation = ntok > 1
            phase = (
                StepPhase.CHUNKED_PREFILL
                if is_chunked_continuation
                else StepPhase.DECODE
            )

            target_max = request_states.get(req_id, {}).get("target_output_len", 0)

            requests.append(
                RequestWorkload(
                    request_id=req_id,
                    phase=phase,
                    num_tokens=ntok,
                    context_len=num_computed + ntok,
                    target_output_len=target_max,
                    generated_tokens=num_output,
                    is_chunked=is_chunked_continuation,
                    chunk_size=ntok if is_chunked_continuation else 0,
                )
            )
            if is_chunked_continuation:
                num_prefill_tokens += ntok
            else:
                num_decode_tokens += ntok

        # ---- 3. 整体阶段判定 ----
        if num_prefill_tokens > 0 and num_decode_tokens > 0:
            phase = StepPhase.MIXED
        elif num_decode_tokens == 0 and num_prefill_tokens > 0:
            phase = StepPhase.PREFILL
        elif num_prefill_tokens == 0 and num_decode_tokens > 0:
            phase = StepPhase.DECODE
        else:
            # 空 step: 仍标记为 decode (no-op), 与 batch_size=0 配合
            phase = StepPhase.DECODE

        return GlobalStepWorkload(
            step_id=step_id,
            phase=phase,
            requests=requests,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            total_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_prefill_requests=sum(
                1 for r in requests
                if r.phase in (StepPhase.PREFILL, StepPhase.CHUNKED_PREFILL)
            ),
            num_decode_requests=sum(
                1 for r in requests if r.phase == StepPhase.DECODE
            ),
            num_prefix_cached_tokens=num_prefix_cached_tokens,
        )


def _max_tokens_or_default(sampling_params: Any, default: int = 128) -> int:
    if sampling_params is None:
        return default
    val = getattr(sampling_params, "max_tokens", None)
    return int(val) if val is not None else default
