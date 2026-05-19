"""Shot → vLLM SchedulerOutput 构造 (详设 §9.4.2 B.2).

策略: 我们直接构造合成 SchedulerOutput, 绕开 vLLM scheduler — 形状跟 shot 1:1
对应, 不会被 scheduler chunk / reorder。关键 trick:

    num_computed_tokens = history
    prompt_token_ids = [1] * (history + new_tokens)

这告诉 vLLM "前 history 个 token 已经 forward 过、KV 在 cache 里", combined with
prompt_token_ids 长度对得上 → engine 当成真请求处理。

vLLM 0.20.1 字段差异 vs LLMServingSim 0.19.0:
  - `CachedRequestData` 加了 `resumed_req_ids` / `all_token_ids` (init 用 make_empty()
    就行, 不动)
  - `NewRequestData.block_ids` 从 `list[list[int]]` 改成 `tuple[list[int], ...]`
  - SchedulerOutput 加了一堆 optional 字段 (kv_connector_metadata / structured_output /
    new_block_ids_to_zero), 全部用默认 None/False/[] 就行
  - SchedulerOutput.make_empty() 是官方推荐的最小构造法

设计参考 LLMServingSim profiler/core/hooks/batch.py (作者 @waneon, vLLM 0.19.0).
"""
from __future__ import annotations

import math
from typing import Any

from llm_infer_sim.calibration.shots import Shot


def assemble_scheduler_output(shot: Shot, model_runner) -> tuple[Any, set[str]]:
    """Build SchedulerOutput describing the shot's synthetic batch.

    Args:
        shot: Hydrated Shot (kind / num_new_tokens / kv_lens_* / prefill_chunk).
        model_runner: vLLM v1 GPUModelRunner instance — 我们从它读 block_tables
                      获取 KV cache group 数 + block_size.

    Returns:
        (scheduler_output, req_ids_set).

    Raises:
        RuntimeError: model_runner.input_batch / block_table 不可访问 (vLLM 内部
                      API 改动); ImportError: vLLM 不可用.
    """
    # 局部 import 避免主模块加载时 pull vLLM (worker 进程才需要)
    try:
        from vllm import SamplingParams
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "calibration.batch.assemble_scheduler_output 需要 vLLM 已安装。"
        ) from e

    # 1-token greedy sampling (我们不关心 token 质量, 只关心 kernel shape)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
    )

    # 多 KV-cache group 支持 (e.g. hybrid 模型多 cache pool, MLA 单 group)
    try:
        block_tables = model_runner.input_batch.block_table.block_tables
    except AttributeError as e:
        raise RuntimeError(
            "vLLM v1 model_runner.input_batch.block_table.block_tables 访问失败; "
            "vLLM 0.20.1 API 可能改了. 详 §9.4.2 反向工程指南."
        ) from e

    block_sizes = [
        bt.block_size * bt.blocks_per_kv_block
        for bt in block_tables
    ]
    num_kv_groups = len(block_sizes)

    # Shot → request list 展开
    requests = _shot_to_requests(shot)
    if not requests:
        raise ValueError(f"Shot {shot} 展开后 request 列表为空")

    scheduled: list = []
    num_scheduled_tokens: dict[str, int] = {}
    total_num_scheduled_tokens = 0
    block_cursor = [0] * num_kv_groups   # 每 KV group 独立游标
    req_ids: list[str] = []

    for idx, (new_tokens, history) in enumerate(requests):
        req_id = f"calib_r{idx}"
        total_len = new_tokens + history

        # 每 KV group 分配足够 block 覆盖 total_len. 部分 block 也算一整块.
        group_block_ids: list[list[int]] = []
        for g, bs in enumerate(block_sizes):
            num_blocks = math.ceil(total_len / bs) if total_len else 1
            ids = list(range(block_cursor[g], block_cursor[g] + num_blocks))
            block_cursor[g] += num_blocks
            group_block_ids.append(ids)

        scheduled.append(NewRequestData(
            req_id=req_id,
            # 内容不重要, 长度必须 = history + new_tokens
            prompt_token_ids=[1] * total_len,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(group_block_ids),  # 0.20.1: tuple[list[int], ...]
            num_computed_tokens=history,        # KV cache "已经有 history 个 tok"
            lora_request=None,
        ))
        num_scheduled_tokens[req_id] = new_tokens
        total_num_scheduled_tokens += new_tokens
        req_ids.append(req_id)

    # SchedulerOutput 构造 — 9 个 required 字段, optional 用默认
    so = SchedulerOutput(
        scheduled_new_reqs=scheduled,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0] * num_kv_groups,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    # invariant 自检 (帮 debug)
    assert so.total_num_scheduled_tokens == sum(so.num_scheduled_tokens.values()), (
        f"invariant 错: total {so.total_num_scheduled_tokens} != sum "
        f"{sum(so.num_scheduled_tokens.values())}"
    )
    return so, set(req_ids)


def _shot_to_requests(shot: Shot) -> list[tuple[int, int]]:
    """Shot 展开为 [(new_tokens, history), ...] 列表, 一项 per request.

    dense:        1 request, new=num_new_tokens, history=0
    per_sequence: num_decode_seqs 个 (1, kv_lens_decode[i]) requests
    attention:    最多 1 个 prefill request (new=prefill_chunk, history=kv_lens_prefill[0])
                  + num_decode_seqs 个 (1, kv_lens_decode[i]) decode requests
    """
    if shot.kind == "dense":
        return [(shot.num_new_tokens, 0)]

    if shot.kind == "per_sequence":
        # 每 seq 1 个 1-tok decode, ctx_len 取 kv_lens_decode[i] (或 0)
        kv_lens = list(shot.kv_lens_decode) + [0] * shot.num_decode_seqs
        return [(1, kv_lens[i]) for i in range(shot.num_decode_seqs)]

    if shot.kind == "attention":
        reqs: list[tuple[int, int]] = []
        if shot.prefill_chunk > 0:
            kv_pref = shot.kv_lens_prefill[0] if shot.kv_lens_prefill else 0
            reqs.append((shot.prefill_chunk, kv_pref))
        if shot.num_decode_seqs > 0:
            # decode 每 seq 1 tok, 各自 ctx_len
            kv_lens = list(shot.kv_lens_decode) + [0] * shot.num_decode_seqs
            for i in range(shot.num_decode_seqs):
                reqs.append((1, kv_lens[i]))
        return reqs

    raise ValueError(f"Unknown shot kind: {shot.kind}")
