"""calibration/batch.py — Shot → SchedulerOutput 构造 (B.2).

策略: mock model_runner.input_batch.block_table.block_tables 提供 fake block_size,
然后调 assemble_scheduler_output. 真 vLLM SchedulerOutput / NewRequestData 也得
能 import (vLLM 0.20.1), 这一步是字段对齐的真测试. 没 GPU 也能跑.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.calibration.batch import _shot_to_requests, assemble_scheduler_output
from llm_infer_sim.calibration.shots import Shot


# ---- _shot_to_requests 展开逻辑 ----

def test_shot_to_requests_dense():
    s = Shot(kind="dense", num_new_tokens=128)
    assert _shot_to_requests(s) == [(128, 0)]


def test_shot_to_requests_per_sequence():
    s = Shot(kind="per_sequence", num_new_tokens=4, num_decode_seqs=4,
             kv_lens_decode=[100, 200, 300, 400])
    assert _shot_to_requests(s) == [(1, 100), (1, 200), (1, 300), (1, 400)]


def test_shot_to_requests_attention_decode_only():
    s = Shot(kind="attention", num_new_tokens=4, num_decode_seqs=4,
             kv_lens_decode=[1024] * 4)
    assert _shot_to_requests(s) == [(1, 1024)] * 4


def test_shot_to_requests_attention_prefill_only():
    s = Shot(kind="attention", num_new_tokens=512, prefill_chunk=512,
             kv_lens_prefill=[0])
    assert _shot_to_requests(s) == [(512, 0)]


def test_shot_to_requests_attention_chunked_continuation():
    """chunked prefill 续段: prefill_chunk=256, 已 forward 1024 → (256, 1024)."""
    s = Shot(kind="attention", num_new_tokens=256, prefill_chunk=256,
             kv_lens_prefill=[1024])
    assert _shot_to_requests(s) == [(256, 1024)]


def test_shot_to_requests_attention_mixed():
    """chunked prefill + decode 同 step."""
    s = Shot(kind="attention",
             num_new_tokens=128 + 4, prefill_chunk=128, num_decode_seqs=4,
             kv_lens_prefill=[0],
             kv_lens_decode=[2048, 2048, 2048, 2048])
    expected = [(128, 0), (1, 2048), (1, 2048), (1, 2048), (1, 2048)]
    assert _shot_to_requests(s) == expected


def test_shot_to_requests_unknown_kind():
    s = Shot(kind="moe", num_new_tokens=128)
    with pytest.raises(ValueError, match="Unknown shot kind"):
        _shot_to_requests(s)


# ---- assemble_scheduler_output (需要 vLLM import 跑通) ----

class _MockBlockTable:
    """模拟 vllm.v1.worker.block_table.BlockTable."""
    def __init__(self, block_size: int, blocks_per_kv_block: int = 1):
        self.block_size = block_size
        self.blocks_per_kv_block = blocks_per_kv_block


class _MockMultiGroupBlockTable:
    def __init__(self, num_groups: int = 1, block_size: int = 16):
        self.block_tables = [_MockBlockTable(block_size) for _ in range(num_groups)]


class _MockInputBatch:
    def __init__(self, num_kv_groups: int = 1, block_size: int = 16):
        self.block_table = _MockMultiGroupBlockTable(num_kv_groups, block_size)


class _MockModelRunner:
    def __init__(self, num_kv_groups: int = 1, block_size: int = 16):
        self.input_batch = _MockInputBatch(num_kv_groups, block_size)


def _has_vllm_v1_scheduler_output() -> bool:
    """vLLM v1 SchedulerOutput 必须能 import (B.2 字段对齐前提)."""
    try:
        from vllm.v1.core.sched.output import (  # noqa: F401
            CachedRequestData, NewRequestData, SchedulerOutput,
        )
        return True
    except ImportError:
        return False


needs_vllm = pytest.mark.skipif(
    not _has_vllm_v1_scheduler_output(),
    reason="vLLM v1 SchedulerOutput 不可 import",
)


@needs_vllm
def test_assemble_dense_shot_invariants():
    """dense shot=128 tok → 1 req, num_scheduled_tokens=128, block_ids 单 group."""
    s = Shot(kind="dense", num_new_tokens=128)
    runner = _MockModelRunner(num_kv_groups=1, block_size=16)
    so, req_ids = assemble_scheduler_output(s, runner)

    assert len(so.scheduled_new_reqs) == 1
    new_req = so.scheduled_new_reqs[0]
    assert new_req.req_id == "calib_r0"
    assert len(new_req.prompt_token_ids) == 128  # history=0, new=128
    assert new_req.num_computed_tokens == 0
    assert isinstance(new_req.block_ids, tuple) and len(new_req.block_ids) == 1
    # ceil(128 / 16) = 8 blocks
    assert len(new_req.block_ids[0]) == 8
    assert so.num_scheduled_tokens == {"calib_r0": 128}
    assert so.total_num_scheduled_tokens == 128
    assert req_ids == {"calib_r0"}


@needs_vllm
def test_assemble_attention_chunked_prefill_history_marker():
    """attention prefill_chunk=256, history=1024 → num_computed_tokens=1024, prompt_len=1280."""
    s = Shot(kind="attention", num_new_tokens=256, prefill_chunk=256,
             kv_lens_prefill=[1024])
    runner = _MockModelRunner(num_kv_groups=1, block_size=16)
    so, _ = assemble_scheduler_output(s, runner)

    new_req = so.scheduled_new_reqs[0]
    assert new_req.num_computed_tokens == 1024
    assert len(new_req.prompt_token_ids) == 256 + 1024
    # ceil(1280 / 16) = 80 blocks
    assert len(new_req.block_ids[0]) == 80
    assert so.num_scheduled_tokens["calib_r0"] == 256


@needs_vllm
def test_assemble_per_sequence_4_decodes():
    """per_sequence 4 个 1-tok decode, 各 256 ctx → 4 reqs, 总 4 tok scheduled."""
    s = Shot(kind="per_sequence", num_new_tokens=4, num_decode_seqs=4,
             kv_lens_decode=[256, 256, 256, 256])
    runner = _MockModelRunner(num_kv_groups=1, block_size=16)
    so, req_ids = assemble_scheduler_output(s, runner)

    assert len(so.scheduled_new_reqs) == 4
    for i, new_req in enumerate(so.scheduled_new_reqs):
        assert new_req.req_id == f"calib_r{i}"
        assert new_req.num_computed_tokens == 256
        # history=256, new=1 → prompt_len=257; ceil(257/16) = 17 blocks
        assert len(new_req.prompt_token_ids) == 257
        assert len(new_req.block_ids[0]) == 17
    assert so.total_num_scheduled_tokens == 4
    assert len(req_ids) == 4


@needs_vllm
def test_assemble_multi_kv_group():
    """MLA 模型多 KV group 场景: block_ids tuple 长度应 = num_groups."""
    s = Shot(kind="dense", num_new_tokens=64)
    runner = _MockModelRunner(num_kv_groups=2, block_size=16)
    so, _ = assemble_scheduler_output(s, runner)
    new_req = so.scheduled_new_reqs[0]
    assert len(new_req.block_ids) == 2
    # 两 group 都该有 ceil(64/16) = 4 blocks
    assert all(len(g) == 4 for g in new_req.block_ids)


@needs_vllm
def test_assemble_invariant_total_eq_sum():
    """SchedulerOutput.total_num_scheduled_tokens 必须 = sum(num_scheduled_tokens)."""
    s = Shot(kind="attention",
             num_new_tokens=128 + 4, prefill_chunk=128, num_decode_seqs=4,
             kv_lens_prefill=[0],
             kv_lens_decode=[2048] * 4)
    runner = _MockModelRunner(block_size=16)
    so, _ = assemble_scheduler_output(s, runner)
    assert so.total_num_scheduled_tokens == sum(so.num_scheduled_tokens.values())
    # 128 prefill + 4 × 1 decode = 132 tokens
    assert so.total_num_scheduled_tokens == 132


@needs_vllm
def test_assemble_block_cursor_increments_across_reqs():
    """多 req 时 block id 不应重复 (cursor 跨 req 单调递增)."""
    s = Shot(kind="per_sequence", num_new_tokens=3, num_decode_seqs=3,
             kv_lens_decode=[256, 256, 256])
    runner = _MockModelRunner(block_size=16)
    so, _ = assemble_scheduler_output(s, runner)
    all_block_ids = []
    for new_req in so.scheduled_new_reqs:
        all_block_ids.extend(new_req.block_ids[0])
    assert len(all_block_ids) == len(set(all_block_ids)), "block_id 重复了!"


@needs_vllm
def test_assemble_raises_on_missing_input_batch():
    """model_runner 缺 input_batch 时 raise RuntimeError 提示版本差异."""
    s = Shot(kind="dense", num_new_tokens=128)
    runner = SimpleNamespace()  # 没 input_batch
    with pytest.raises(RuntimeError, match="input_batch.block_table"):
        assemble_scheduler_output(s, runner)
