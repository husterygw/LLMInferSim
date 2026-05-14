"""KVBlockAllocator — block 字节公式 + per-step alloc/free + dedup + cumulative."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.simulation.kv_block_allocator import (
    KVBlockAllocator,
    compute_block_bytes,
)


# ---------- helpers (复用 step_extractor 测试 shape) ----------

def _so(new_reqs, cached, num_scheduled, total_tokens, finished=None):
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=total_tokens,
        finished_req_ids=finished or set(),
        preempted_req_ids=None,
    )


def _new_req(req_id, prompt_len, computed=0):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=[0] * prompt_len,
        num_computed_tokens=computed,
        sampling_params=SimpleNamespace(max_tokens=128),
    )


def _cached(req_ids, num_computed, num_output):
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens=list(num_computed),
        num_output_tokens=list(num_output),
    )


# ---------- block_bytes 公式 ----------

def test_block_bytes_standard_mha():
    """标准 MHA: block_size × layers × kv_heads × head_dim × 2(K+V) × kv_byte。"""
    model = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, kv_lora_rank=0)
    # 16 × 32 × 8 × 128 × 2 × 2.0 = 2,097,152 bytes
    assert compute_block_bytes(model, block_size=16, kv_byte=2.0) == 2097152


def test_block_bytes_mla_much_smaller():
    """MLA: 单 c_kv tensor, 无 K+V ×2。V3 配置 ~57× 小于 MHA。"""
    model = ModelConfig(
        num_layers=61, num_kv_heads=128, head_dim=128,
        kv_lora_rank=512, rope_head_dim=64,
    )
    # MLA: 16 × 61 × (512+64) × 2.0 = 1,124,352 bytes
    mla_bytes = compute_block_bytes(model, block_size=16, kv_byte=2.0)
    assert mla_bytes == 1124352

    # 对比 MHA 同模型 (kv_lora_rank=0): 16 × 61 × 128 × 128 × 2 × 2.0 = 63,963,136
    model_mha = ModelConfig(
        num_layers=61, num_kv_heads=128, head_dim=128, kv_lora_rank=0,
    )
    mha_bytes = compute_block_bytes(model_mha, block_size=16, kv_byte=2.0)
    assert mha_bytes / mla_bytes > 50, f"MLA 应 ~57× 小, 实测 {mha_bytes / mla_bytes:.1f}×"


def test_block_bytes_fp8_kv_halves():
    """KV fp8 比 fp16 字节数减半。"""
    model = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, kv_lora_rank=0)
    bytes_fp16 = compute_block_bytes(model, block_size=16, kv_byte=2.0)
    bytes_fp8 = compute_block_bytes(model, block_size=16, kv_byte=1.0)
    assert bytes_fp8 == bytes_fp16 // 2


# ---------- alloc / free 增量 ----------

def _mk_alloc(num_blocks_total=1000):
    model = ModelConfig(num_layers=4, num_kv_heads=2, head_dim=64, kv_lora_rank=0)
    return KVBlockAllocator(
        model, block_size=16, num_blocks_total=num_blocks_total, kv_byte=2.0,
    )


def test_alloc_new_prefill_request():
    """单 new_req prompt=1000 tok, block=16 → ceil(1000/16)=63 block。"""
    alloc = _mk_alloc()
    so = _so(
        new_reqs=[_new_req("p1", prompt_len=1000)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 1000}, total_tokens=1000,
    )
    stats = alloc.step(so, num_prefix_cached_tokens=0)
    assert stats.new_blocks_allocated == 63
    assert stats.blocks_dedup_hit == 0
    assert stats.blocks_in_use_after == 63
    assert alloc.get_req_blocks("p1") == 63


def test_alloc_prefix_cache_dedup():
    """prompt=1000, cache 命中 800 tok = 50 block. 净新分配 13 block。"""
    alloc = _mk_alloc()
    so = _so(
        new_reqs=[_new_req("p1", prompt_len=1000, computed=800)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 200}, total_tokens=200,
    )
    stats = alloc.step(so, num_prefix_cached_tokens=800)
    assert stats.new_blocks_allocated == 63 - 50    # 13
    assert stats.blocks_dedup_hit == 50
    assert stats.cached_tokens_this_step == 800


def test_alloc_decode_crosses_block_boundary():
    """cached_req 起始持 5 block (80 tok), decode 第 81 tok → +1 block。"""
    alloc = _mk_alloc()
    # 先 prefill 80 tok
    so1 = _so(
        new_reqs=[_new_req("p1", prompt_len=80)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 80}, total_tokens=80,
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    assert alloc.get_req_blocks("p1") == 5

    # decode 1 步: 80 → 81 跨边界
    so2 = _so(
        new_reqs=[],
        cached=_cached(["p1"], num_computed=[80], num_output=[0]),
        num_scheduled={"p1": 1}, total_tokens=1,
    )
    stats = alloc.step(so2, num_prefix_cached_tokens=0)
    assert stats.new_blocks_allocated == 1  # 跨块: +1
    assert alloc.get_req_blocks("p1") == 6


def test_alloc_decode_within_block_no_growth():
    """decode 在 block 内: 不分配新 block。"""
    alloc = _mk_alloc()
    so1 = _so(
        new_reqs=[_new_req("p1", prompt_len=70)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 70}, total_tokens=70,
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    # 70 tok = ceil(70/16)=5 block, 5×16=80 容量, 还能装 10 tok 不跨块
    assert alloc.get_req_blocks("p1") == 5
    so2 = _so(
        new_reqs=[],
        cached=_cached(["p1"], num_computed=[70], num_output=[0]),
        num_scheduled={"p1": 1}, total_tokens=1,
    )
    stats = alloc.step(so2, num_prefix_cached_tokens=0)
    assert stats.new_blocks_allocated == 0
    assert alloc.get_req_blocks("p1") == 5


def test_free_on_finished():
    alloc = _mk_alloc()
    so1 = _so(
        new_reqs=[_new_req("p1", prompt_len=160)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 160}, total_tokens=160,
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    assert alloc.get_req_blocks("p1") == 10

    # 下一步 finished
    so2 = _so(
        new_reqs=[],
        cached=_cached([], [], []),
        num_scheduled={}, total_tokens=0,
        finished={"p1"},
    )
    stats = alloc.step(so2, num_prefix_cached_tokens=0)
    assert stats.blocks_freed == 10
    assert stats.blocks_in_use_after == 0
    assert alloc.get_req_blocks("p1") == 0


# ---------- cumulative ----------

def test_cumulative_hit_rate():
    alloc = _mk_alloc()
    # batch 1: cold 1000 tok
    so1 = _so(
        new_reqs=[_new_req("p1", prompt_len=1000)],
        cached=_cached([], [], []),
        num_scheduled={"p1": 1000}, total_tokens=1000,
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    # batch 2: warm 1100 tok, 1000 命中
    so2 = _so(
        new_reqs=[_new_req("p2", prompt_len=1100, computed=1000)],
        cached=_cached([], [], []),
        num_scheduled={"p2": 100}, total_tokens=100,
    )
    alloc.step(so2, num_prefix_cached_tokens=1000)

    c = alloc.cumulative
    assert c.cumulative_cached_tokens == 1000
    assert c.cumulative_total_prompt_tokens == 2100
    assert c.prefix_cache_hit_rate == pytest.approx(1000 / 2100, rel=1e-3)


def test_cumulative_peak_in_use():
    alloc = _mk_alloc(num_blocks_total=100)
    so1 = _so(
        new_reqs=[_new_req("a", prompt_len=160), _new_req("b", prompt_len=320)],
        cached=_cached([], [], []),
        num_scheduled={"a": 160, "b": 320}, total_tokens=480,
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    # 10 + 20 = 30 block in use
    assert alloc.cumulative.peak_blocks_in_use == 30

    so2 = _so(
        new_reqs=[],
        cached=_cached([], [], []),
        num_scheduled={}, total_tokens=0,
        finished={"a"},
    )
    alloc.step(so2, num_prefix_cached_tokens=0)
    # peak 仍 30 (历史最高)
    assert alloc.cumulative.peak_blocks_in_use == 30


def test_over_capacity_counted():
    alloc = _mk_alloc(num_blocks_total=5)
    so = _so(
        new_reqs=[_new_req("a", prompt_len=200)],  # 13 block, 远超 5
        cached=_cached([], [], []),
        num_scheduled={"a": 200}, total_tokens=200,
    )
    alloc.step(so, num_prefix_cached_tokens=0)
    assert alloc.cumulative.num_steps_over_capacity == 1


def test_block_dedup_hit_rate():
    alloc = _mk_alloc()
    so1 = _so(
        new_reqs=[_new_req("p1", prompt_len=320)],  # 20 block cold
        cached=_cached([], [], []),
        num_scheduled={"p1": 320}, total_tokens=320,
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    so2 = _so(
        new_reqs=[_new_req("p2", prompt_len=320, computed=160)],  # 10 命中 + 10 新
        cached=_cached([], [], []),
        num_scheduled={"p2": 160}, total_tokens=160,
    )
    alloc.step(so2, num_prefix_cached_tokens=160)
    # 总计: 20 cold alloc + 10 alloc + 10 dedup = 40 ops, 10 dedup
    assert alloc.cumulative.block_dedup_hit_rate == pytest.approx(10 / 40, rel=1e-3)


# ---------- PD 接口预演 (req_kv_bytes) ----------

def test_req_kv_bytes_for_pd_transfer():
    """PD transfer cost 的字节数依据。"""
    model = ModelConfig(num_layers=4, num_kv_heads=2, head_dim=64, kv_lora_rank=0)
    alloc = KVBlockAllocator(model, block_size=16, num_blocks_total=1000, kv_byte=2.0)
    so = _so(
        new_reqs=[_new_req("a", prompt_len=1024)],
        cached=_cached([], [], []),
        num_scheduled={"a": 1024}, total_tokens=1024,
    )
    alloc.step(so, num_prefix_cached_tokens=0)
    # 1024/16 = 64 block; per block = 16×4×2×64×2×2 = 32768
    assert alloc.get_req_blocks("a") == 64
    expected_bytes = 64 * 16 * 4 * 2 * 64 * 2 * 2
    assert alloc.req_kv_bytes("a") == expected_bytes
