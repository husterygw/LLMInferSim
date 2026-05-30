"""PD 分离 KV transfer 集成测试 — virtual_model_runner._compute_pd_transfer_cost.

测试范围:
  1. role=None → 0 cost (gate 未启用)
  2. producer: new_req 一步完成 prefill → 计 send cost
  3. producer: chunked prefill 续段最后一步 → 计 send cost (cached_req 路径)
  4. producer: 同 req 不重复 send (idempotent on retry)
  5. consumer: new_req with num_computed_tokens==prompt_len → 计 recv cost
  6. 不同 connector 带宽: bw=12 (LMCache) 比 bw=25 (PyNccl) 慢 ~2×
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.core.deployment.pd_disagg import PDDisaggConfig
from tests.helpers.support import make_model_config
from llm_infer_sim.core.simulation.kv_block_allocator import KVBlockAllocator


def _so(new_reqs, cached, num_scheduled, finished=None):
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
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


class _MockRunner:
    """最小 stub 模拟 VirtualModelRunner._compute_pd_transfer_cost 所需上下文."""
    def __init__(self, pd_cfg: PDDisaggConfig, allocator: KVBlockAllocator):
        self._pd_cfg = pd_cfg
        self._block_allocator = allocator
        self._request_states: dict = {}
        self._pd_handled_reqs: set = set()
        self._pd_total_transfer_time = 0.0
        self._pd_total_transfer_bytes = 0
        self._pd_num_transfers = 0

    _compute_pd_transfer_cost = None  # 引用从真类拷过来


def _mk_runner(pd: PDDisaggConfig, num_blocks=1000):
    """模拟 runner: 复用 KVBlockAllocator + PD 实例."""
    from llm_infer_sim.adapters.vllm.virtual_model_runner import VirtualModelRunner
    model = make_model_config(num_layers=4, num_kv_heads=2, head_dim=64, kv_lora_rank=0)
    alloc = KVBlockAllocator(
        model, block_size=16, num_blocks_total=num_blocks, kv_byte=2.0,
    )
    # 直接 bind unbound method
    runner = _MockRunner(pd, alloc)
    runner._compute_pd_transfer_cost = (
        VirtualModelRunner._compute_pd_transfer_cost.__get__(runner)
    )
    return runner, alloc


# ---------- ----------

def test_disabled_pd_returns_zero():
    runner, alloc = _mk_runner(PDDisaggConfig())  # role=None
    so = _so(
        new_reqs=[_new_req("a", prompt_len=512)],
        cached=_cached([], [], []),
        num_scheduled={"a": 512},
    )
    alloc.step(so, num_prefix_cached_tokens=0)
    t, b = runner._compute_pd_transfer_cost(so)
    assert t == 0 and b == 0


def test_producer_emits_send_at_prefill_finish():
    pd = PDDisaggConfig(role="kv_producer", connector_name="P2pNcclConnector")
    runner, alloc = _mk_runner(pd)
    so = _so(
        new_reqs=[_new_req("a", prompt_len=512)],
        cached=_cached([], [], []),
        num_scheduled={"a": 512},
    )
    alloc.step(so, num_prefix_cached_tokens=0)
    t, b = runner._compute_pd_transfer_cost(so)
    # 32 block × block_bytes; bw=25 GB/s, lat=5us
    expected_bytes = alloc.req_kv_bytes("a")
    assert b == expected_bytes
    assert t > 0


def test_producer_idempotent_on_second_call():
    pd = PDDisaggConfig(role="kv_producer", connector_name="P2pNcclConnector")
    runner, alloc = _mk_runner(pd)
    so = _so(
        new_reqs=[_new_req("a", prompt_len=512)],
        cached=_cached([], [], []),
        num_scheduled={"a": 512},
    )
    alloc.step(so, num_prefix_cached_tokens=0)
    t1, b1 = runner._compute_pd_transfer_cost(so)
    # 重复同 so → 已 handled, 不重复 send
    t2, b2 = runner._compute_pd_transfer_cost(so)
    assert t1 > 0 and b1 > 0
    assert t2 == 0 and b2 == 0


def test_producer_chunked_prefill_send_at_last_chunk():
    """chunked prefill: 前面 chunk 不 send, 最后一段完成时 send 全部 KV."""
    pd = PDDisaggConfig(role="kv_producer", connector_name="P2pNcclConnector")
    runner, alloc = _mk_runner(pd)
    # state cache prompt_token_ids (runner 通常在 _update_request_states 做)
    runner._request_states["a"] = {"prompt_token_ids": [0] * 1000}

    # chunk 1: 256 / 1000
    so1 = _so(
        new_reqs=[_new_req("a", prompt_len=1000, computed=0)],
        cached=_cached([], [], []),
        num_scheduled={"a": 256},
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    t1, _ = runner._compute_pd_transfer_cost(so1)
    assert t1 == 0, "中间 chunk 不应 send"

    # chunk 2: 续 256 = 512 (未完成)
    so2 = _so(
        new_reqs=[],
        cached=_cached(["a"], num_computed=[256], num_output=[0]),
        num_scheduled={"a": 256},
    )
    alloc.step(so2, num_prefix_cached_tokens=0)
    t2, _ = runner._compute_pd_transfer_cost(so2)
    assert t2 == 0

    # chunk 3: 续 488 → 1000 (收尾)
    so3 = _so(
        new_reqs=[],
        cached=_cached(["a"], num_computed=[512], num_output=[0]),
        num_scheduled={"a": 488},
    )
    alloc.step(so3, num_prefix_cached_tokens=0)
    t3, b3 = runner._compute_pd_transfer_cost(so3)
    assert t3 > 0, "末段完成 prefill 应 send"
    assert b3 == alloc.req_kv_bytes("a")


def test_consumer_recv_at_prefill_already_done():
    pd = PDDisaggConfig(role="kv_consumer", connector_name="P2pNcclConnector")
    runner, alloc = _mk_runner(pd)
    # consumer 侧首次见 new_req 时 num_computed = prompt_len (vLLM 表示外部已完成 prefill)
    so = _so(
        new_reqs=[_new_req("a", prompt_len=512, computed=512)],
        cached=_cached([], [], []),
        num_scheduled={"a": 1},  # 进入第一个 decode
    )
    # NOTE: consumer 侧不一定 step allocator (取决于设计), 但 recv bytes 是从 prompt_len 推
    t, b = runner._compute_pd_transfer_cost(so)
    assert t > 0
    # 32 block × block_bytes
    expected = 32 * alloc.block_bytes
    assert b == expected


def test_lmcache_slower_than_pynccl():
    """LMCache (12 GB/s) 应比 PyNccl (25 GB/s) 慢 ~2×."""
    so = _so(
        new_reqs=[_new_req("a", prompt_len=1024)],
        cached=_cached([], [], []),
        num_scheduled={"a": 1024},
    )
    r1, a1 = _mk_runner(
        PDDisaggConfig(role="kv_producer", connector_name="P2pNcclConnector")
    )
    a1.step(so, num_prefix_cached_tokens=0)
    t_nccl, _ = r1._compute_pd_transfer_cost(so)

    r2, a2 = _mk_runner(
        PDDisaggConfig(role="kv_producer", connector_name="LMCacheConnectorV1")
    )
    a2.step(so, num_prefix_cached_tokens=0)
    t_lmc, _ = r2._compute_pd_transfer_cost(so)

    # 带宽 25/12 ≈ 2.08, 加上 latency 5us vs 100us 差异, ratio 1.5-2.5 都合理
    assert t_lmc > t_nccl
    assert t_lmc / t_nccl > 1.5


def test_finished_req_cleared_from_handled_set():
    """finished_req_ids 进来时清掉 _pd_handled_reqs."""
    pd = PDDisaggConfig(role="kv_producer", connector_name="P2pNcclConnector")
    runner, alloc = _mk_runner(pd)
    so1 = _so(
        new_reqs=[_new_req("a", prompt_len=512)],
        cached=_cached([], [], []),
        num_scheduled={"a": 512},
    )
    alloc.step(so1, num_prefix_cached_tokens=0)
    runner._compute_pd_transfer_cost(so1)
    assert "a" in runner._pd_handled_reqs

    so2 = _so(
        new_reqs=[],
        cached=_cached([], [], []),
        num_scheduled={},
        finished={"a"},
    )
    runner._compute_pd_transfer_cost(so2)
    assert "a" not in runner._pd_handled_reqs
