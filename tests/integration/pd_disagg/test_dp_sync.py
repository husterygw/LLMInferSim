"""DP step-latency 同步 (G3, §10.5 4.5 / DP G1-G5).

vLLM v1 DP: 各 dp rank 独立 scheduler, padding-token 强同步, 慢者拖快者. 我们要在
execute_step 后用 all_reduce MAX 在 dp_group 上把 latency 抬到 max → time_emulator
按 max sleep, metrics 按 max 记录.

测试 monkey-patch get_dp_group 注入假 PG, 验证:
  1. dp_size=1 → 直接返本地 latency (fast path, 不调 all_reduce)
  2. dp_size>1 → 调 all_reduce(MAX) 把 local latency 抬到 group max
  3. PG 异常 / 未初始化 → fallback 本地 latency + log warn (不挂)
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from llm_infer_sim.adapters.vllm.virtual_model_runner import VirtualModelRunner


def _mock_runner_for_sync(dp_size: int):
    """构造最小 runner stub, 只测 _sync_dp_latency 方法."""
    r = mock.Mock(spec=VirtualModelRunner)
    r.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=dp_size)
    )
    r._dp_sync_warned = False
    # 绑 unbound method
    r._sync_dp_latency = VirtualModelRunner._sync_dp_latency.__get__(r)
    return r


def test_dp1_fast_path_no_sync():
    """dp_size=1 不走 all_reduce, 直接返本地值. monkey-patch get_dp_group 抛错验证 fast path."""
    r = _mock_runner_for_sync(dp_size=1)
    with mock.patch(
        "vllm.distributed.parallel_state.get_dp_group",
        side_effect=RuntimeError("should not call"),
    ):
        out = r._sync_dp_latency(0.012)
    assert out == 0.012


def test_dp_size_greater_than_1_triggers_max_reduce():
    """dp_size=2 调 all_reduce, 模拟把 0.010 抬到 group max=0.018."""
    r = _mock_runner_for_sync(dp_size=2)

    class _FakeGroup:
        world_size = 2
        device_group = "fake_gloo_pg"

    captured_ops = []

    def fake_all_reduce(tensor, op=None, group=None):
        # 模拟 max reduce 拉到 0.018
        captured_ops.append((tensor.clone(), op, group))
        tensor[0] = 0.018

    with mock.patch("vllm.distributed.parallel_state.get_dp_group",
                    return_value=_FakeGroup()), \
         mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
        out = r._sync_dp_latency(0.010)

    assert out == pytest.approx(0.018, rel=1e-6)
    assert len(captured_ops) == 1
    # MAX op assert
    import torch
    assert captured_ops[0][1] == torch.distributed.ReduceOp.MAX
    assert captured_ops[0][2] == "fake_gloo_pg"


def test_dp_group_none_fallback():
    """get_dp_group 返 None → fallback 本地值, 不挂."""
    r = _mock_runner_for_sync(dp_size=2)
    with mock.patch("vllm.distributed.parallel_state.get_dp_group", return_value=None):
        out = r._sync_dp_latency(0.005)
    assert out == 0.005


def test_dp_group_world_size_1_no_op():
    """dp_group.world_size == 1 (e.g. degenerate)  → fast path."""
    r = _mock_runner_for_sync(dp_size=2)

    class _FakeGroup:
        world_size = 1
        device_group = None

    with mock.patch("vllm.distributed.parallel_state.get_dp_group",
                    return_value=_FakeGroup()):
        out = r._sync_dp_latency(0.007)
    assert out == 0.007


def test_all_reduce_exception_falls_back():
    """all_reduce 抛错 (e.g. PG 未注册) → log warn, 返本地值."""
    r = _mock_runner_for_sync(dp_size=2)

    class _FakeGroup:
        world_size = 2
        device_group = "fake"

    with mock.patch("vllm.distributed.parallel_state.get_dp_group",
                    return_value=_FakeGroup()), \
         mock.patch("torch.distributed.all_reduce",
                    side_effect=RuntimeError("ProcessGroup not ready")):
        out = r._sync_dp_latency(0.003)
    assert out == 0.003
    assert r._dp_sync_warned is True


def test_local_is_already_max():
    """本 rank 已经是慢者 (local > others) → output == local."""
    r = _mock_runner_for_sync(dp_size=2)

    class _FakeGroup:
        world_size = 2
        device_group = "fake"

    def fake_all_reduce(tensor, op=None, group=None):
        # local 是最大值, max reduce 保持 unchanged
        pass

    with mock.patch("vllm.distributed.parallel_state.get_dp_group",
                    return_value=_FakeGroup()), \
         mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
        out = r._sync_dp_latency(0.025)
    assert out == 0.025
