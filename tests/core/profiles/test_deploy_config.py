"""V3 §4.6 DeployConfig 单测.

锁住:
  - 字段集 (V3 §4.6 13 个字段 + pd 扩展)
  - 默认值
  - frozen + hashable (供 OperatorDB key / search 用)
"""
from __future__ import annotations

import dataclasses

import pytest

from llm_infer_sim.core.profiles.deploy import DeployConfig


def test_default_values():
    cfg = DeployConfig()
    assert cfg.tp_size == 1
    assert cfg.pp_size == 1
    assert cfg.dp_size == 1
    assert cfg.ep_size == 1
    assert cfg.moe_tp_size == 1
    assert cfg.moe_ep_size == 1
    assert cfg.max_num_batched_tokens is None
    assert cfg.max_num_seqs is None
    assert cfg.block_size == 16
    assert cfg.num_gpu_blocks is None
    assert cfg.execution_mode == "eager"
    assert cfg.backend == "vllm"
    assert cfg.backend_version is None


def test_field_set_is_minimal():
    """V3 §4.6 13 字段 + pd 扩展."""
    names = {f.name for f in dataclasses.fields(DeployConfig)}
    expected = {
        "tp_size", "pp_size", "dp_size", "ep_size",
        "moe_tp_size", "moe_ep_size",
        "max_num_batched_tokens", "max_num_seqs",
        "block_size", "num_gpu_blocks",
        "execution_mode", "backend", "backend_version",
        "pd",
    }
    assert names == expected


def test_frozen():
    cfg = DeployConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tp_size = 2  # type: ignore[misc]


def test_hashable_and_stable_across_equal_instances():
    """同字段实例必须 hash 相等, 供 search / OperatorDB key 用."""
    a = DeployConfig(tp_size=2, ep_size=4, execution_mode="cudagraph")
    b = DeployConfig(tp_size=2, ep_size=4, execution_mode="cudagraph")
    assert a == b
    assert hash(a) == hash(b)


def test_different_instances_have_different_hash():
    a = DeployConfig(tp_size=1)
    b = DeployConfig(tp_size=2)
    assert a != b
    assert hash(a) != hash(b)


def test_execution_mode_choices_are_strings():
    """execution_mode/backend 留作字符串 (V3 §4.6 未限定 enum), 由下游 backend 校验."""
    for mode in ("eager", "cudagraph", "graph"):
        cfg = DeployConfig(execution_mode=mode)
        assert cfg.execution_mode == mode


def test_moe_parallelism_independent_of_top_level():
    """MoE 的 tp/ep 是独立轴 (DeepSeek V4 类型需求), 不强约束 = top-level."""
    cfg = DeployConfig(tp_size=4, ep_size=4, moe_tp_size=1, moe_ep_size=16)
    assert cfg.tp_size == 4
    assert cfg.ep_size == 4
    assert cfg.moe_tp_size == 1
    assert cfg.moe_ep_size == 16
