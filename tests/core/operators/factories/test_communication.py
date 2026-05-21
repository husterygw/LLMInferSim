"""CollectiveOpFactory 单测 — V3 §5.4 / IMPL_PLAN §5.

锁住 5 个 collective subtype: allreduce / alltoall / allgather / reduce_scatter / p2p.
每个 subtype 应能生成有效 CollectiveOp,
signature 通过 dispatch 进 OperatorDB 之路径相互独立 (不互相命中).
"""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operators.factories.communication import CollectiveOpFactory
from llm_infer_sim.core.operators.ops import CollectiveOp
from llm_infer_sim.core.operator_schema import operator_to_signature
from llm_infer_sim.core.profiles.deploy import DeployConfig


@pytest.fixture
def factory():
    return CollectiveOpFactory(DeployConfig(tp_size=4, ep_size=4))


# ---- 每个 subtype 都能 emit ----

def test_allreduce(factory):
    op = factory.allreduce(
        name="ar", message_bytes=1024, phase="prefill",
        layer_idx=0, world_size=4,
    )
    assert isinstance(op, CollectiveOp)
    assert op.op_subtype == "allreduce"
    f = op.formula()
    assert f.comm_type == "allreduce"
    assert f.comm_bytes == 1024.0
    assert f.op_category == "communication"


def test_alltoall(factory):
    op = factory.alltoall(
        name="a2a", message_bytes=2048, phase="decode",
        layer_idx=0, world_size=4,
    )
    assert op.op_subtype == "alltoall"
    assert op.formula().comm_type == "alltoall"


def test_allgather(factory):
    op = factory.allgather(
        name="ag", message_bytes=1024, phase="prefill",
        layer_idx=0, world_size=4,
    )
    assert op.op_subtype == "allgather"
    assert op.formula().comm_type == "allgather"


def test_reduce_scatter(factory):
    op = factory.reduce_scatter(
        name="rs", message_bytes=1024, phase="prefill",
        layer_idx=0, world_size=4,
    )
    assert op.op_subtype == "reduce_scatter"
    assert op.formula().comm_type == "reduce_scatter"


def test_p2p(factory):
    op = factory.p2p(
        name="kv_send", message_bytes=8192, phase="decode",
        layer_idx=None,
    )
    assert op.op_subtype == "p2p"
    assert op.formula().comm_type == "p2p"
    # default world_size = 2 (sender + receiver)
    assert op.parallel["world_size"] == 2


# ---- 不同 subtype → 不同 signature (OperatorDB 隔离) ----

def test_all_five_collective_signatures_distinct(factory):
    """5 个 subtype 的同 (message_bytes, world_size) 应在 OperatorSignature 上互不相同."""
    ops = [
        factory.allreduce(name="x", message_bytes=1024, phase="prefill",
                          layer_idx=0, world_size=4),
        factory.alltoall(name="x", message_bytes=1024, phase="prefill",
                         layer_idx=0, world_size=4),
        factory.allgather(name="x", message_bytes=1024, phase="prefill",
                          layer_idx=0, world_size=4),
        factory.reduce_scatter(name="x", message_bytes=1024, phase="prefill",
                               layer_idx=0, world_size=4),
        factory.p2p(name="x", message_bytes=1024, phase="prefill",
                    layer_idx=0, world_size=4),
    ]
    hashes = {operator_to_signature(op).stable_hash() for op in ops}
    assert len(hashes) == 5


# ---- shape / parallel / runtime 字段一致性 ----

def test_collective_op_parallel_carries_world_tp_ep(factory):
    op = factory.allreduce(name="x", message_bytes=128, phase="prefill",
                            layer_idx=0, world_size=4)
    p = op.parallel
    assert p["world_size"] == 4
    assert p["tp"] == 4
    assert p["ep"] == 4


def test_collective_op_runtime_has_backend_and_topology(factory):
    op = factory.alltoall(name="x", message_bytes=128, phase="prefill",
                          layer_idx=0, world_size=4)
    rt = op.runtime
    assert rt["backend"] == "nccl"
    assert rt["topology"] == "single_node"
    assert rt["framework"] == "vllm"


def test_topology_override():
    """CollectiveOpFactory 接收 topology hint, 在 runtime 字段反映."""
    f = CollectiveOpFactory(DeployConfig(tp_size=16), topology="cross_node")
    op = f.allreduce(name="x", message_bytes=128, phase="prefill",
                     layer_idx=0, world_size=16)
    assert op.runtime["topology"] == "cross_node"
