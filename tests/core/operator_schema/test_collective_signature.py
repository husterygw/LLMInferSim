"""Collective canonicalizer 单测 — Step 2.4.

锁住:
  - num_gpus → world_size mapping
  - topology_hint → topology in runtime
  - in_context 不进 signature (sweep dim, 不是 identity)
  - collector ↔ runtime signature 一致
"""
from __future__ import annotations

from llm_infer_sim.core.operator_schema.collective import (
    collective_case_params_to_signature,
    collective_virtual_op_to_signature,
)
from llm_infer_sim.core.operators.ops import CollectiveOp
from llm_infer_sim.core.operators.specs import OperatorFormula


_CTX = dict(
    framework="vllm", framework_version="0.20.1",
    kernel_source="vllm_nccl_allreduce", backend="nccl",
)


def _coll_case(subtype="allreduce", num_gpus=2, bytes_=1024 * 1024,
                topology="single_node", in_context=False, mode="eager"):
    return {
        "op_subtype": subtype,
        "num_gpus": num_gpus,
        "message_size_bytes": bytes_,
        "dtype": "bf16",
        "topology_hint": topology,
        "in_context": in_context,
        "execution_mode": mode,
    }


def _coll_op(subtype="allreduce", world_size=2, bytes_=1024 * 1024,
              topology="single_node", mode="eager"):
    return CollectiveOp(
        name="attn_allreduce", op_kind="collective", op_subtype=subtype,
        phase="prefill", layer_idx=0, dtype="bf16",
        shape_fields={"message_bytes": bytes_},
        parallel_fields={"world_size": world_size},
        runtime_fields={
            "framework": "vllm", "framework_version": "0.20.1",
            "backend": "nccl", "topology": topology,
            "execution_mode": mode, "kernel_source": "vllm_nccl_allreduce",
        },
        formula_value=OperatorFormula(
            comm_bytes=bytes_, comm_type=subtype, op_category="communication",
        ),
    )


def test_collector_and_runtime_signature_match():
    sig_c = collective_case_params_to_signature(_coll_case(), **_CTX)
    sig_r = collective_virtual_op_to_signature(_coll_op())
    assert sig_c == sig_r
    assert sig_c.stable_hash() == sig_r.stable_hash()


def test_in_context_does_not_enter_signature():
    """collector params.in_context 是 sweep flag (warm vs cold), 不进 signature key."""
    sig_warm = collective_case_params_to_signature(
        _coll_case(in_context=True), **_CTX,
    )
    sig_cold = collective_case_params_to_signature(
        _coll_case(in_context=False), **_CTX,
    )
    assert sig_warm == sig_cold


def test_world_size_mapped_from_num_gpus():
    sig = collective_case_params_to_signature(_coll_case(num_gpus=4), **_CTX)
    parallel = dict(sig.parallel)
    assert parallel["world_size"] == 4


def test_topology_in_runtime_key():
    sig_a = collective_case_params_to_signature(
        _coll_case(topology="single_node"), **_CTX,
    )
    sig_b = collective_case_params_to_signature(
        _coll_case(topology="cross_numa"), **_CTX,
    )
    assert sig_a != sig_b


def test_message_size_in_shape_key():
    sig_a = collective_case_params_to_signature(
        _coll_case(bytes_=1024), **_CTX,
    )
    sig_b = collective_case_params_to_signature(
        _coll_case(bytes_=1024 * 1024), **_CTX,
    )
    assert sig_a != sig_b


def test_op_subtype_separates_allreduce_alltoall():
    sig_ar = collective_case_params_to_signature(
        _coll_case(subtype="allreduce"), **_CTX,
    )
    sig_a2a = collective_case_params_to_signature(
        _coll_case(subtype="alltoall"), **_CTX,
    )
    assert sig_ar != sig_a2a


def test_world_size_in_parallel_key():
    sig_2 = collective_case_params_to_signature(
        _coll_case(num_gpus=2), **_CTX,
    )
    sig_4 = collective_case_params_to_signature(
        _coll_case(num_gpus=4), **_CTX,
    )
    assert sig_2 != sig_4
