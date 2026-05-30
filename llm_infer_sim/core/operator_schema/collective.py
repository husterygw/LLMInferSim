"""Collective canonicalizer — V3 §5.4 / IMPL_PLAN §2.4.

字段:
    op_kind   = collective
    op_subtype= allreduce / allgather / reduce_scatter / alltoall / p2p
    dtype
    shape     = {message_bytes}
    parallel  = {world_size, tp, ep, node_count, gpus_per_node}
    runtime   = {framework, framework_version, backend, algo, protocol,
                 topology, execution_mode, kernel_source}

Mapping:
    collector params num_gpus           -> world_size
    collector params message_size_bytes -> message_bytes
    collector params topology_hint      -> topology
    collector params 不带 algo/protocol/tp/ep/node_count/gpus_per_node 这些通过 ctx 传入
    collector params 的 in_context 不进 signature (是 sweep dimension, 不是 op identity)
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature

_SHAPE_KEYS = ("message_bytes",)
_PARALLEL_KEYS = ("world_size", "tp", "ep", "node_count", "gpus_per_node")
_RUNTIME_KEYS = (
    "framework", "framework_version", "backend",
    "algo", "protocol", "topology",
    "execution_mode", "kernel_source",
)


def collective_case_params_to_signature(
    params: dict[str, Any],
    *,
    framework: str,
    framework_version: str,
    kernel_source: str,
    backend: str = "nccl",
    algo: str | None = None,
    protocol: str | None = None,
    tp: int | None = None,
    ep: int | None = None,
    node_count: int | None = None,
    gpus_per_node: int | None = None,
) -> OperatorSignature:
    """collector collective Case.params + RawRecord top-level → OperatorSignature.

    Case.params 必含: op_subtype, num_gpus, message_size_bytes, dtype, topology_hint,
                      in_context, execution_mode
    """
    shape_fields = {"message_bytes": int(params["message_size_bytes"])}
    parallel_fields = {
        "world_size": int(params["num_gpus"]),
        "tp": tp,
        "ep": ep,
        "node_count": node_count,
        "gpus_per_node": gpus_per_node,
    }
    runtime_fields = {
        "framework": framework,
        "framework_version": framework_version,
        "backend": backend,
        "algo": algo,
        "protocol": protocol,
        "topology": params["topology_hint"],
        "execution_mode": params["execution_mode"],
        "kernel_source": kernel_source,
    }
    return OperatorSignature(
        op_kind="collective",
        op_subtype=params["op_subtype"],
        dtype=params["dtype"],
        shape=to_canonical(shape_fields),
        parallel=to_canonical(parallel_fields),
        runtime=to_canonical(runtime_fields),
    )


def collective_operator_to_signature(op: Any) -> OperatorSignature:
    """runtime operator descriptor → OperatorSignature."""
    if op.op_kind != "collective":
        raise ValueError(f"expected op_kind=collective, got {op.op_kind!r}")
    return OperatorSignature(
        op_kind="collective",
        op_subtype=op.op_subtype,
        dtype=op.dtype,
        shape=to_canonical(project(op.shape, _SHAPE_KEYS)),
        parallel=to_canonical(project(op.parallel, _PARALLEL_KEYS)),
        runtime=to_canonical(project(op.runtime, _RUNTIME_KEYS)),
    )
