"""Collective operator (allreduce / alltoall / allgather / reduce_scatter / p2p).

``make_collective`` helper 给模板直接构造用.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class Collective:
    name: str
    op_subtype: str   # "allreduce" / "alltoall" / "allgather" / "reduce_scatter" / "p2p"
    phase: str
    layer_idx: int | None
    message_bytes: int
    world_size: int
    roofline_spec_value: RooflineSpec
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    comm_backend: str = ""
    algo: str = ""
    protocol: str = ""
    topology: str = ""
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "collective"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {"message_bytes": self.message_bytes}

    @property
    def parallel(self) -> dict[str, Any]:
        return {"world_size": self.world_size}

    @property
    def runtime(self) -> dict[str, Any]:
        out = {
            "framework": self.ctx.framework,
            "framework_version": self.ctx.framework_version,
            "execution_mode": self.ctx.execution_mode,
            "kernel_source": self.kernel_source,
        }
        if self.comm_backend:
            out["backend"] = self.comm_backend
        if self.algo:
            out["algo"] = self.algo
        if self.protocol:
            out["protocol"] = self.protocol
        if self.topology:
            out["topology"] = self.topology
        return out

    def roofline_spec(self) -> RooflineSpec:
        return self.roofline_spec_value

    def signature(self) -> OperatorSignature:
        return OperatorSignature(
            op_kind="collective",
            op_subtype=self.op_subtype,
            dtype=self.dtype,
            shape=to_canonical(project(self.shape, ("message_bytes",))),
            parallel=to_canonical(project(self.parallel, (
                "world_size", "tp", "ep", "node_count", "gpus_per_node",
            ))),
            runtime=to_canonical(project(self.runtime, (
                "framework", "framework_version", "backend",
                "algo", "protocol", "topology",
                "execution_mode", "kernel_source",
            ))),
        )


def make_collective(
    *,
    kind: str,                   # "allreduce" / "alltoall" / "allgather" / "reduce_scatter" / "p2p"
    name: str,
    message_bytes: int,
    world_size: int,
    phase: str,
    layer_idx: int | None,
    ctx: OperatorContext,
    comm_backend: str = "nccl",
    topology: str = "single_node",
    dtype: str = "bf16",
) -> Collective:
    """Convenience helper for constructing Collective ops."""
    formula = RooflineSpec(
        comm_bytes=float(message_bytes),
        comm_type=kind,
        op_category="communication",
    )
    return Collective(
        name=name, op_subtype=kind,
        phase=phase, layer_idx=layer_idx,
        message_bytes=int(message_bytes),
        world_size=world_size,
        ctx=ctx,
        dtype_override=dtype,
        comm_backend=comm_backend,
        topology=topology,
        roofline_spec_value=formula,
    )
