"""Non-GEMM semantic operator classes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.specs import OperatorFormula


@dataclass(frozen=True)
class FormulaOp:
    name: str
    op_kind: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    dtype: str
    shape_fields: dict[str, Any]
    parallel_fields: dict[str, Any]
    runtime_fields: dict[str, Any]
    formula_value: OperatorFormula
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def shape(self) -> dict[str, Any]:
        return dict(self.shape_fields)

    @property
    def parallel(self) -> dict[str, Any]:
        return dict(self.parallel_fields)

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.runtime_fields)

    def formula(self) -> OperatorFormula:
        return self.formula_value

    def signature(self) -> OperatorSignature:
        raise ValueError(
            f"{self.op_kind!r} is not in the OperatorDB signature contract"
        )


@dataclass(frozen=True)
class EmbeddingOp(FormulaOp):
    pass


@dataclass(frozen=True)
class NormOp(FormulaOp):
    pass


@dataclass(frozen=True)
class ElementwiseOp(FormulaOp):
    pass


@dataclass(frozen=True)
class KvTransferOp(FormulaOp):
    pass


@dataclass(frozen=True)
class AttentionOp(FormulaOp):
    def signature(self) -> OperatorSignature:
        if self.op_kind != "attention":
            raise ValueError(f"expected op_kind=attention, got {self.op_kind!r}")
        return OperatorSignature(
            op_kind="attention",
            op_subtype=self.op_subtype,
            dtype=self.dtype,
            shape=to_canonical(project(self.shape, (
                "num_tokens", "num_seqs", "q_len", "kv_len",
                "num_q_heads", "num_kv_heads", "head_dim",
            ))),
            parallel=to_canonical(project(self.parallel, ("tp",))),
            runtime=to_canonical(project(self.runtime, (
                "framework", "framework_version", "execution_mode",
                "kernel_source", "attention_backend", "kv_dtype", "block_size",
            ))),
        )


@dataclass(frozen=True)
class CollectiveOp(FormulaOp):
    def signature(self) -> OperatorSignature:
        if self.op_kind != "collective":
            raise ValueError(f"expected op_kind=collective, got {self.op_kind!r}")
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


@dataclass(frozen=True)
class FusedMoeOp(FormulaOp):
    def signature(self) -> OperatorSignature:
        if self.op_kind != "moe":
            raise ValueError(f"expected op_kind=moe, got {self.op_kind!r}")
        return OperatorSignature(
            op_kind="moe",
            op_subtype=self.op_subtype,
            dtype=self.dtype,
            shape=to_canonical(project(self.shape, (
                "num_tokens", "hidden", "moe_intermediate", "topk",
                "num_experts", "routing_distribution", "power_law_alpha",
            ))),
            parallel=to_canonical(project(self.parallel, ("tp", "ep"))),
            runtime=to_canonical(project(self.runtime, (
                "framework", "framework_version", "execution_mode", "kernel_source",
            ))),
        )
