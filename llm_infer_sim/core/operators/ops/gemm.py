"""GEMM operator class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.formulas.gemm import gemm_formula
from llm_infer_sim.core.operators.specs import OperatorFormula

_SHAPE_KEYS = ("m", "n", "k")
_PARALLEL_KEYS = ("tp",)
_RUNTIME_KEYS = ("framework", "framework_version", "execution_mode", "kernel_source")


@dataclass(frozen=True)
class GemmOp:
    name: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    dtype: str
    m: int
    n: int
    k: int
    tp: int | None
    framework: str
    framework_version: str
    execution_mode: str
    kernel_source: str
    weight_bytes_per_elem: float | None = None
    act_bytes_per_elem: float | None = None
    out_bytes_per_elem: float | None = None
    is_kv_proj: bool = False
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "gemm"

    @property
    def shape(self) -> dict[str, Any]:
        return {"m": self.m, "n": self.n, "k": self.k}

    @property
    def parallel(self) -> dict[str, Any]:
        return {"tp": self.tp}

    @property
    def runtime(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "execution_mode": self.execution_mode,
            "kernel_source": self.kernel_source,
        }

    def signature(self) -> OperatorSignature:
        return OperatorSignature(
            op_kind=self.op_kind,
            op_subtype=self.op_subtype,
            dtype=self.dtype,
            shape=to_canonical(project(self.shape, _SHAPE_KEYS)),
            parallel=to_canonical(project(self.parallel, _PARALLEL_KEYS)),
            runtime=to_canonical(project(self.runtime, _RUNTIME_KEYS)),
        )

    def formula(self) -> OperatorFormula:
        return gemm_formula(
            m=self.m,
            n=self.n,
            k=self.k,
            dtype=self.dtype,
            weight_bytes_per_elem=self.weight_bytes_per_elem,
            act_bytes_per_elem=self.act_bytes_per_elem,
            out_bytes_per_elem=self.out_bytes_per_elem,
            is_kv_proj=self.is_kv_proj,
        )

    @classmethod
    def from_record(
        cls,
        record: OperatorRecord,
        *,
        phase: str = "operator_report",
    ) -> "GemmOp":
        if record.signature.op_kind != "gemm":
            raise ValueError(
                f"expected GEMM OperatorRecord, got {record.signature.op_kind!r}"
            )
        shape = dict(record.signature.shape)
        parallel = dict(record.signature.parallel)
        runtime = dict(record.signature.runtime)
        m = int(shape["m"])
        n = int(shape["n"])
        k = int(shape["k"])
        return cls(
            name=f"{record.signature.op_subtype}_{m}x{n}x{k}",
            op_subtype=record.signature.op_subtype,
            phase=phase,
            layer_idx=None,
            dtype=record.signature.dtype,
            m=m,
            n=n,
            k=k,
            tp=parallel.get("tp"),
            framework=runtime.get("framework", record.framework),
            framework_version=runtime.get(
                "framework_version", record.framework_version,
            ),
            execution_mode=runtime.get("execution_mode", record.execution_mode),
            kernel_source=runtime.get("kernel_source", record.kernel_source),
        )
