"""GEMM operator — #158 ctx-based.

GEMM 持有 OperatorContext, 从 ctx 派生 framework / framework_version / execution_mode /
tp / w_byte / a_byte / kv_byte. signature() 用 ctx-derived 字段构造, 但 ctx 本身不进
hash/eq (field compare=False), 不污染 OperatorDB lookup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig
from llm_infer_sim.core.profiles.model_config import ModelConfig

_SHAPE_KEYS = ("m", "n", "k")
_PARALLEL_KEYS = ("tp",)
_RUNTIME_KEYS = ("framework", "framework_version", "execution_mode", "kernel_source")


def dtype_to_bytes(dtype: str) -> float:
    """dtype 字符串 → 每元素字节数."""
    normalized = dtype.lower()
    if normalized in ("bf16", "bfloat16", "fp16", "float16"):
        return 2.0
    if normalized in ("fp32", "float32"):
        return 4.0
    if normalized in ("fp8", "int8"):
        return 1.0
    if normalized in ("fp4", "int4"):
        return 0.5
    raise ValueError(f"unsupported GEMM dtype: {dtype!r}")


@dataclass(frozen=True)
class GEMM:
    name: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    m: int
    n: int
    k: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)
    dtype_override: str | None = None
    weight_bytes_per_elem: float | None = None
    act_bytes_per_elem: float | None = None
    out_bytes_per_elem: float | None = None
    op_precision_override: str | None = None
    kernel_source: str = "vllm_default"
    is_kv_proj: bool = False
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def op_kind(self) -> str:
        return "gemm"

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

    @property
    def shape(self) -> dict[str, Any]:
        return {"m": self.m, "n": self.n, "k": self.k}

    @property
    def parallel(self) -> dict[str, Any]:
        return {"tp": self.ctx.tp_size}

    @property
    def runtime(self) -> dict[str, Any]:
        return {
            "framework": self.ctx.framework,
            "framework_version": self.ctx.framework_version,
            "execution_mode": self.ctx.execution_mode,
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

    def roofline_spec(self) -> RooflineSpec:
        w_byte = (
            self.weight_bytes_per_elem if self.weight_bytes_per_elem is not None
            else self.ctx.w_byte
        )
        a_byte = (
            self.act_bytes_per_elem if self.act_bytes_per_elem is not None
            else self.ctx.a_byte
        )
        if self.out_bytes_per_elem is not None:
            out_byte = self.out_bytes_per_elem
        elif self.is_kv_proj:
            out_byte = self.ctx.kv_byte
        else:
            out_byte = a_byte
        out_bytes = int(self.m * self.n * out_byte)
        precision = self.op_precision_override if self.op_precision_override else self.dtype
        return RooflineSpec(
            flops=int(2 * self.m * self.n * self.k),
            load_weight=int(self.k * self.n * w_byte),
            load_act=int(self.m * self.k * a_byte),
            store_act=0 if self.is_kv_proj else out_bytes,
            store_kv_cache=out_bytes if self.is_kv_proj else 0,
            op_precision=precision,
            op_category="matmul",
        )

    @classmethod
    def from_record(
        cls,
        record: OperatorRecord,
        *,
        hw: HardwareConfig,
        phase: str = "operator_report",
    ) -> "GEMM":
        """构造 GEMM (供 scripts/report_operator_roofline_gap 用)."""
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
        deploy = DeployConfig(
            tp_size=int(parallel.get("tp", 1) or 1),
            execution_mode=runtime.get("execution_mode", "eager"),
            backend=runtime.get("framework", record.framework),
            backend_version=runtime.get("framework_version", record.framework_version),
        )
        elem_bytes = dtype_to_bytes(record.signature.dtype)
        ctx = OperatorContext(
            model=ModelConfig(),  # dummy, GEMM.formula 不依赖 model
            deploy=deploy,
            hw=hw,
            w_byte=elem_bytes, a_byte=elem_bytes, kv_byte=elem_bytes,
            dtype=record.signature.dtype,
        )
        return cls(
            name=f"{record.signature.op_subtype}_{m}x{n}x{k}",
            op_subtype=record.signature.op_subtype,
            phase=phase,
            layer_idx=None,
            m=m,
            n=n,
            k=k,
            ctx=ctx,
            kernel_source=runtime.get("kernel_source", record.kernel_source),
        )
