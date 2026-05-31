"""GEMM operator — #158 ctx-based.

GEMM 持有 OperatorContext, 从 ctx 派生 framework / framework_version / execution_mode /
tp / w_byte / a_byte / kv_byte. signature() 用 ctx-derived 字段构造, 但 ctx 本身不进
hash/eq (field compare=False), 不污染 OperatorDB lookup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llm_infer_sim.core.step.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import OperatorBase, RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext
from llm_infer_sim.core.hardware.device import HardwareConfig
from llm_infer_sim.core.models.config import ModelConfig

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
class GEMM(OperatorBase):
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
    # Phase 2 static contract (op_plan §6). count = static layer multiplicity
    # (replaces GroupedOperator.count). m_fn computes the step-varying M from
    # StepRuntime; n/k are already ctx-static on the op. compare=False so the
    # migration fields don't perturb hash/eq / OperatorDB lookup.
    count: int = 1
    m_fn: Callable[[StepRuntime], int] | None = field(
        default=None, compare=False, hash=False, repr=False,
    )

    @property
    def op_kind(self) -> str:
        return "gemm"

    @property
    def shape(self) -> dict[str, Any]:
        return {"m": self.m, "n": self.n, "k": self.k}

    # dtype / parallel ({"tp": tp_size}) / runtime 走 OperatorBase 默认。

    # ---- Phase 2 static + forward contract ----
    # signature / roofline_spec are dual-mode during migration: called with no
    # op_runtime → legacy (uses self.m/n/k from construction); called with an
    # OpRuntime (from forward()) → uses its shape. The two MUST be numerically
    # identical for the same effective shape (locked by tests) so the eventual
    # engine switch to forward() does not move any sim number.

    def forward(self, step: StepRuntime) -> OpRuntime:
        m = self.m_fn(step) if self.m_fn is not None else self.m
        return OpRuntime(
            phase=step.phase,
            op_subtype=None,  # GEMM subtype is static; signature uses self.op_subtype
            shape={"m": int(m), "n": self.n, "k": self.k},
            parallel=dict(self.parallel),
            runtime=dict(self.runtime),
        )

    def signature(self, op_runtime: OpRuntime | None = None) -> OperatorSignature:
        return self.resolved_signature(
            op_runtime,
            shape_keys=_SHAPE_KEYS,
            parallel_keys=_PARALLEL_KEYS,
            runtime_keys=_RUNTIME_KEYS,
        )

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        if op_runtime is None:
            m, n, k = self.m, self.n, self.k
        else:
            m, n, k = (int(op_runtime.shape["m"]), int(op_runtime.shape["n"]),
                       int(op_runtime.shape["k"]))
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
        out_bytes = int(m * n * out_byte)
        precision = self.op_precision_override if self.op_precision_override else self.dtype
        return RooflineSpec(
            flops=int(2 * m * n * k),
            load_weight=int(k * n * w_byte),
            load_act=int(m * k * a_byte),
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
        from llm_infer_sim.core.deployment.profile import DeploymentProfile
        from llm_infer_sim.core.runtime.profile import RuntimeProfile

        elem_bytes = dtype_to_bytes(record.signature.dtype)
        ctx = OperatorContext(
            model=ModelConfig(),  # dummy, GEMM.formula 不依赖 model
            deployment=DeploymentProfile.flat(tp=int(parallel.get("tp", 1) or 1)),
            runtime=RuntimeProfile.flat(
                execution_mode=runtime.get("execution_mode", "eager"),
                backend=runtime.get("framework", record.framework),
                backend_version=runtime.get(
                    "framework_version", record.framework_version
                ),
            ),
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
