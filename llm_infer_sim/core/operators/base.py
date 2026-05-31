"""Operator base layer — Operator Protocol + OperatorBase 实现基类 + RooflineSpec.

每个 op class 在 operators/{gemm,norm,elementwise,embedding,attention,collective,moe,
mla}.py 里独立; base.py 提供 contract (Operator Protocol)、公共实现 (OperatorBase) 和
通用 payload (RooflineSpec)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llm_infer_sim.core.step.runtime import OpRuntime
from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature

if TYPE_CHECKING:
    from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True)
class RooflineSpec:
    """Roofline/communication formula payload shared by all operator classes."""

    flops: int = 0
    load_weight: int = 0
    load_act: int = 0
    store_act: int = 0
    load_kv_cache: int = 0
    store_kv_cache: int = 0
    op_precision: str = ""
    comm_bytes: float = 0.0
    comm_type: str = ""
    op_category: str = ""

    @property
    def mem_bytes(self) -> int:
        return (
            self.load_weight
            + self.load_act
            + self.store_act
            + self.load_kv_cache
            + self.store_kv_cache
        )

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.mem_bytes if self.mem_bytes > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "flops": self.flops,
            "load_weight": self.load_weight,
            "load_act": self.load_act,
            "store_act": self.store_act,
            "load_kv_cache": self.load_kv_cache,
            "store_kv_cache": self.store_kv_cache,
            "op_precision": self.op_precision,
            "comm_bytes": self.comm_bytes,
            "comm_type": self.comm_type,
            "op_category": self.op_category,
        }


@runtime_checkable
class Operator(Protocol):
    """Semantic operator protocol consumed by cost backends."""

    name: str
    op_kind: str
    op_subtype: str
    phase: str
    layer_idx: int | None
    dtype: str
    dependencies: tuple[str, ...]
    tags: tuple[str, ...]

    @property
    def shape(self) -> dict[str, Any]: ...

    @property
    def parallel(self) -> dict[str, Any]: ...

    @property
    def runtime(self) -> dict[str, Any]: ...

    def signature(self) -> OperatorSignature: ...

    def roofline_spec(self) -> RooflineSpec: ...


# ============================================================================
# OperatorBase — 语义 op 的实现复用基类 (基类化阶段一).
# ============================================================================

class OperatorBase:
    """Shared implementation for semantic operators.

    **不是 dataclass, 不声明任何 field** —— 只收各 op 逐字重复的属性/helper
    (dtype / 默认 parallel / 默认 runtime / signature 默认抛错 / resolved_signature)。
    具体 op 继续 ``@dataclass(frozen=True)`` 并继承本类; 下面的属性引用每个子类都声明的
    字段 (``ctx`` / ``dtype_override`` / ``kernel_source``)。有额外 parallel/runtime
    键的 op (Collective / Attention / MLAAttention / MoE / MoEDispatch) 覆盖对应属性。

    结构类型走 ``Operator`` Protocol —— 不继承本类的对象只要满足 Protocol 即可。
    """

    # 以下属性由子类 (dataclass) 提供; 这里只声明给类型检查/读者, 非 dataclass field
    # (本类不是 dataclass, 注解不会被子类的 @dataclass 收为字段)。
    ctx: OperatorContext
    dtype_override: str | None
    kernel_source: str
    op_kind: str
    op_subtype: str

    @property
    def dtype(self) -> str:
        return self.dtype_override if self.dtype_override else self.ctx.dtype

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

    def signature(self, op_runtime: OpRuntime | None = None) -> OperatorSignature:
        """默认: 不在 OperatorDB 契约里的 op (norm/embedding/elementwise) 抛错.
        DB op (GEMM/Collective/Attention/MLAAttention) 覆盖, 通常委托
        ``resolved_signature``。"""
        raise ValueError(f"{self.op_kind} not in OperatorDB signature contract")

    def resolved_signature(
        self,
        op_runtime: OpRuntime | None,
        *,
        shape_keys: tuple[str, ...],
        parallel_keys: tuple[str, ...],
        runtime_keys: tuple[str, ...],
    ) -> OperatorSignature:
        """dual-mode signature 构造: op_runtime=None → self 静态字段; 否则用
        OpRuntime 解析值 (subtype = op_runtime.op_subtype or self.op_subtype)。
        各 DB op 的 signature() 收成对本方法的一行调用 (口径逐字节不变)。"""
        if op_runtime is None:
            shape, parallel, runtime = self.shape, self.parallel, self.runtime
            subtype = self.op_subtype
        else:
            shape, parallel, runtime = (
                op_runtime.shape, op_runtime.parallel, op_runtime.runtime,
            )
            subtype = op_runtime.op_subtype or self.op_subtype
        return OperatorSignature(
            op_kind=self.op_kind,
            op_subtype=subtype,
            dtype=self.dtype,
            shape=to_canonical(project(dict(shape), shape_keys)),
            parallel=to_canonical(project(dict(parallel), parallel_keys)),
            runtime=to_canonical(project(dict(runtime), runtime_keys)),
        )
