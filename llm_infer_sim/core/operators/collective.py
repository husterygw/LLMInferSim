"""Collective operators (AllReduce / AllGather / ReduceScatter / AllToAll / P2P).

Step 1 of comm_plan: 替换原来的 ``Collective(op_subtype=...)`` + ``make_collective(kind=...)``
弱语义模式, 改成具体子类:

    AllReduce(name=..., message_bytes=..., world_size=..., ...)
    AllToAll(name=..., message_bytes=..., world_size=..., ...)
    ...

`Collective` 基类保留, 接受所有共享字段 + auto-compute `roofline_spec_value` (若没传).
五个具体子类只覆盖 `op_subtype` 默认值, 其它字段全继承.

`make_collective(...)` 保留为 deprecated wrapper, 转发到对应子类. 等所有模板模板迁完
(qwen / deepseek / 各 moe) 后删除.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True, kw_only=True)
class Collective:
    """通信 op 基类. 通常用具体子类 (AllReduce / AllToAll / ...) 构造,
    不要直接实例化 Collective(op_subtype=...) — 会失去 op_subtype 的静态正确性.
    """

    # ---- 必填 ----
    name: str
    phase: str
    layer_idx: int | None
    message_bytes: int
    world_size: int
    ctx: OperatorContext = field(compare=False, hash=False, repr=False)

    # ---- 可选 / 子类覆盖 ----
    # op_subtype 子类覆盖默认值 (AllReduce → "allreduce" 等). 直接用基类时
    # 必须显式传, 不然会传到 RooflineSpec.comm_type 跟下游 dispatch 不上.
    op_subtype: str = "collective"
    # 不传则在 __post_init__ 里按 (op_subtype, message_bytes) auto-compute.
    roofline_spec_value: RooflineSpec | None = None
    dtype_override: str | None = None
    kernel_source: str = "vllm_default"
    comm_backend: str = "nccl"
    algo: str = ""
    protocol: str = ""
    topology: str = ""
    dependencies: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.roofline_spec_value is None:
            spec = RooflineSpec(
                comm_bytes=float(self.message_bytes),
                comm_type=self.op_subtype,
                op_category="communication",
            )
            object.__setattr__(self, "roofline_spec_value", spec)

    # ---- 静态属性 ----

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
        return self.roofline_spec_value  # type: ignore[return-value]

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


# ---------------------------------------------------------------------------
# Concrete subclasses (op_subtype 静态固定)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class AllReduce(Collective):
    op_subtype: str = "allreduce"


@dataclass(frozen=True, kw_only=True)
class AllGather(Collective):
    op_subtype: str = "allgather"


@dataclass(frozen=True, kw_only=True)
class ReduceScatter(Collective):
    op_subtype: str = "reducescatter"


@dataclass(frozen=True, kw_only=True)
class AllToAll(Collective):
    op_subtype: str = "alltoall"


@dataclass(frozen=True, kw_only=True)
class P2P(Collective):
    op_subtype: str = "p2p"


# ---------------------------------------------------------------------------
# Deprecated wrapper — kept for callers not yet migrated to concrete classes.
# ---------------------------------------------------------------------------

_KIND_TO_CLASS: dict[str, type[Collective]] = {
    "allreduce":      AllReduce,
    "allgather":      AllGather,
    "reducescatter":  ReduceScatter,
    "reduce_scatter": ReduceScatter,
    "alltoall":       AllToAll,
    "p2p":            P2P,
}


def make_collective(
    *,
    kind: str,
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
    """[Deprecated] 用具体子类 (AllReduce / AllToAll / ...) 替代.

    本 wrapper 保留供旧模板 (deepseek, 部分 moe) 渐进迁移; 全部迁完后删除.
    """
    cls = _KIND_TO_CLASS.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown collective kind: {kind!r}; "
            f"supported: {sorted(_KIND_TO_CLASS)}"
        )
    return cls(
        name=name,
        message_bytes=int(message_bytes),
        world_size=world_size,
        phase=phase,
        layer_idx=layer_idx,
        ctx=ctx,
        dtype_override=dtype,
        comm_backend=comm_backend,
        topology=topology,
    )
