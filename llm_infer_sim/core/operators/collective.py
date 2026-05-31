"""Collective operators (AllReduce / AllGather / ReduceScatter / AllToAll / P2P).

具体子类模式 (强语义): 直接构造 ``AllReduce(name=..., message_bytes=..., world_size=...)``
等, 不用弱语义的 ``Collective(op_subtype=...)``。

`Collective` 基类保留, 接受所有共享字段 + auto-compute `roofline_spec_value` (若没传).
五个具体子类只覆盖 `op_subtype` 默认值, 其它字段全继承.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llm_infer_sim.core.step.runtime import OpRuntime, StepRuntime
from llm_infer_sim.core.operator_schema.signature import OperatorSignature
from llm_infer_sim.core.operators.base import OperatorBase, RooflineSpec
from llm_infer_sim.core.operators.context import OperatorContext


@dataclass(frozen=True, kw_only=True)
class Collective(OperatorBase):
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
    # Phase 3 static contract (op_plan §6/§7). message_bytes_fn computes the
    # step-varying message size; world_size / topology / op_subtype are static.
    count: int = 1
    message_bytes_fn: Callable[[StepRuntime], int] | None = field(
        default=None, compare=False, hash=False, repr=False,
    )

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

    # dtype 走 OperatorBase 默认; parallel/runtime/signature 通信专属, 下面覆盖。

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

    def forward(self, step: StepRuntime) -> OpRuntime | None:
        # world_size <= 1 (e.g. tp=1): the collective is a structural no-op. Gate it
        # out (return None) so the router skips it — same idiom as attention's
        # inactive-regime forward()->None. Lets the graph stay tp-uniform (the op is
        # built unconditionally) while tp=1 contributes no trace entry / no cost.
        if self.world_size <= 1:
            return None
        mb = (self.message_bytes_fn(step) if self.message_bytes_fn is not None
              else self.message_bytes)
        return OpRuntime(
            phase=step.phase, op_subtype=None,  # collective subtype is static
            shape={"message_bytes": int(mb)},
            parallel=dict(self.parallel), runtime=dict(self.runtime),
        )

    def roofline_spec(self, op_runtime: OpRuntime | None = None) -> RooflineSpec:
        if op_runtime is None:
            return self.roofline_spec_value  # type: ignore[return-value]
        return RooflineSpec(
            comm_bytes=float(op_runtime.shape["message_bytes"]),
            comm_type=self.op_subtype,
            op_category="communication",
        )

    def signature(self, op_runtime: OpRuntime | None = None) -> OperatorSignature:
        return self.resolved_signature(
            op_runtime,
            shape_keys=("message_bytes",),
            parallel_keys=(
                "world_size", "tp", "ep", "node_count", "gpus_per_node",
            ),
            runtime_keys=(
                "framework", "framework_version", "backend",
                "algo", "protocol", "topology",
                "execution_mode", "kernel_source",
            ),
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
