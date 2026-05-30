"""MoE canonicalizer — V3 §5.2 / IMPL_PLAN §2.3.

moe_plan Phase 2: 唯一 AIC-aligned raw → internal canonical 转换点.

internal canonical signature (不变, 跟 FusedMoE op 字段对齐):
    op_kind   = moe
    op_subtype= fused_moe
    dtype                              # 'bf16' / 'fp16'
    shape     = {num_tokens, hidden, moe_intermediate, topk, num_experts,
                 routing_distribution, power_law_alpha}
    parallel  = {tp, ep}
    runtime   = {framework, framework_version, execution_mode, kernel_source}

raw params 接受两种格式:
  (A) AIC-aligned (Phase 2 后新采集):
      {moe_dtype, num_tokens, hidden_size, inter_size, topk, num_experts,
       moe_tp_size, moe_ep_size, distribution, execution_mode}
      distribution: 'balanced' | 'power_law_<alpha>'
  (B) Legacy internal (_legacy/ archive 数据兼容):
      {dtype, num_tokens, hidden, moe_intermediate, topk, num_experts,
       tp, ep, routing_distribution, power_law_alpha, execution_mode}

routing_distribution 必须进入 key — balanced 与 power_law 不能互相命中.
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.operator_schema.canonical import project, to_canonical
from llm_infer_sim.core.operator_schema.signature import OperatorSignature

_SHAPE_KEYS = (
    "num_tokens", "hidden", "moe_intermediate", "topk", "num_experts",
    "routing_distribution", "power_law_alpha",
)
_PARALLEL_KEYS = ("tp", "ep")
_RUNTIME_KEYS = ("framework", "framework_version", "execution_mode", "kernel_source")


# ---------------------------------------------------------------------------
# AIC-aligned → internal canonical mapping (单点)
# ---------------------------------------------------------------------------

def _aic_distribution_to_internal(distribution: str) -> tuple[str, float]:
    """AIC 单字段 distribution → (routing_distribution, power_law_alpha).

    'balanced'         → ('balanced',  0.0)
    'power_law_<a>'    → ('power_law', float(a))
    """
    if distribution == "balanced":
        return "balanced", 0.0
    if distribution.startswith("power_law_"):
        try:
            alpha = float(distribution[len("power_law_"):])
        except ValueError as e:
            raise ValueError(
                f"distribution={distribution!r}: power_law_<alpha> 解析失败"
            ) from e
        return "power_law", alpha
    raise ValueError(
        f"distribution={distribution!r} not supported (仅 balanced / power_law_<alpha>)"
    )


def _moe_dtype_to_internal(moe_dtype: str) -> str:
    """AIC moe_dtype → internal dtype 缩写.

    'bfloat16'  → 'bf16'
    'float16'   → 'fp16'
    其他保持原样 (后续 fp8 / nvfp4 等 Phase 6+ 接入时再扩展)。
    """
    table = {"bfloat16": "bf16", "float16": "fp16"}
    return table.get(moe_dtype, moe_dtype)


def _params_to_internal(params: dict[str, Any]) -> dict[str, Any]:
    """raw params → internal canonical params. 自动检测 AIC vs legacy."""
    if "moe_dtype" in params or "hidden_size" in params or "moe_tp_size" in params:
        # AIC-aligned: 单点字段转换
        rd, alpha = _aic_distribution_to_internal(str(params["distribution"]))
        return {
            "num_tokens": params["num_tokens"],
            "hidden": params["hidden_size"],
            "moe_intermediate": params["inter_size"],
            "topk": params["topk"],
            "num_experts": params["num_experts"],
            "tp": params["moe_tp_size"],
            "ep": params["moe_ep_size"],
            "routing_distribution": rd,
            "power_law_alpha": alpha,
            "dtype": _moe_dtype_to_internal(str(params["moe_dtype"])),
            "execution_mode": params["execution_mode"],
        }
    # legacy internal: 原样返回 (用于 _legacy/ archive 数据 import)
    return params


def moe_case_params_to_signature(
    params: dict[str, Any],
    *,
    framework: str,
    framework_version: str,
    kernel_source: str,
) -> OperatorSignature:
    """collector MoE Case.params + RawRecord top-level → OperatorSignature.

    Accept AIC-aligned (Phase 2+) 或 legacy internal (_legacy archive) params.
    内部 canonical signature 字段保持不变.
    """
    internal = _params_to_internal(params)
    runtime = {
        "framework": framework,
        "framework_version": framework_version,
        "execution_mode": internal["execution_mode"],
        "kernel_source": kernel_source,
    }
    return OperatorSignature(
        op_kind="moe",
        op_subtype="fused_moe",
        dtype=internal["dtype"],
        shape=to_canonical(project(internal, _SHAPE_KEYS)),
        parallel=to_canonical(project(internal, _PARALLEL_KEYS)),
        runtime=to_canonical(runtime),
    )


#: canonical kernel identity for the signature. The model graph names the node
#: ``routed_experts`` (for trace / humans); the DB/kernel canonical subtype is
#: ``fused_moe``. These must not be conflated — query side must sign as fused_moe
#: to match the collector (which hardcodes fused_moe), regardless of op.op_subtype.
_SIGNATURE_OP_SUBTYPE = "fused_moe"


def _moe_kernel_parallel(tp: int, ep: int) -> tuple[int, int]:
    """Global (tp, ep) → per-kernel (moe_tp, moe_ep) for the signature.

    Contract: the signature's tp/ep must describe how the *expert kernel* is
    sharded, not the global parallel config.
      - ep > 1 (vLLM EP path): experts split across the EP group, a single
        expert's weight is NOT TP-sharded → moe_tp = 1.
      - ep == 1 (TP-only MoE): expert intermediate is TP-sharded → moe_tp = tp.
    Mirrors the MoE op's own compute (``expert_dim // tp if ep==1 else
    expert_dim``) and the collector convention (moe_tp_size=1 under EP). Without
    this, a global tp=4 EP run signs tp=4 but the collector stored tp=1 → miss.
    """
    moe_ep = ep
    moe_tp = tp if ep == 1 else 1
    return moe_tp, moe_ep


def moe_operator_to_signature(op: Any, op_runtime: Any = None) -> OperatorSignature:
    """runtime operator descriptor → OperatorSignature.

    qwen.py / deepseek.py 直接构造 MoE (op_kind=moe, op_subtype=routed_experts),
    此 canonicalizer 把它转成 OperatorDB signature: subtype 归一到 fused_moe,
    parallel 用 per-kernel 切分 (见 _moe_kernel_parallel).

    op_runtime 给定时 (Phase 4 静态契约) 从它读 shape/parallel/runtime (num_tokens
    来自 step); 否则从 op 的属性读 (legacy)。dtype 始终取 op (静态)。
    """
    if op.op_kind != "moe":
        raise ValueError(f"expected op_kind=moe, got {op.op_kind!r}")
    shape = op_runtime.shape if op_runtime is not None else op.shape
    parallel = dict(op_runtime.parallel if op_runtime is not None else op.parallel)
    runtime = op_runtime.runtime if op_runtime is not None else op.runtime
    moe_tp, moe_ep = _moe_kernel_parallel(int(parallel["tp"]), int(parallel["ep"]))
    return OperatorSignature(
        op_kind="moe",
        op_subtype=_SIGNATURE_OP_SUBTYPE,
        dtype=op.dtype,
        shape=to_canonical(project(dict(shape), _SHAPE_KEYS)),
        parallel=to_canonical({"tp": moe_tp, "ep": moe_ep}),
        runtime=to_canonical(project(dict(runtime), _RUNTIME_KEYS)),
    )
