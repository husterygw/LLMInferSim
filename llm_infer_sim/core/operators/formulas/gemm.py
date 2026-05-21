"""GEMM roofline formula builders."""
from __future__ import annotations

from llm_infer_sim.core.operators.specs import OperatorFormula


def dtype_to_bytes(dtype: str) -> float:
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


def gemm_formula(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str,
    weight_bytes_per_elem: float | None = None,
    act_bytes_per_elem: float | None = None,
    out_bytes_per_elem: float | None = None,
    is_kv_proj: bool = False,
) -> OperatorFormula:
    """Standard GEMM formula: C[M,N] = A[M,K] x B[K,N].

    is_kv_proj=True: output 写到 KV cache, 不在 activation. 用于 MLA kv_a_proj_with_mqa
    这类把投影结果直接送进 paged cache 的 op.
    """
    elem_bytes = dtype_to_bytes(dtype)
    w_byte = elem_bytes if weight_bytes_per_elem is None else weight_bytes_per_elem
    a_byte = elem_bytes if act_bytes_per_elem is None else act_bytes_per_elem
    out_byte = elem_bytes if out_bytes_per_elem is None else out_bytes_per_elem
    out_bytes = int(m * n * out_byte)
    return OperatorFormula(
        flops=int(2 * m * n * k),
        load_weight=int(k * n * w_byte),
        load_act=int(m * k * a_byte),
        store_act=0 if is_kv_proj else out_bytes,
        store_kv_cache=out_bytes if is_kv_proj else 0,
        op_precision=dtype,
        op_category="matmul",
    )
