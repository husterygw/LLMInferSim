"""Linear projection operators: QKV, output, FFN up/gate/down.

Each function returns an OperatorProfile matching the exact same OPs and
memory access formulas as the original model_analyzer.py inline code.

阶段 3 新增 fusion 算子 (详设 §4.7.1a):
  - fused_qkv_gemm  : QKVParallelLinear, 输入只读 1 次 (vs naive 3 次)
  - fused_gate_up_gemm : MergedColumnParallelLinear, 同上
"""

from llm_infer_sim.core.ops.base import OperatorProfile


from llm_infer_sim.core.profiles.shape_buckets import (
    OP_KIND_DENSE_GEMM, dense_efficiency_key,
)


def linear_layer(
    name: str,
    ic: int,
    oc: int,
    tokens: int,
    w_byte: float,
    a_byte: float,
    kv_byte: float,
    is_kv_proj: bool = False,
    op_precision: str = "",
    store_a_byte: float | None = None,
) -> OperatorProfile:
    """Generic linear layer: y = x @ W^T.

    Args:
        name: layer name (e.g., "q_proj", "gate_proj")
        ic: input channels
        oc: output channels
        tokens: number of tokens (batch*1 for decode, batch*seqlen for prefill)
        w_byte: weight byte width
        a_byte: activation byte width
        kv_byte: KV cache byte width
        is_kv_proj: True for k_proj/v_proj (output goes to KV cache, not activation)
        op_precision: per-op precision override for roofline ("fp8"|"bf16"|"fp32"|"")
        store_a_byte: output activation byte width override (defaults to a_byte)
    """
    out_a_byte = a_byte if store_a_byte is None else store_a_byte
    return OperatorProfile(
        name=name,
        op_category="matmul",
        flops=ic * oc * tokens * 2,
        load_weight=int(ic * oc * w_byte),
        load_act=int(ic * tokens * a_byte),
        store_act=0 if is_kv_proj else int(oc * tokens * out_a_byte),
        store_kv_cache=int(oc * tokens * kv_byte) if is_kv_proj else 0,
        op_precision=op_precision,
        efficiency_key=dense_efficiency_key(OP_KIND_DENSE_GEMM, tokens),
    )


def fused_qkv_gemm(
    name: str,
    hidden: int,
    num_q_heads_per_tp: int,
    num_kv_heads_per_tp: int,
    head_dim: int,
    tokens: int,
    w_byte: float,
    a_byte: float,
    kv_byte: float,
    op_precision: str = "",
) -> OperatorProfile:
    """QKVParallelLinear: [Q;K;V] = X @ [W_q;W_k;W_v]  (详设 §4.7.1a (1))。

    vs naive 3 个独立 GEMM:
      FLOPs 不变, 但输入 X 只读 1 次 (省 2 次 M*K*a_byte 读)。
    Q 输出走 store_act (送进 attention), K/V 输出走 store_kv_cache。
    """
    n_q = num_q_heads_per_tp * head_dim
    n_kv = num_kv_heads_per_tp * head_dim
    n_total = n_q + 2 * n_kv  # Q + K + V output dim
    return OperatorProfile(
        name=name,
        op_category="matmul",
        flops=hidden * n_total * tokens * 2,
        load_weight=int(hidden * n_total * w_byte),
        load_act=int(hidden * tokens * a_byte),         # 输入只读 1 次
        store_act=int(n_q * tokens * a_byte),           # Q 输出
        store_kv_cache=int(2 * n_kv * tokens * kv_byte),  # K + V → KV cache
        op_precision=op_precision,
        efficiency_key=dense_efficiency_key(OP_KIND_DENSE_GEMM, tokens),
    )


def fused_gate_up_gemm(
    name: str,
    hidden: int,
    intermediate_per_tp: int,
    tokens: int,
    w_byte: float,
    a_byte: float,
    op_precision: str = "",
) -> OperatorProfile:
    """MergedColumnParallelLinear: [gate;up] = X @ [W_gate;W_up]  (详设 §4.7.1a (2))。

    vs naive gate_proj + up_proj 两个独立 GEMM:
      FLOPs 不变, 输入 X 只读 1 次 (省 1 次 M*K*a_byte 读)。
    输出形状 [tokens, 2 * intermediate / tp], 后续 SiLU activation 把它砍半。
    """
    n_total = 2 * intermediate_per_tp
    return OperatorProfile(
        name=name,
        op_category="matmul",
        flops=hidden * n_total * tokens * 2,
        load_weight=int(hidden * n_total * w_byte),
        load_act=int(hidden * tokens * a_byte),
        store_act=int(n_total * tokens * a_byte),
        op_precision=op_precision,
        efficiency_key=dense_efficiency_key(OP_KIND_DENSE_GEMM, tokens),
    )
