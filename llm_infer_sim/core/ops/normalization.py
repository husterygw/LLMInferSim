"""Normalization, residual add, and activation operators.

Formulas extracted verbatim from model_analyzer.py.
"""

from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.profiles.shape_buckets import (
    OP_KIND_RMSNORM, OP_KIND_SWIGLU, dense_efficiency_key,
)


def norm_layer(
    name: str,
    tokens: int,
    hidden_size: int,
    a_byte: float,
) -> OperatorProfile:
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.

    Per-element FLOPs (4):
      1 mul (x²) + 1 add (reduce 累加) + 1 mul (× rsqrt) + 1 mul (× weight) = 4
    Per-row global ops (sqrt / 1/x / eps / /N ≈ 4) 对 hidden_size 摊销 ≈ 0, 忽略.

    历史: 旧值 7 沿用 llm-viewer "sum sub pow sum div mul add" 字面计数, 重复算了
    pow + sub + div + sum 这些其实不是 RMSNorm 真实做的事. 改成 4 跟 PyTorch /
    Triton 实际 kernel 计数一致.
    """
    return OperatorProfile(
        name=name,
        op_category="norm",
        flops=tokens * hidden_size * 4,
        load_act=int(tokens * hidden_size * a_byte),
        store_act=int(tokens * hidden_size * a_byte),
        efficiency_key=dense_efficiency_key(OP_KIND_RMSNORM, tokens),
    )


def residual_add(
    name: str,
    tokens: int,
    hidden_size: int,
    a_byte: float,
) -> OperatorProfile:
    """Residual add: 1 op/element."""
    return OperatorProfile(
        name=name,
        op_category="activation",
        flops=tokens * hidden_size,
        load_act=int(tokens * hidden_size * a_byte),
        store_act=int(tokens * hidden_size * a_byte),
    )


def mlp_activation(
    name: str,
    tokens: int,
    hidden_size: int,
    a_byte: float,
) -> OperatorProfile:
    """SwiGLU activation: out = silu(x) * gate, where silu(x) = x * sigmoid(x).

    Per output-element FLOPs (5):
      1 exp(-x) + 1 add (1+exp) + 1 reciprocal (1/_) + 1 mul (x * σ) + 1 mul (× gate) = 5
    (transcendentals like exp 算 1 op 是标准约定)

    历史: 旧值 2 严重低估 (只算了 mul + 1). hidden_size 这里是 intermediate dim,
    输出元素数 = tokens × hidden_size; 输入读 2× (gate + up). 改 5 跟 vLLM SiluAndMul
    实际 op count 对齐.
    """
    return OperatorProfile(
        name=name,
        op_category="activation",
        flops=tokens * hidden_size * 5,
        load_act=int(tokens * hidden_size * a_byte * 2),
        store_act=int(tokens * hidden_size * a_byte),
        efficiency_key=dense_efficiency_key(OP_KIND_SWIGLU, tokens),
    )


def activation_quantize(
    name: str,
    tokens: int,
    hidden_size: int,
    base_a_byte: float,
    a_byte: float,
    block_size: int = 128,
    scale_byte: float = 4.0,
) -> OperatorProfile:
    """Dynamic activation quantize op: bf16 → fp8 (+ scales).

    建模 fp8 量化模型 (activation_scheme="dynamic" / "static") 的 quantize step:
      - 读 bf16 X: tokens × hidden × base_a_byte
      - 算 scale: max-reduce per group + cast (flops ~5/elem)
      - 写 fp8 X_q: tokens × hidden × a_byte
      - 写 scales: tokens × ceil(hidden/block_size) × scale_byte (fp32)

    vLLM 默认 `fuse_norm_quant=True`, RMSNorm 与此 op 合体, 省 1 次 read.
    我们不显式建模 fusion, 把 quant 独立计费, 总 bandwidth 略偏高 ~5-10%, 但比
    完全忽略 (旧版本 deploy.a_byte=1 时 norm 自身 a_byte 实际是 fp8 的"伪压缩"读)
    更接近真实。

    Args:
        base_a_byte: 输入 buffer dtype 字节 (一般 = deploy.base_a_byte = 2.0 bf16)
        a_byte: 输出 quantized dtype 字节 (= deploy.a_byte = 1.0 fp8 / 0.5 fp4)
        block_size: per-group scale block (vLLM default 128, DeepSeek-V3 = 128)
        scale_byte: scale 张量 dtype (fp32 = 4.0)
    """
    elements = tokens * hidden_size
    scale_groups = tokens * ((hidden_size + block_size - 1) // block_size)
    return OperatorProfile(
        name=name,
        op_category="activation",
        flops=elements * 5,
        load_act=int(elements * base_a_byte),
        store_act=int(elements * a_byte + scale_groups * scale_byte),
    )


def fused_add_rms_norm(
    name: str,
    tokens: int,
    hidden_size: int,
    a_byte: float,
) -> OperatorProfile:
    """Fused residual add + RMSNorm (详设 §4.7.1a (4))。

    vs naive residual_add + norm_layer:
      Naive: 3 reads + 2 writes (read x, read residual, write x;
             read x, write x_normed)
      Fused: 2 reads + 2 writes (read x + read residual,
             write x + write x_normed)
      → 省 1 次 [tokens, hidden] 的读取
    FLOPs ≈ residual_add (1/elem) + RMSNorm (7/elem) = 8/elem
    """
    return OperatorProfile(
        name=name,
        op_category="norm",
        flops=tokens * hidden_size * 8,
        load_act=int(2 * tokens * hidden_size * a_byte),
        store_act=int(2 * tokens * hidden_size * a_byte),
    )


# ---------------------------------------------------------------------------
# V4 Hyper-Connections (replaces residual add + norm)
# ---------------------------------------------------------------------------

def hc_pre(
    name: str,
    tokens: int,
    hidden_size: int,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    a_byte: float,
    w_byte: float,
) -> OperatorProfile:
    """Hyper-Connection pre-block: RMSNorm + linear mix + Sinkhorn + weighted sum.

    Includes: flatten(x) → RMSNorm → F.linear(x, hc_fn) → Sinkhorn → weighted sum.
    hc_fn shape: [mix_hc, hc_dim] where mix_hc=(2+hc_mult)*hc_mult, hc_dim=hc_mult*dim.
    """
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size

    # RMSNorm over hc_dim: 7 ops/element
    norm_flops = tokens * hc_dim * 7
    # F.linear: [tokens, hc_dim] × [mix_hc, hc_dim]^T → [tokens, mix_hc]
    linear_flops = tokens * hc_dim * mix_hc * 2
    # Sinkhorn: hc_mult iterations, each iter has exp + row_sum + col_sum + div ≈ 5 ops per element
    sinkhorn_flops = tokens * hc_sinkhorn_iters * hc_mult * hc_mult * 5
    # Weighted sum: [tokens, hc_mult] weights × [tokens, hc_mult, dim] → [tokens, dim]
    wsum_flops = tokens * hc_mult * hidden_size

    total_flops = norm_flops + linear_flops + sinkhorn_flops + wsum_flops

    # Weight IO: hc_fn [mix_hc, hc_dim] + hc_base [mix_hc] + hc_scale [3]
    # hc_fn stored in fp32 per reference model
    weight_io = int(mix_hc * hc_dim * 4 + mix_hc * 4 + 3 * 4)

    # Activation IO: read hc_mult copies of hidden state, write:
    # - y:    [tokens, hidden_size] in activation dtype
    # - post: [tokens, hc_mult] in fp32
    # - comb: [tokens, hc_mult, hc_mult] in fp32
    load_act = int(tokens * hc_mult * hidden_size * a_byte)
    store_act = int(
        tokens * hidden_size * a_byte
        + tokens * hc_mult * 4
        + tokens * hc_mult * hc_mult * 4
    )

    return OperatorProfile(
        name=name,
        op_category="norm",
        flops=total_flops,
        load_weight=weight_io,
        load_act=load_act,
        store_act=store_act,
    )


def hc_post(
    name: str,
    tokens: int,
    hidden_size: int,
    hc_mult: int,
    a_byte: float,
) -> OperatorProfile:
    """Hyper-Connection post-block: expand 1 output → hc_mult copies.

    post[b,s,hc] * x[b,s,d] → [b,s,hc,d]  (broadcast multiply)
    + comb[b,s,hc,hc] @ residual[b,s,hc,d] → [b,s,hc,d] (batched matmul)
    """
    # post * x broadcast: tokens * hc_mult * dim multiplies
    broadcast_flops = tokens * hc_mult * hidden_size
    # comb @ residual: for each of hc_mult output slots, dot product over hc_mult input slots
    comb_flops = tokens * hc_mult * hc_mult * hidden_size

    total_flops = broadcast_flops + comb_flops

    # Activation IO: read x[tokens,dim] + residual[tokens,hc_mult,dim] + post/comb from hc_pre
    # post: [tokens, hc_mult], comb: [tokens, hc_mult, hc_mult] (intermediate, on-chip likely)
    load_act = int(tokens * hidden_size * a_byte
                   + tokens * hc_mult * hidden_size * a_byte)
    store_act = int(tokens * hc_mult * hidden_size * a_byte)

    return OperatorProfile(
        name=name,
        op_category="activation",
        flops=total_flops,
        load_act=load_act,
        store_act=store_act,
    )
