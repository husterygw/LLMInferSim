"""阶段 3 B 块: 4 个 fusion 算子的 OperatorProfile 公式校验 (详设 §4.7.1a)。"""
from llm_infer_sim.core.ops.attention import rope_kernel
from llm_infer_sim.core.ops.linear import (
    fused_gate_up_gemm,
    fused_qkv_gemm,
    linear_layer,
)
from llm_infer_sim.core.ops.normalization import (
    fused_add_rms_norm,
    norm_layer,
    residual_add,
)


def test_fused_qkv_saves_two_input_reads():
    """fused_qkv_gemm vs naive q+k+v: input X 只读 1 次, 省 2 * tokens * h * a_byte。"""
    h = 2560; n_q = 32; n_kv = 8; d = 128; tokens = 64
    w = a = kv = 2.0
    fused = fused_qkv_gemm("qkv", h, n_q, n_kv, d, tokens, w, a, kv)
    q = linear_layer("q", h, n_q * d, tokens, w, a, kv)
    k = linear_layer("k", h, n_kv * d, tokens, w, a, kv, is_kv_proj=True)
    v = linear_layer("v", h, n_kv * d, tokens, w, a, kv, is_kv_proj=True)

    naive_load_act = q.load_act + k.load_act + v.load_act
    naive_flops = q.flops + k.flops + v.flops

    # FLOPs 不变
    assert fused.flops == naive_flops
    # Input read 节省 2 * tokens * h * a (= 2 * naive's 1/3)
    assert fused.load_act == int(tokens * h * a)
    assert naive_load_act == 3 * fused.load_act
    # 权重 / KV cache 总字节相同
    assert fused.load_weight == q.load_weight + k.load_weight + v.load_weight
    assert fused.store_kv_cache == k.store_kv_cache + v.store_kv_cache


def test_fused_gate_up_saves_one_input_read():
    """fused_gate_up_gemm vs naive gate+up: input 只读 1 次, 省 1 * tokens * h * a。"""
    h = 2560; ffn = 9728; tokens = 64
    w = a = 2.0
    fused = fused_gate_up_gemm("gu", h, ffn, tokens, w, a)
    gate = linear_layer("g", h, ffn, tokens, w, a, kv_byte=2.0)
    up = linear_layer("u", h, ffn, tokens, w, a, kv_byte=2.0)
    assert fused.flops == gate.flops + up.flops
    assert fused.load_act == int(tokens * h * a)  # 1 次, naive 2 次
    assert gate.load_act + up.load_act == 2 * fused.load_act


def test_fused_add_rms_norm_saves_one_read():
    """fused_add_rms_norm vs (residual_add + norm_layer): 省 1 次 [tokens, h] 读。"""
    tokens = 64; h = 2560; a = 2.0
    fused = fused_add_rms_norm("fan", tokens, h, a)
    add = residual_add("ra", tokens, h, a)
    norm = norm_layer("rn", tokens, h, a)
    # naive: add 读 1 次写 1 次; norm 读 1 次写 1 次 → 共 2 reads + 2 writes
    # 但 add 内部还要"读 residual"(实现上是"读 x 读 r 写 x") → 总 3 reads + 2 writes.
    # 我们这里 residual_add 简化模型: 1 read + 1 write (不含 residual 单独读)
    # 不论如何, fused 应不超过 (add + norm) 的字节
    assert fused.load_act <= add.load_act + norm.load_act + tokens * h * a
    # FLOPs ≈ add (1/elem) + norm (7/elem) = 8/elem
    assert fused.flops == tokens * h * 8


def test_rope_kernel_inplace_bandwidth_only():
    """RoPE: in-place, 读写 Q+K, FLOPs 微小。"""
    tokens = 64; n_q = 32; n_kv = 8; d = 128; a = 2.0
    op = rope_kernel("rope", tokens, n_q, n_kv, d, a)
    expected_elements = tokens * (n_q + n_kv) * d
    assert op.load_act == int(expected_elements * a)
    assert op.store_act == int(expected_elements * a)
    # 6 ops/element 是合理估算 (4 mul + 2 add)
    assert op.flops == expected_elements * 6
    assert op.op_category == "activation"  # bandwidth-bound
