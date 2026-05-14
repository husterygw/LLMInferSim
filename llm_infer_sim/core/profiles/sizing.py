"""Parameter / activation byte estimation helpers (详设 §4.3.3 阶段 4)。

阶段 4 范围:
  - estimate_param_bytes(model, w_byte): dense + GQA + MoE 参数量估算
  - estimate_activation_bytes(): 简化激活缓冲估算
  - per_rank_param_bytes(model, w_byte, tp_size, ep_size=1): TP+EP shard 后每 rank 权重字节数

阶段 9.5 (§10.5 8.5 FP8 收尾) 起新增:
  - dtype-aware: routed expert 用 `expert_w_byte` (V4 fp4=0.5), 其他用 w_byte
  - EP-aware: routed expert 按 ep_size 切, 其他按 tp_size 切

阶段 4 显式简化:
  - 不区分 RMSNorm 等小开销 (相对 dense weight 可忽略)
  - 不算 LoRA / fused 系数 (Feature gate 已拦)
  - 激活只算 hidden-state buffer, 不算 KV scratch (KV 由 num_blocks 决定, 是输出)

阶段 X calibration 触发后可换成真实 profiling 路径。
"""
from __future__ import annotations

from llm_infer_sim.core.profiles.model_config import ModelConfig


def _count_attention_per_layer(model: ModelConfig) -> int:
    h = model.hidden_dim
    h_q = model.num_heads * model.head_dim          # Q 输出维 (Qwen3 ≠ h)
    h_kv = model.num_kv_heads * model.head_dim      # K/V 输出维 (GQA)
    # QKVO; MLA 高估一点 (q_a + q_b 比 Q proj 多, kv_a+kv_b 也比 K+V 多), 但量级一致.
    return h * h_q + 2 * h * h_kv + h_q * h


def _count_dense_ffn_per_layer(model: ModelConfig) -> int:
    # SwiGLU: gate + up + down 三个 GEMM
    return 3 * model.hidden_dim * model.ffn_dim


def _count_routed_experts_per_layer(model: ModelConfig) -> int:
    """Routed experts 参数: n_routed × (gate + up + down) × per-expert dim.
    路由 gate 参数小 (h × n_routed), 一并算进去 (跟 V3/V4 expert 主体相比 < 1%).
    """
    h = model.hidden_dim
    routed = model.num_experts * 3 * h * model.expert_dim
    gate = h * model.num_experts
    return routed + gate


def _count_shared_experts_per_layer(model: ModelConfig) -> int:
    h = model.hidden_dim
    return model.num_shared_experts * 3 * h * model.expert_dim


def estimate_param_count(model: ModelConfig) -> int:
    """估算模型总参数量 (单位: 个参数, 不含 byte width)。

    覆盖:
      - dense decoder (MHA / GQA): QKVO + SwiGLU FFN + 2x RMSNorm
      - MoE: routed experts + shared experts + router gate
      - embedding + lm_head (假设独立, 不 tied; Qwen3 实际 tied 但估算保守)
    """
    h = model.hidden_dim
    attn_per_layer = _count_attention_per_layer(model)
    dense_ffn = _count_dense_ffn_per_layer(model)
    routed = _count_routed_experts_per_layer(model)
    shared = _count_shared_experts_per_layer(model)

    layer_total = 0
    for layer_idx in range(model.num_layers):
        layer_total += attn_per_layer
        if model.is_moe_layer(layer_idx):
            layer_total += routed + shared
        else:
            layer_total += dense_ffn
        layer_total += 2 * h  # 2 × RMSNorm

    embed = model.vocab_size * h
    lm_head = model.vocab_size * h
    final_norm = h

    return int(layer_total + embed + lm_head + final_norm)


def estimate_param_bytes(
    model: ModelConfig,
    w_byte: float = 2.0,
    expert_w_byte: float | None = None,
) -> int:
    """估算模型总权重字节数, dtype-aware.

    Args:
      w_byte: 非 routed-expert 部分 dtype (attention / dense FFN / shared expert / embed / lm_head / norm)
      expert_w_byte: routed expert 专用 dtype. None 时按 model.expert_fp4 推断:
                     expert_fp4=True → 0.5 (fp4), 否则 fallback 到 w_byte.

    V4 production 真实: attention/wo/shared = fp8 (1.0), routed expert = fp4 (0.5).
    旧版本 sizing 按统一 w_byte 算导致 V4 weights/rank 高估 ~4× (`阶段 9-ε` 退出验证记录).
    """
    if expert_w_byte is None:
        expert_w_byte = 0.5 if model.expert_fp4 else w_byte
    h = model.hidden_dim
    attn_per_layer = _count_attention_per_layer(model)
    dense_ffn = _count_dense_ffn_per_layer(model)
    routed = _count_routed_experts_per_layer(model)
    shared = _count_shared_experts_per_layer(model)

    bytes_total = 0
    for layer_idx in range(model.num_layers):
        bytes_total += int(attn_per_layer * w_byte)
        if model.is_moe_layer(layer_idx):
            bytes_total += int(routed * expert_w_byte)
            bytes_total += int(shared * w_byte)
        else:
            bytes_total += int(dense_ffn * w_byte)
        bytes_total += int(2 * h * w_byte)

    embed = model.vocab_size * h
    lm_head = model.vocab_size * h
    final_norm = h
    bytes_total += int((embed + lm_head + final_norm) * w_byte)
    return bytes_total


def per_rank_param_bytes(
    model: ModelConfig,
    w_byte: float,
    tp_size: int,
    ep_size: int = 1,
    expert_w_byte: float | None = None,
) -> int:
    """估算 TP+EP shard 后单 rank 的权重字节数.

    切分规则 (vLLM FusedMoE):
      - routed experts: 按 max(tp_size, ep_size) 切
          - TP-only (ep=1): expert_dim // tp (每 rank 持全部 num_experts, 内部切 intermediate)
          - EP-enabled (ep=tp): num_experts // ep (每 rank 持 1/ep 个 expert, 内部不切)
        两种布局存储量等价, 都是 routed_total / tp.
      - 其他 (attention / dense / shared / embed / lm_head): 按 tp_size 切

    EP=1 时退化到纯 TP 切分.
    """
    if expert_w_byte is None:
        expert_w_byte = 0.5 if model.expert_fp4 else w_byte
    h = model.hidden_dim
    attn_per_layer = _count_attention_per_layer(model)
    dense_ffn = _count_dense_ffn_per_layer(model)
    routed = _count_routed_experts_per_layer(model)
    shared = _count_shared_experts_per_layer(model)
    tp = max(1, tp_size)
    ep = max(1, ep_size)
    expert_shard = max(tp, ep)

    # norm bytes 量级 << 其他 (< 0.1% 总量), 简化跟着 tp 一起切 — 保持 sizing 对 tp 线性,
    # 跟 `estimate_param_bytes / tp` 等价 (现有测试约定).
    bytes_total = 0
    for layer_idx in range(model.num_layers):
        bytes_total += int(attn_per_layer * w_byte // tp)
        if model.is_moe_layer(layer_idx):
            bytes_total += int(routed * expert_w_byte // expert_shard)
            bytes_total += int(shared * w_byte // tp)
        else:
            bytes_total += int(dense_ffn * w_byte // tp)
        bytes_total += int(2 * h * w_byte // tp)

    embed = model.vocab_size * h
    lm_head = model.vocab_size * h
    final_norm = h
    bytes_total += int((embed + lm_head + final_norm) * w_byte // tp)
    return bytes_total


def estimate_activation_bytes(
    model: ModelConfig,
    max_num_batched_tokens: int,
    a_byte: float = 2.0,
    fudge: float = 4.0,
) -> int:
    """估算激活缓冲字节数 (阶段 4 粗估).

    formula: max_num_batched_tokens × hidden × a_byte × fudge

    fudge 反映: FA workspace + intermediate buffers + 短时峰值, 默认 4× 保守.
    阶段 X profiling 校准后替换.
    """
    return int(max_num_batched_tokens * model.hidden_dim * a_byte * fudge)
