"""Parameter / activation byte estimation helpers (详设 §4.3.3 阶段 4)。

阶段 4 范围:
  - estimate_param_bytes(model, w_byte): dense + GQA + MoE 参数量估算
  - estimate_activation_bytes(): 简化激活缓冲估算 (取 max_num_batched_tokens × hidden ×
    num_layers × a_byte × fudge_factor)
  - per_rank_param_bytes(model, w_byte, tp_size): TP shard 后每 rank 权重字节数

阶段 4 显式简化:
  - 不区分 RMSNorm 等小开销 (相对 dense weight 可忽略)
  - 不算 LoRA / fused 系数 (Feature gate 已拦)
  - 激活只算 hidden-state buffer, 不算 KV scratch (KV 由 num_blocks 决定, 是输出)

阶段 X calibration 触发后可换成真实 profiling 路径。
"""
from __future__ import annotations

from llm_infer_sim.core.profiles.model_config import ModelConfig


def estimate_param_count(model: ModelConfig) -> int:
    """估算模型总参数量 (单位: 个参数, 不含 byte width)。

    覆盖:
      - dense decoder (MHA / GQA): QKVO + SwiGLU FFN + 2x RMSNorm
      - MoE: routed experts (n_routed × 3 × hidden × expert_dim) + shared experts
             + router gate (hidden × n_routed, 量级小忽略)
      - embedding + lm_head (假设独立, 不 tied; Qwen3 实际 tied 但估算保守)
    """
    h = model.hidden_dim
    h_q = model.num_heads * model.head_dim          # Q 输出维 (Qwen3 ≠ h)
    h_kv = model.num_kv_heads * model.head_dim      # K/V 输出维 (GQA)

    # ---- per-layer attention ----
    attn_per_layer = (
        h * h_q                # Q proj
        + 2 * h * h_kv         # K, V proj (GQA-aware)
        + h_q * h              # O proj
    )

    # ---- per-layer FFN: dense vs MoE ----
    def _dense_ffn():
        # SwiGLU: gate + up + down 三个 GEMM
        return 3 * h * model.ffn_dim

    def _moe_ffn():
        # routed experts: n_routed × (gate + up + down) × per-expert dim
        routed = model.num_experts * 3 * h * model.expert_dim
        shared = model.num_shared_experts * 3 * h * model.expert_dim
        gate = h * model.num_experts  # router gate
        return routed + shared + gate

    # ---- 总参数: 区分 MoE / dense 层 ----
    layer_total = 0
    for layer_idx in range(model.num_layers):
        layer_total += attn_per_layer
        if model.is_moe_layer(layer_idx):
            layer_total += _moe_ffn()
        else:
            layer_total += _dense_ffn()
        # 2 × RMSNorm = 2 × h, 量级 << 其他, 但加上保守
        layer_total += 2 * h

    # ---- embedding + lm_head + final norm ----
    embed = model.vocab_size * h
    lm_head = model.vocab_size * h
    final_norm = h

    return int(layer_total + embed + lm_head + final_norm)


def estimate_param_bytes(model: ModelConfig, w_byte: float = 2.0) -> int:
    """估算模型总权重字节数 = 参数数 × byte width。"""
    return int(estimate_param_count(model) * w_byte)


def per_rank_param_bytes(model: ModelConfig, w_byte: float, tp_size: int) -> int:
    """估算 TP shard 后单 rank 的权重字节数。

    简化: 假设所有权重均匀切分到 tp_size (实际 embedding / norm 不切, 量级小忽略)。
    阶段 6 EP 启动时需要重写: routed experts 按 ep_size 切, 其他按 tp_size 切。
    """
    total_bytes = estimate_param_bytes(model, w_byte)
    if tp_size <= 1:
        return total_bytes
    return total_bytes // tp_size


def estimate_activation_bytes(
    model: ModelConfig,
    max_num_batched_tokens: int,
    a_byte: float = 2.0,
    fudge: float = 4.0,
) -> int:
    """估算激活缓冲字节数 (阶段 4 粗估)。

    formula: max_num_batched_tokens × hidden × a_byte × fudge

    fudge 反映: FA workspace + intermediate buffers + 短时峰值, 默认 4× 保守。
    阶段 X profiling 校准后替换。
    """
    return int(max_num_batched_tokens * model.hidden_dim * a_byte * fudge)
