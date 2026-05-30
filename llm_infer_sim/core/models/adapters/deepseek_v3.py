"""DeepSeek-V3 family adapter (model_type=deepseek_v3).

DeepSeek-V3 关键 hf_config 字段:
  hidden_size, num_attention_heads, num_key_value_heads(MLA 下 = num_heads),
  num_hidden_layers, intermediate_size(dense FFN 层用), vocab_size

MLA 字段在 profile_extractor 中独立透传(kv_lora_rank / qk_nope_head_dim /
qk_rope_head_dim / v_head_dim / q_lora_rank), 本 adapter 只负责 6 个基础 getter。

MoE 字段(n_routed_experts / num_experts_per_tok / moe_intermediate_size /
n_shared_experts / first_k_dense_replace)也由 profile_extractor 处理。

阶段 8 范围: DeepSeek-V3 FP16, FP8 推到 §10.5 8.5。
"""


def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    # MLA: num_kv_heads = num_attention_heads (每 head 都有自己的 c_kv slice).
    # 实际 KV bytes 由 kv_lora_rank 决定 (在 layer_builder MLA 分支), 不走 num_kv_heads × head_dim.
    return getattr(model_params, "num_key_value_heads", get_num_attention_heads(model_params))


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    # DeepSeek-V3 既有 dense FFN (前 first_k_dense_replace 层) 也有 MoE 层.
    # intermediate_size 用于 dense FFN, moe_intermediate_size 用于 MoE 层.
    return getattr(model_params, "intermediate_size")


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")
