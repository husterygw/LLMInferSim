"""DeepSeek-V4 family adapter (model_type=deepseek_v4).

V4-Flash 关键 hf_config 字段 (跟 V3 不同的部分):
  - 全 MoE, 无 intermediate_size 字段 → fallback 到 0
  - 无 kv_lora_rank / qk_nope_head_dim (这两个是 MLA 字段, V4 走 sparse attention path)
  - 有 head_dim=512 (显式), qk_rope_head_dim=64, qk_nope = head_dim - rope = 448

V4 sparse attention / HC / o_proj LoRA 等字段在 profile_extractor 中独立透传, 本 adapter
只负责 6 个基础 getter.
"""


def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    # V4: num_key_value_heads=1 (MQA-style after compressor; KV is single-head latent)
    return getattr(model_params, "num_key_value_heads", get_num_attention_heads(model_params))


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    # V4 全 MoE, 没有 dense FFN intermediate_size 字段; 返回 0 (layer_builder MoE path
    # 用 moe_intermediate_size, 不读 ffn_dim)
    return getattr(model_params, "intermediate_size", 0) or 0


def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")
