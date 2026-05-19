"""Qwen3-30B-A3B-Instruct-2507 shape profile (MoE).

Model config:
  hidden_size              = 2048
  num_attention_heads      = 32
  num_key_value_heads      = 4
  head_dim                 = 128
  intermediate_size        = 6144  (本模型所有层都是 MoE, 不实际用)
  moe_intermediate_size    = 768
  num_hidden_layers        = 48
  num_experts              = 128
  num_experts_per_tok      = 8
  vocab_size               = 151936
"""
from collector.profiles._dims import ProfileSpec


PROFILE = ProfileSpec(
    profile_name="qwen3_30b_a3b",
    hidden=2048,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    intermediate=6144,
    num_layers=48,
    vocab=151936,
    has_moe=True,
    moe_num_experts=128,
    moe_top_k=8,
    moe_intermediate=768,
)
