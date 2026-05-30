"""Qwen3-4B-Instruct-2507 shape profile.

Model config (from /data1/home/ygw268/models/Qwen3-4B-Instruct-2507/config.json):
  hidden_size           = 2560
  num_attention_heads   = 32
  num_key_value_heads   = 8
  head_dim              = 128
  intermediate_size     = 9728
  num_hidden_layers     = 36
  vocab_size            = 151936
"""
from collector.profiles._dims import ProfileSpec


PROFILE = ProfileSpec(
    profile_name="qwen3_4b",
    hidden=2560,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    intermediate=9728,
    num_layers=36,
    vocab=151936,
)
