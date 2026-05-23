"""ModelConfig — 模型架构参数 dataclass (框架无关, V3 §4.1).

profiles/ 放配置 dataclass; layer 顺序 / 公式由 core/models/ 模板生成.

V4 字段 (window_size / o_groups / compress_ratio_* / hc_* / num_hash_layers /
expert_fp4 / no_compress_last_n / first_compress_b_layer / compress_ratios /
o_lora_rank / get_compress_ratio()) 已在 #157 删除. V3 / V3.2 共用字段
(q_lora_rank / index_topk / index_n_heads / index_head_dim) 保留.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture parameters for parallel analysis."""

    name: str = "model"
    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    ffn_dim: int = 11008       # Dense FFN intermediate size
    num_layers: int = 32
    vocab_size: int = 32000

    # MoE
    is_moe: bool = False
    num_experts: int = 0
    num_activated_experts: int = 0  # top_k
    expert_dim: int = 0             # per-expert intermediate dim
    num_shared_experts: int = 0
    moe_layer_freq: int = 1         # every N layers is MoE (1 = all MoE)
    first_moe_layer: int = 0        # first MoE layer index

    # MLA (Multi-head Latent Attention) — set to kv_lora_rank + qk_rope_head_dim
    # If > 0, KV cache uses this dim instead of num_kv_heads * head_dim
    kv_latent_dim: int = 0
    kv_lora_rank: int = 0       # MLA: compressed KV dim (c_kv); 0 = standard MHA
    v_head_dim: int = 0         # MLA: per-head V dim (may differ from head_dim)
    qk_nope_head_dim: int = 0   # MLA: qk head dim without rope (e.g. 128 in DSV3)
    rope_head_dim: int = 0      # qk_rope_head_dim (64 in V3/V3.2)

    # Q low-rank projection (V3 / V3.2 用; V4 用过的 o_lora_rank 已删)
    q_lora_rank: int = 0        # Q low-rank bottleneck (1536 in V3); 0 = direct projection

    # V3.2 lightning indexer
    index_topk: int = 0         # Top-k context positions (2048 in V3.2); 0 = no indexer
    index_n_heads: int = 0      # Indexer attention heads (64 in V3.2)
    index_head_dim: int = 0     # Indexer head dim (128 in V3.2)

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if layer_idx < self.first_moe_layer:
            return False
        return (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0
