"""ModelConfig — 模型架构参数 dataclass.

来源: 复制自 llm-viewer models/parallel.py (ModelConfig 部分),
拆分时把 layer_time / model_inference_time 函数留在 cost_model/layer_builder.py。

ops/ 只放算子, profiles/ 放配置 dataclass —— 阶段 2 重构时确立的边界。
"""
from __future__ import annotations

from dataclasses import dataclass, field


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

    # V4 low-rank projections
    q_lora_rank: int = 0        # Q low-rank bottleneck (1536 in V4); 0 = direct projection
    o_lora_rank: int = 0        # O low-rank bottleneck (1024 in V4)
    o_groups: int = 0           # Grouped O projection (16 in V4); 0 = single o_proj

    # V4 sparse attention
    window_size: int = 0        # Sliding window size (128 in V4); 0 = dense attention
    compress_ratio_a: int = 0   # High compression ratio (128 in V4, for odd-index layers)
    compress_ratio_b: int = 0   # Low compression ratio (4 in V4, for even-index layers)
    first_compress_b_layer: int = 0  # First layer using ratio_b (2 in V4)
    index_topk: int = 0         # Top-k selection for ratio_b layers (1024 in V4)
    index_n_heads: int = 0      # Indexer attention heads (64 in V4)
    index_head_dim: int = 0     # Indexer head dim (128 in V4)
    no_compress_last_n: int = 0 # Last N layers with ratio=0 (1 in V4)
    rope_head_dim: int = 0      # qk_rope_head_dim (64 in V4); rope-vs-nope split
                                # for KV mixed-precision storage
    # Per-layer compression ratios (preferred over inferred a/b pattern; V4-Flash
    # has leading 0 layers that the a/b inference can't represent).
    compress_ratios: list = field(default_factory=list)

    # V4 Hyper-Connections
    hc_mult: int = 0            # HC parallel copies (4 in V4); 0 = simple residual
    hc_sinkhorn_iters: int = 0  # Sinkhorn iterations (20 in V4)

    # V4 expert quantization
    expert_fp4: bool = False    # Experts use FP4 weights (0.5 byte/param)

    def get_compress_ratio(self, layer_idx: int) -> int:
        """Return compression ratio for a given layer index.

        Prefers the explicit per-layer `compress_ratios` list (matches official
        config.json verbatim). Falls back to the (a, b, first_b) pattern only
        when the list is unavailable.
        """
        if self.window_size == 0:
            return 0
        if self.compress_ratios and layer_idx < len(self.compress_ratios):
            return int(self.compress_ratios[layer_idx])
        # Fallback: a/b alternation pattern (works for V4-Pro but NOT V4-Flash,
        # which has leading-zero SW-only layers).
        if layer_idx >= self.num_layers - self.no_compress_last_n:
            return 0
        if layer_idx < self.first_compress_b_layer:
            return self.compress_ratio_a
        offset = layer_idx - self.first_compress_b_layer
        return self.compress_ratio_b if offset % 2 == 0 else self.compress_ratio_a

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if layer_idx < self.first_moe_layer:
            return False
        return (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0
