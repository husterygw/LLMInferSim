"""ProfileSpec — framework-agnostic, model 维度可识别的 shape 描述.

只描述"测什么 shape", 不绑定 OperatorDB 的查询键. 同一组 shape 在不同 profile
里只要一致, 会被去重.

字段直接对应 model card config.json (HF format), 加少量 MoE 字段.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSpec:
    """模型 shape 描述. profile_name 仅作 provenance, 不进 case_id."""
    profile_name: str           # e.g. "qwen3_4b" — 仅 metadata, 不进 case hash
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int           # dense FFN intermediate size; MoE-only 模型设 0
    num_layers: int
    vocab: int

    # MoE 字段 (非 MoE 模型这些都 0)
    has_moe: bool = False
    moe_num_experts: int = 0
    moe_top_k: int = 0
    moe_intermediate: int = 0

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_out(self) -> int:
        """fused qkv_proj 输出维 = q + 2 * kv (Q + K + V)."""
        return self.q_dim + 2 * self.kv_dim
