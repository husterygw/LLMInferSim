"""MoE router 行为建模.

阶段 5-δ 公式 (V3 §4.7.1, IMPL_PLAN §4.2):
    E[distinct] = N × (1 − (1 − top_k/N)^T) × (1 − skew) + top_k × skew

边界:
    T=1, skew=0.0       → top_k          (decode 单 token, 严格成立)
    T→∞, skew=0.0       → num_experts    (全 sweep, 严格成立)
    skew=1.0            → top_k          (极端 imbalance)
    skew=0.0 (默认)     → 跟阶段 0-9 placeholder 哲学一致

MoERoutingProfile (V3 §4.7.1 distribution + alpha):
    balanced (= uniform skew=0.0)
    power_law(alpha)  — 在 signature 里区分; 转 routing skew 在 cost 端处理
"""
from __future__ import annotations

from dataclasses import dataclass


def estimate_distinct_experts(
    tokens: int,
    top_k: int,
    num_experts: int,
    skew: float = 0.0,
) -> float:
    """期望一个 step 内被命中的 distinct expert 数."""
    if tokens <= 0 or top_k <= 0 or num_experts <= 0:
        return 0.0
    top_k_eff = min(top_k, num_experts)
    p_miss = (1.0 - top_k_eff / num_experts) ** tokens
    uniform = num_experts * (1.0 - p_miss)
    worst = float(top_k_eff)
    skew_clipped = max(0.0, min(1.0, skew))
    return uniform * (1.0 - skew_clipped) + worst * skew_clipped


@dataclass(frozen=True)
class MoERoutingProfile:
    """Routing 分布 + skew. distribution / alpha 进 OperatorSignature, skew 决定公式数字."""
    distribution: str = "balanced"   # "balanced" | "power_law"
    power_law_alpha: float = 0.0
    skew: float = 0.0                 # 0=uniform, 1=极端 imbalance
    layer_skews: tuple[float, ...] | None = None

    def get_skew_for_layer(self, layer_idx: int) -> float:
        if self.layer_skews is not None and 0 <= layer_idx < len(self.layer_skews):
            return self.layer_skews[layer_idx]
        return self.skew

    @classmethod
    def balanced(cls) -> "MoERoutingProfile":
        return cls(distribution="balanced", power_law_alpha=0.0, skew=0.0)

    @classmethod
    def power_law(cls, alpha: float, skew: float = 0.0) -> "MoERoutingProfile":
        return cls(distribution="power_law", power_law_alpha=alpha, skew=skew)
