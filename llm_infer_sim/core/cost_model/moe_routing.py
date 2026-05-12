"""MoE router 行为建模 (阶段 5-δ, 详设 §4.7.1)。

阶段 5 默认走 Tier 1 + Tier 2:
  - Tier 1 (closed-form): coupon collector 给"uniform router 下 distinct expert 期望数"
  - Tier 2 (parameterizable skew): 在 uniform 与极端 imbalance 之间线性插值

  E[distinct] = N × (1 − (1 − top_k/N)^T) × (1 − skew) + top_k × skew

边界正确性:
  T=1, skew=0.0       → top_k         (decode 单 token, 严格成立)
  T→∞, skew=0.0       → num_experts   (全 sweep, 严格成立)
  skew=1.0 (任意 T)   → top_k         (永远只有 hot expert, 极端 imbalance)
  skew=0.0 (默认)     → 跟阶段 0-9 placeholder 哲学一致 (不在 cost path 上放未校准折扣)

详设引用:
  §4.7.1   ModelCoreCostModel MoE 层 token-per-expert uniform 估算
  §10 阶段 5 / 阶段 6 (EP 时 skew 真实影响 per-rank 负载)
  §10 阶段 X §9.4.2 (per-layer skew + capacity 模型, 需 microbench 数据)
"""
from __future__ import annotations

from dataclasses import dataclass


def estimate_distinct_experts(
    tokens: int,
    top_k: int,
    num_experts: int,
    skew: float = 0.0,
) -> float:
    """期望一个 step 内被命中的 distinct expert 数。

    公式 = uniform × (1 − skew) + top_k × skew
      uniform = num_experts × (1 − (1 − top_k/num_experts) ** tokens)
      top_k   = worst-case (永远只 hit top_k 个 hot expert)

    Args:
        tokens: 该 step 内进入 MoE 层的 token 总数。
        top_k: 每 token 激活的 expert 数 (Qwen3-30B-A3B=8, DeepSeek-V3=8)。
        num_experts: 总 expert 数 (Qwen3-30B-A3B=128, DeepSeek-V3=256)。
        skew: 路由偏度 ∈ [0, 1]。0=完美 uniform, 1=极端 imbalance。
              阶段 0-9 默认 0 (与 EfficiencyProfile.placeholder=1.0 同哲学)。

    Returns:
        期望 distinct expert 数 (float, ∈ [0, min(top_k×tokens, num_experts)])。
    """
    if tokens <= 0 or top_k <= 0 or num_experts <= 0:
        return 0.0
    top_k_eff = min(top_k, num_experts)
    p_miss = (1.0 - top_k_eff / num_experts) ** tokens
    uniform = num_experts * (1.0 - p_miss)
    worst = float(top_k_eff)
    skew_clipped = max(0.0, min(1.0, skew))
    return uniform * (1.0 - skew_clipped) + worst * skew_clipped


@dataclass
class MoERoutingPolicy:
    """MoE router 行为建模参数 (详设 §4.7.1, 阶段 5-δ)。

    阶段 0-9 默认 skew=0.0 (与 EfficiencyProfile.placeholder=1.0 哲学一致),
    用户可在 standalone 模式 / 测试中覆盖做敏感性分析。

    阶段 X §9.4.2 起 layer_skews 由 microbench 数据填,取代单一 scalar。
    """
    # Tier 2: 全局 skew (uniform 与极端 imbalance 之间的线性插值)
    skew: float = 0.0
    # Tier 3 hook (阶段 X 起填): 每层独立 skew, None 表示 fallback 到全局 skew
    layer_skews: list[float] | None = None
    # Tier 4 hook (阶段 X 起接 trace replay): 暂未启用
    use_trace: bool = False

    def get_skew_for_layer(self, layer_idx: int) -> float:
        """返回该层使用的 skew (有 layer_skews 时优先)。"""
        if self.layer_skews is not None and 0 <= layer_idx < len(self.layer_skews):
            return self.layer_skews[layer_idx]
        return self.skew
