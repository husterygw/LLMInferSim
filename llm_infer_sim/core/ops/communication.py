"""Communication primitives: AllReduce, AllGather, AllToAll, P2P.

阶段 7 起 (跨节点修正, 详设 §4.7.3):
  - 当 n > hw.intra_node_size 且 inter_node_bandwidth > 0 时, 启用 hierarchical 2-level ring
    (intra phase + inter phase, 各自用对应带宽)
  - 当 n <= intra_node_size 或 inter_node_bandwidth = 0 时, fallback 到 flat ring 旧路径
    (阶段 0-6 baseline 保持不变)

实现策略:
  - AllReduce: hierarchical reduce-scatter + inter allreduce + intra allgather 三段
                每段用对应 alpha / beta
  - AllGather: 类似两段
  - AllToAll: 简化版 — 跨节点时整体用 inter_bw (alltoall 真实 hierarchical 公式很复杂,
              推到阶段 X §9.4.2 校准时再细化)
"""

from math import log2, ceil
from llm_infer_sim.core.profiles.hardware import HardwareConfig


def _select_algo(algo: str, data_bytes: float, n: int, allowed: list[str]) -> str:
    if algo != "auto":
        return algo
    threshold = 1 * 1024 * 1024  # 1 MB
    if data_bytes < threshold and n > 2:
        return "tree" if "tree" in allowed else allowed[0]
    return allowed[0]


def _is_cross_node(n: int, hw: HardwareConfig) -> bool:
    """是否跨节点 (n 大于单节点 GPU 数 且 inter_bw 已配)。"""
    return n > hw.intra_node_size and hw.effective_inter_bw > 0


def allreduce_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    algo: str = "auto",
) -> float:
    """AllReduce latency (seconds)。

    阶段 0-6: flat ring (alpha + data/(n*beta_intra)) × 2(n-1).
    阶段 7+:  跨节点时走 hierarchical (详 §4.7.3).
    """
    if n <= 1 or data_bytes == 0:
        return 0.0

    if _is_cross_node(n, hw):
        return _hierarchical_allreduce(data_bytes, n, hw)

    # ---- flat ring (单节点 / 阶段 0-6 baseline) ----
    algo = _select_algo(algo, data_bytes, n, ["ring", "tree", "multishot"])
    alpha = hw.link_latency
    beta = hw.effective_intra_bw
    if beta == 0:
        return float("inf")

    if algo == "ring":
        return 2 * (n - 1) * (alpha + data_bytes / (n * beta))
    elif algo == "tree":
        return 2 * ceil(log2(n)) * (alpha + data_bytes / beta)
    elif algo == "multishot":
        return 2 * (alpha + data_bytes * (n - 1) / (n * beta))
    else:
        raise ValueError(f"Unknown AllReduce algo: {algo}")


def _hierarchical_allreduce(
    data_bytes: float, n: int, hw: HardwareConfig
) -> float:
    """Hierarchical 2-level ring allreduce (详 §4.7.3 阶段 7)。

    Algorithm (NCCL "rings"):
      1. Intra reduce-scatter (within each node, N1 GPUs): N1-1 steps, data/N1 per hop
      2. Inter allreduce (across N2 nodes, on the data/N1 chunk): 2(N2-1) steps, data/(N1*N2) per hop
      3. Intra all-gather (within each node): N1-1 steps, data/N1 per hop

    Total:
      t = 2(N1-1)(α_intra + data/(N1*β_intra))
        + 2(N2-1)(α_inter + data/(N1*N2*β_inter))
    """
    n1 = hw.intra_node_size
    n2 = (n + n1 - 1) // n1
    alpha_intra = hw.link_latency
    alpha_inter = hw.inter_node_latency
    beta_intra = hw.effective_intra_bw
    beta_inter = hw.effective_inter_bw

    intra_part = 2 * (n1 - 1) * (alpha_intra + data_bytes / (n1 * beta_intra))
    inter_part = 2 * (n2 - 1) * (alpha_inter + data_bytes / (n1 * n2 * beta_inter))
    return intra_part + inter_part


def allgather_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    algo: str = "auto",
) -> float:
    """AllGather latency (seconds)。

    阶段 0-6: flat ring (n-1)(α + data/(n*β)).
    阶段 7+:  跨节点时走 hierarchical 简化 (intra 两段 + inter 一段).
    """
    if n <= 1 or data_bytes == 0:
        return 0.0

    if _is_cross_node(n, hw):
        return _hierarchical_allgather(data_bytes, n, hw)

    algo = _select_algo(algo, data_bytes, n, ["ring", "tree"])
    alpha = hw.link_latency
    beta = hw.effective_intra_bw
    if beta == 0:
        return float("inf")

    if algo == "ring":
        return (n - 1) * (alpha + data_bytes / (n * beta))
    elif algo == "tree":
        return ceil(log2(n)) * (alpha + data_bytes * (n - 1) / (n * beta))
    else:
        raise ValueError(f"Unknown AllGather algo: {algo}")


def _hierarchical_allgather(
    data_bytes: float, n: int, hw: HardwareConfig
) -> float:
    """Hierarchical 2-level ring allgather。

    Approximation (data starts as data/N per GPU, ends as full data on each):
      intra:  (N1-1) × (α_intra + (data/N2)/(N1*β_intra))  ← gathers within node
              after intra, each rank has data/N2 (its node's share)
      inter:  (N2-1) × (α_inter + (data/N2)/β_inter)        ← gathers across nodes
              after inter, each rank has full data
    """
    n1 = hw.intra_node_size
    n2 = (n + n1 - 1) // n1
    alpha_intra = hw.link_latency
    alpha_inter = hw.inter_node_latency
    beta_intra = hw.effective_intra_bw
    beta_inter = hw.effective_inter_bw

    intra_part = (n1 - 1) * (alpha_intra + (data_bytes / n2) / (n1 * beta_intra))
    inter_part = (n2 - 1) * (alpha_inter + (data_bytes / n2) / beta_inter)
    return intra_part + inter_part


def alltoall_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    algo: str = "auto",
) -> float:
    """AllToAll latency (seconds, EP expert routing).

    阶段 0-6: direct algo (n-1)(α + data/(n²*β)).
    阶段 7+: 跨节点时按"瓶颈带宽"近似 — 用 inter_bw + n=节点数² 项收口.
              真实 hierarchical alltoall 公式很复杂 (intra-AAv → inter-AAv → intra-AAv),
              阶段 X §9.4.2 校准时再细化。
    """
    if n <= 1 or data_bytes == 0:
        return 0.0

    if _is_cross_node(n, hw):
        return _hierarchical_alltoall(data_bytes, n, hw)

    algo = _select_algo(algo, data_bytes, n, ["direct", "ring"])
    alpha = hw.link_latency
    beta = hw.effective_intra_bw
    if beta == 0:
        return float("inf")

    if algo == "direct":
        return (n - 1) * (alpha + data_bytes / (n * n * beta))
    elif algo == "ring":
        return (n - 1) * (alpha + data_bytes / (n * beta))
    else:
        raise ValueError(f"Unknown AllToAll algo: {algo}")


def _hierarchical_alltoall(
    data_bytes: float, n: int, hw: HardwareConfig
) -> float:
    """Hierarchical 2-level alltoall (阶段 7 简化版)。

    Approximation (intra direct + inter direct):
      intra: (N1-1) × (α_intra + data/(N1²*β_intra))      ← shuffle within node
      inter: (N2-1) × (α_inter + (data*N1)/(N²*β_inter))  ← per-rank 对外发 data*N1/N²

    Note: 每 rank 在 alltoall 中向 N-1 个对端发 data/N 字节. 跨节点时其中
    (N1-1) 个对端在 intra, (N1*(N2-1)) 个对端在 inter. 上式按"两个独立 direct alltoall"
    的下界估计, 阶段 X 校准时若发现 NCCL 实现是 hierarchical-pairwise 再细化。
    """
    n1 = hw.intra_node_size
    n2 = (n + n1 - 1) // n1
    alpha_intra = hw.link_latency
    alpha_inter = hw.inter_node_latency
    beta_intra = hw.effective_intra_bw
    beta_inter = hw.effective_inter_bw

    intra_part = (n1 - 1) * (alpha_intra + data_bytes / (n1 * n1 * beta_intra))
    inter_part = (n2 - 1) * (
        alpha_inter + (data_bytes * n1) / (n * n * beta_inter)
    )
    return intra_part + inter_part


def p2p_time(data_bytes: float, hw: HardwareConfig) -> float:
    """Point-to-point send/recv latency (PP 用)。

    阶段 7+: 暂用 intra_bw 作为 p2p 默认带宽 (PP 通常 within node).
              跨节点 PP 推到 §10.5 7.5 子阶段 + 阶段 X 校准.
    """
    if data_bytes == 0:
        return 0.0
    alpha = hw.link_latency
    beta = hw.effective_intra_bw
    if beta == 0:
        return float("inf")
    return alpha + data_bytes / beta
