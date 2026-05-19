"""Communication primitives: AllReduce, AllGather, ReduceScatter, AllToAll,
Broadcast, P2P (intra-node + cross-node).

Phase 5 重构(详 docs/COMMUNICATION_MODELING.md):

数据语义统一:
    所有公开函数的 `data_bytes` 参数 = **per-rank input bytes**。
    比如 AllGather(data_bytes=1MB, n=4) 表示每 rank 输入 1MB shard,
    总输出 4MB; AllReduce 表示每 rank 待 reduce 的 tensor 大小。

3 层公式分解(跟计算算子对称):
    T = optional_collective_floor                            [默认 None, 不生效]
      + selected_algorithm_term(α, β_eff, n, data)            [教科书公式 + min(候选)]
      + framework_call_overhead × [mode == "eager"]           [PyTorch/ATen, graph 下消失]
      + cross_node_term if cross_node                         [跨节点 hook, 仅留路由]

NCCL 算法选择: `min(candidate × algo_bias)` 近似 NCCL 的"选最优"启发。
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from llm_infer_sim.core.profiles.hardware import HardwareConfig


CollectiveMode = Literal["eager", "cudagraph"]
TopologyHint = Literal["concentrated", "balanced"]


# ---------------------------------------------------------------------------
# Module-level defaults (跟硬件无关, 跟 PyTorch + NCCL 版本相关)
# 实测来源: RTX 4090 server, PyTorch 2.10 + NCCL 2.27 cudagraph vs eager delta.
# A100/H100 应该接近,跨硬件应该不变,见 docs/COMMUNICATION_MODELING.md §7.
# ---------------------------------------------------------------------------

DEFAULT_FRAMEWORK_CALL_OVERHEAD: dict[str, float] = {
    # 来源: scripts/measure_framework_overhead.py + measure_p2p_latency.py @ 2026-05-17.
    # 实测条件: RTX 4090 × 4 (TP=4 same NUMA), 1KB 消息(latency 主导域),
    # eager - cudagraph delta. raw 数据见 configs/calibration/raw/RTX_4090/2026-05-17/.
    # 这些值跟硬件无关, 跟 PyTorch + NCCL 版本相关.
    "allreduce":     69e-6,
    "allgather":     131e-6,
    "reducescatter": 134e-6,
    "alltoall":      119e-6,
    "broadcast":     56e-6,
    "p2p":           172e-6,
    "default":       100e-6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _framework_overhead(
    hw: HardwareConfig, collective: str, mode: CollectiveMode
) -> float:
    """Per-call framework overhead. cudagraph 模式下返 0 (跟计算 op 的 kernel_overhead 对称)."""
    if mode == "cudagraph":
        return 0.0
    table = hw.framework_call_overhead or DEFAULT_FRAMEWORK_CALL_OVERHEAD
    return table.get(collective, table.get("default", 0.0))


def _optional_collective_floor(hw: HardwareConfig, collective: str) -> float:
    """可选硬件级 floor (default None → 0). 用于 H100/B200/inter-node fallback."""
    if hw.optional_collective_floor is None:
        return 0.0
    return hw.optional_collective_floor.get(
        collective, hw.optional_collective_floor.get("default", 0.0)
    )


def _algo_bias(hw: HardwareConfig, collective: str, algo: str) -> float:
    """算法 bias multiplier. 默认 1.0 (即 min(candidate) 自然选择最快)."""
    if hw.collective_algo_bias is None:
        return 1.0
    return hw.collective_algo_bias.get(collective, {}).get(algo, 1.0)


def _select_min_candidate(
    hw: HardwareConfig,
    collective: str,
    candidates: dict[str, float],
) -> tuple[str, float]:
    """选 min(candidate × bias) 算法, 返回 (algo_name, time)."""
    best_algo: Optional[str] = None
    best_time = float("inf")
    for algo, t in candidates.items():
        adjusted = t * _algo_bias(hw, collective, algo)
        if adjusted < best_time:
            best_time = adjusted
            best_algo = algo
    assert best_algo is not None, f"no candidates for {collective}"
    return best_algo, best_time


def _is_cross_node(n: int, hw: HardwareConfig) -> bool:
    """是否跨节点 (n 大于单节点 GPU 数 且 inter_bw 已配)。"""
    return n > hw.intra_node_size and hw.effective_inter_bw > 0


# ---------------------------------------------------------------------------
# Algorithm library: per-collective candidate formulas
# 详 docs/COMMUNICATION_MODELING.md §8.
# data 约定: 各函数内部已说明的"per-rank input bytes"。
# ---------------------------------------------------------------------------

def _allreduce_ring(n: int, data: float, alpha: float, beta: float) -> float:
    """Ring AllReduce. data = per-rank input bytes.

    Reduce-scatter (n-1) hops × data/n bytes + AllGather (n-1) hops × data/n bytes
    = 2(n-1) latency + 2(n-1)/n × data/β
    """
    return 2 * (n - 1) * alpha + (2 * (n - 1) / n) * data / beta


def _allreduce_tree(n: int, data: float, alpha: float, beta: float) -> float:
    """Tree AllReduce. Reduce up + Broadcast down.

    Latency: 2 × depth, Data: 2 × data (each hop pipelines full data).
    """
    depth = math.ceil(math.log2(n))
    return 2 * depth * alpha + 2 * data / beta


def _allreduce_nvls(n: int, data: float, alpha: float, beta_aggregate: float) -> float:
    """NVSwitch SHARP / NVLS AllReduce (粗略上界).

    SHARP 网内 reduction, 2 hops (gather + scatter), β_aggregate ≈ n × β_link.
    实际 NVSwitch backplane BW 限制更细,实测校准后 enable_nvls_model=True 才用。
    """
    return 2 * alpha + data / beta_aggregate


def _broadcast_ring(n: int, data: float, alpha: float, beta: float) -> float:
    """Ring chain Broadcast. data = root broadcast bytes."""
    return (n - 1) * alpha + (n - 1) * data / beta


def _broadcast_tree(n: int, data: float, alpha: float, beta: float) -> float:
    """Tree Broadcast. depth hops, each hop transfers full data."""
    depth = math.ceil(math.log2(n))
    return depth * alpha + depth * data / beta


def _allgather_ring(n: int, data: float, alpha: float, beta: float) -> float:
    """Ring AllGather. data = per-rank input shard bytes. (n-1) hops × data."""
    return (n - 1) * alpha + (n - 1) * data / beta


def _reducescatter_ring(n: int, data: float, alpha: float, beta: float) -> float:
    """Ring ReduceScatter. data = per-rank input full tensor bytes.

    每个 rank 最终得到 data/n, 总传 factor = (n-1)/n × data.
    """
    return (n - 1) * alpha + ((n - 1) / n) * data / beta


def _alltoall_pairwise(n: int, data: float, alpha: float, beta: float) -> float:
    """Pairwise AllToAll. data = per-rank total input bytes (sent to all peers).

    (n-1) 轮, 每轮跟 1 peer 换 data/n 字节. 总数据因子 (n-1)/n × data.
    """
    return (n - 1) * alpha + ((n - 1) / n) * data / beta


def _p2p_single(data: float, alpha: float, beta: float) -> float:
    """P2P single-direction send. data = send bytes."""
    return alpha + data / beta


# ---------------------------------------------------------------------------
# Public API: top-level dispatch per collective
# ---------------------------------------------------------------------------

def allreduce_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
    cross_node: bool = False,
    algo: str = "auto",   # back-compat: 老调用可能传 algo, Phase 5 改用 min(候选)
) -> float:
    """AllReduce wall-clock time (seconds).

    Args:
        data_bytes: per-rank input tensor bytes.
        n: world size for this collective.
        mode: "eager" or "cudagraph". cudagraph 下不加 framework_call_overhead.
        topology_hint: 当 visible_devices 没给时的拓扑假设(concentrated / balanced).
        visible_devices: 实际 GPU id list (例 [0,1,4,5]); 给定时优先用来推 n_per_root.
        cross_node: 跨节点路由; Phase 5 保持现状, 走旧 hierarchical 公式.
        algo: 兼容老接口, Phase 5 内部用 min(候选) 选, 该参数仅当 != "auto" 时强制.
    """
    if n <= 1 or data_bytes <= 0:
        return 0.0

    if cross_node or _is_cross_node(n, hw):
        # 跨节点路径只算 algorithm; framework_overhead 顶层加, 跟 single-node 对称
        return (
            _hierarchical_allreduce(data_bytes, n, hw)
            + _framework_overhead(hw, "allreduce", mode)
        )

    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(
        n=n, topology_hint=topology_hint, visible_devices=visible_devices,
    )
    if beta == 0:
        return float("inf")

    # 算法候选
    candidates: dict[str, float] = {
        "ring": _allreduce_ring(n, data_bytes, alpha, beta),
        "tree": _allreduce_tree(n, data_bytes, alpha, beta),
    }
    if hw.has_nvlink_sharp and hw.enable_nvls_model:
        candidates["nvls"] = _allreduce_nvls(
            n, data_bytes, alpha, beta_aggregate=beta * n
        )

    # 老 algo 参数兼容: 强制指定算法时单选, 否则 min(候选 × bias)
    if algo != "auto" and algo in candidates:
        algo_term = candidates[algo]
    else:
        _, algo_term = _select_min_candidate(hw, "allreduce", candidates)

    return (
        _optional_collective_floor(hw, "allreduce")
        + algo_term
        + _framework_overhead(hw, "allreduce", mode)
    )


def broadcast_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
) -> float:
    """Broadcast wall-clock time (s). data_bytes = root broadcast bytes."""
    if n <= 1 or data_bytes <= 0:
        return 0.0
    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)
    if beta == 0:
        return float("inf")
    candidates: dict[str, float] = {
        "ring": _broadcast_ring(n, data_bytes, alpha, beta),
        "tree": _broadcast_tree(n, data_bytes, alpha, beta),
    }
    _, algo_term = _select_min_candidate(hw, "broadcast", candidates)
    return (
        _optional_collective_floor(hw, "broadcast")
        + algo_term
        + _framework_overhead(hw, "broadcast", mode)
    )


def allgather_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
    algo: str = "auto",
) -> float:
    """AllGather wall-clock time (s). data_bytes = per-rank input shard bytes."""
    if n <= 1 or data_bytes <= 0:
        return 0.0
    if _is_cross_node(n, hw):
        return (
            _hierarchical_allgather(data_bytes, n, hw)
            + _framework_overhead(hw, "allgather", mode)
        )
    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)
    if beta == 0:
        return float("inf")
    algo_term = _allgather_ring(n, data_bytes, alpha, beta)
    return (
        _optional_collective_floor(hw, "allgather")
        + algo_term
        + _framework_overhead(hw, "allgather", mode)
    )


def reducescatter_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
) -> float:
    """ReduceScatter wall-clock time (s). data_bytes = per-rank input full tensor bytes."""
    if n <= 1 or data_bytes <= 0:
        return 0.0
    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)
    if beta == 0:
        return float("inf")
    algo_term = _reducescatter_ring(n, data_bytes, alpha, beta)
    return (
        _optional_collective_floor(hw, "reducescatter")
        + algo_term
        + _framework_overhead(hw, "reducescatter", mode)
    )


def alltoall_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
    algo: str = "auto",
) -> float:
    """AllToAll wall-clock time (s). data_bytes = per-rank total input bytes (sent to all peers)."""
    if n <= 1 or data_bytes <= 0:
        return 0.0
    if _is_cross_node(n, hw):
        return (
            _hierarchical_alltoall(data_bytes, n, hw)
            + _framework_overhead(hw, "alltoall", mode)
        )
    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)
    if beta == 0:
        return float("inf")
    algo_term = _alltoall_pairwise(n, data_bytes, alpha, beta)
    return (
        _optional_collective_floor(hw, "alltoall")
        + algo_term
        + _framework_overhead(hw, "alltoall", mode)
    )


def p2p_time(
    data_bytes: float,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
) -> float:
    """P2P send/recv wall-clock time (s). data_bytes = send bytes."""
    if data_bytes <= 0:
        return 0.0
    alpha = hw.comm_step_latency
    # p2p 仅 1 个 sender, 在 pcie_shared_root 拓扑下 root 不被多 GPU 瓜分
    beta = hw.effective_intra_bw(
        n=1, topology_hint=topology_hint, visible_devices=visible_devices,
    )
    if beta == 0:
        return float("inf")
    algo_term = _p2p_single(data_bytes, alpha, beta)
    return (
        _optional_collective_floor(hw, "p2p")
        + algo_term
        + _framework_overhead(hw, "p2p", mode)
    )


# ---------------------------------------------------------------------------
# Hierarchical (cross-node) helpers — Phase 5 不重构, 沿用现公式 + 新 β/α 接口
# ---------------------------------------------------------------------------

def _hierarchical_allreduce(
    data_bytes: float, n: int, hw: HardwareConfig
) -> float:
    """Hierarchical 2-level ring allreduce (详 §4.7.3). Phase 5 暂不改公式主体."""
    n1 = hw.intra_node_size
    n2 = (n + n1 - 1) // n1
    alpha_intra = hw.comm_step_latency
    alpha_inter = hw.inter_node_latency
    beta_intra = hw.effective_intra_bw(n1)
    beta_inter = hw.effective_inter_bw
    intra_part = 2 * (n1 - 1) * (alpha_intra + data_bytes / (n1 * beta_intra))
    inter_part = 2 * (n2 - 1) * (alpha_inter + data_bytes / (n1 * n2 * beta_inter))
    return intra_part + inter_part


def _hierarchical_allgather(
    data_bytes: float, n: int, hw: HardwareConfig
) -> float:
    """Hierarchical 2-level ring allgather."""
    n1 = hw.intra_node_size
    n2 = (n + n1 - 1) // n1
    alpha_intra = hw.comm_step_latency
    alpha_inter = hw.inter_node_latency
    beta_intra = hw.effective_intra_bw(n1)
    beta_inter = hw.effective_inter_bw
    intra_part = (n1 - 1) * (alpha_intra + (data_bytes / n2) / (n1 * beta_intra))
    inter_part = (n2 - 1) * (alpha_inter + (data_bytes / n2) / beta_inter)
    return intra_part + inter_part


def _hierarchical_alltoall(
    data_bytes: float, n: int, hw: HardwareConfig
) -> float:
    """Hierarchical 2-level alltoall (Phase 5 不动)."""
    n1 = hw.intra_node_size
    n2 = (n + n1 - 1) // n1
    alpha_intra = hw.comm_step_latency
    alpha_inter = hw.inter_node_latency
    beta_intra = hw.effective_intra_bw(n1)
    beta_inter = hw.effective_inter_bw
    intra_part = (n1 - 1) * (alpha_intra + data_bytes / (n1 * n1 * beta_intra))
    inter_part = (n2 - 1) * (
        alpha_inter + (data_bytes * n1) / (n * n * beta_inter)
    )
    return intra_part + inter_part
