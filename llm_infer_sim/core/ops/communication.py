"""Communication primitives: AllReduce, AllGather, AllToAll, P2P.

Ported from llm_inference_eval/operators/communication.py.
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


def allreduce_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    algo: str = "auto",
) -> float:
    """AllReduce latency in seconds."""
    if n <= 1 or data_bytes == 0:
        return 0.0
    algo = _select_algo(algo, data_bytes, n, ["ring", "tree", "multishot"])
    alpha = hw.link_latency
    beta = hw.effective_comm_bandwidth
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


def allgather_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    algo: str = "auto",
) -> float:
    """AllGather latency in seconds."""
    if n <= 1 or data_bytes == 0:
        return 0.0
    algo = _select_algo(algo, data_bytes, n, ["ring", "tree"])
    alpha = hw.link_latency
    beta = hw.effective_comm_bandwidth
    if beta == 0:
        return float("inf")

    if algo == "ring":
        return (n - 1) * (alpha + data_bytes / (n * beta))
    elif algo == "tree":
        return ceil(log2(n)) * (alpha + data_bytes * (n - 1) / (n * beta))
    else:
        raise ValueError(f"Unknown AllGather algo: {algo}")


def alltoall_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    algo: str = "auto",
) -> float:
    """AllToAll latency in seconds (for MoE expert routing)."""
    if n <= 1 or data_bytes == 0:
        return 0.0
    algo = _select_algo(algo, data_bytes, n, ["direct", "ring"])
    alpha = hw.link_latency
    beta = hw.effective_comm_bandwidth
    if beta == 0:
        return float("inf")

    if algo == "direct":
        return (n - 1) * (alpha + data_bytes / (n * n * beta))
    elif algo == "ring":
        return (n - 1) * (alpha + data_bytes / (n * beta))
    else:
        raise ValueError(f"Unknown AllToAll algo: {algo}")


def p2p_time(data_bytes: float, hw: HardwareConfig) -> float:
    """Point-to-point send/recv latency (for Pipeline Parallelism)."""
    if data_bytes == 0:
        return 0.0
    alpha = hw.link_latency
    beta = hw.effective_comm_bandwidth
    if beta == 0:
        return float("inf")
    return alpha + data_bytes / beta
