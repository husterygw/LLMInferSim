"""Communication roofline helpers: AllReduce, AllGather, ReduceScatter, AllToAll,
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

comm_plan Step 4: AllReduce 统一走 NCCL allreduce 参数化候选:
ll_tree / ll128_tree / simple_ring / simple_tree (+ nvls), 不再共享全局
comm_step_latency, 也不再保留旧 ring/tree fallback.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from llm_infer_sim.core.profiles.communication import AllReduceParams
from llm_infer_sim.core.profiles.hardware import HardwareConfig


CollectiveMode = Literal["eager", "cudagraph"]
TopologyHint = Literal["concentrated", "balanced"]


DEFAULT_NCCL_ALLREDUCE_PARAMS = AllReduceParams(
    ll_tree_alpha_s=7.0e-6,
    ll_tree_max_bytes=8 * 1024,
)


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
# Step 4: AllReduce — 走 NCCL allreduce candidates
# 候选 ll_tree / ll128_tree / simple_ring / simple_tree (+ nvls).
# 详 comm_plan §7.2.
# ---------------------------------------------------------------------------

def _nccl_allreduce_params(hw: HardwareConfig) -> AllReduceParams:
    """Return NCCL AllReduce params, falling back to module defaults.

    这里的 fallback 仍然是新版 NCCL 候选模型的默认参数, 不是旧 ring/tree 公式。
    """
    if hw.communication is None:
        return DEFAULT_NCCL_ALLREDUCE_PARAMS
    backend = hw.communication.backends.get("nccl")
    if backend is None:
        return DEFAULT_NCCL_ALLREDUCE_PARAMS
    return backend.allreduce


def _fabric_bw_no_protocol(
    hw: HardwareConfig,
    n: int,
    topology_hint: TopologyHint,
    visible_devices: Optional[list],
) -> float:
    """Step 4 fabric β: 保留 topology contention, 去掉旧全局 protocol_efficiency,
    让每个候选用自己的 beta_scale 当 protocol 因子."""
    bw_with_protocol = hw.effective_intra_bw(
        n=n, topology_hint=topology_hint, visible_devices=visible_devices,
    )
    return bw_with_protocol / max(hw.intra_node_protocol_efficiency, 1e-9)


def _allreduce_candidates_nccl(
    hw: HardwareConfig,
    n: int,
    message_bytes: float,
    *,
    topology_hint: TopologyHint,
    visible_devices: Optional[list],
    protocol_hint: Optional[str] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Step 4 候选生成. 返回 (candidate_name -> latency_s, name -> bias)."""
    params = _nccl_allreduce_params(hw)
    fabric_bw = _fabric_bw_no_protocol(hw, n, topology_hint, visible_devices)
    if fabric_bw <= 0:
        return {}, {}

    depth = max(math.ceil(math.log2(n)), 1)
    candidates: dict[str, float] = {}

    if message_bytes <= params.ll_tree_max_bytes and protocol_hint in (None, "ll"):
        beta_ll = fabric_bw * params.ll_tree_beta_scale
        candidates["ll_tree"] = (
            2 * depth * params.ll_tree_alpha_s + 2 * message_bytes / beta_ll
        )

    if message_bytes <= params.ll128_tree_max_bytes and protocol_hint in (None, "ll128"):
        beta_ll128 = fabric_bw * params.ll128_tree_beta_scale
        candidates["ll128_tree"] = (
            2 * depth * params.ll128_tree_alpha_s + 2 * message_bytes / beta_ll128
        )

    if protocol_hint in (None, "simple"):
        beta_ring = fabric_bw * params.ring_beta_scale
        candidates["simple_ring"] = (
            2 * (n - 1) * params.ring_startup_alpha_s
            + (2 * (n - 1) / n) * message_bytes / beta_ring
        )

        beta_tree = fabric_bw * params.tree_beta_scale
        candidates["simple_tree"] = (
            2 * depth * params.tree_startup_alpha_s + 2 * message_bytes / beta_tree
        )

    if hw.has_nvlink_sharp and hw.enable_nvls_model:
        candidates["nvls"] = _allreduce_nvls(
            n, message_bytes,
            alpha=params.ring_startup_alpha_s,
            beta_aggregate=fabric_bw * n,
        )

    return candidates, dict(params.algorithm_bias or {})


def _select_with_bias(
    candidates: dict[str, float], biases: dict[str, float], algo: str,
) -> tuple[str, float]:
    """min(candidate × bias). 若 algo != 'auto' 强制单选."""
    if algo != "auto" and algo in candidates:
        return algo, candidates[algo]
    best_algo: Optional[str] = None
    best_time = float("inf")
    for name, t in candidates.items():
        adj = t * biases.get(name, 1.0)
        if adj < best_time:
            best_time = adj
            best_algo = name
    assert best_algo is not None, "no candidates produced"
    return best_algo, candidates[best_algo]


# ---------------------------------------------------------------------------
# Public API: top-level dispatch per collective
# ---------------------------------------------------------------------------

def allreduce_time_with_breakdown(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
    algo: str = "auto",
    protocol_hint: Optional[str] = None,
) -> tuple[float, dict[str, Any]]:
    """Step 4: AllReduce wall-clock + algorithm breakdown.

    Returns:
        (latency_s, breakdown)
        breakdown keys:
          path: "nccl"
          candidates: {name: time_s}
          selected: 选中算法名
          algorithm_term: 选中算法的 algorithm time (s, 不含 framework_overhead/floor)
          framework_overhead_s: launch overhead
    """
    if n <= 1 or data_bytes <= 0:
        return 0.0, {
            "path": "nccl",
            "candidates": {},
            "selected": None,
            "algorithm_term": 0.0,
            "framework_overhead_s": 0.0,
        }

    candidates, biases = _allreduce_candidates_nccl(
        hw, n, data_bytes,
        topology_hint=topology_hint,
        visible_devices=visible_devices,
        protocol_hint=protocol_hint,
    )
    if not candidates:
        raise ValueError(
            "NCCL AllReduce produced no candidates; check intra-node bandwidth "
            "and protocol_hint"
        )
    selected, algo_term = _select_with_bias(candidates, biases, algo)
    fo = _framework_overhead(hw, "allreduce", mode)
    return (
        _optional_collective_floor(hw, "allreduce") + algo_term + fo,
        {
            "path": "nccl",
            "candidates": candidates,
            "selected": selected,
            "algorithm_term": algo_term,
            "framework_overhead_s": fo,
        },
    )


def allreduce_time(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
    algo: str = "auto",
    protocol_hint: Optional[str] = None,
) -> float:
    """AllReduce wall-clock time (seconds). Step 4: thin wrapper over
    allreduce_time_with_breakdown(), 老调用方拿 scalar."""
    latency, _ = allreduce_time_with_breakdown(
        data_bytes, n, hw,
        mode=mode, topology_hint=topology_hint, visible_devices=visible_devices,
        algo=algo, protocol_hint=protocol_hint,
    )
    return latency


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


def _alltoall_candidates_v2(
    hw: HardwareConfig,
    n: int,
    data_bytes: float,
    *,
    topology_hint: TopologyHint,
    visible_devices: Optional[list],
) -> dict[str, float]:
    """Step 6-A AllToAll v2 候选. 详 comm_plan §8.3.

    pairwise: (n-1)*alpha + (n-1)/n * data/beta_pairwise, 再 × contention_factor
              (n>4 时 root 抢带宽).
    后续 batched_pairwise / hierarchical_alltoall 跟 calibration 同步落地.
    """
    backend = hw.communication.backends["nccl"]
    params = backend.alltoall
    fabric_bw = _fabric_bw_no_protocol(hw, n, topology_hint, visible_devices)
    if fabric_bw <= 0:
        return {}

    beta_pairwise = fabric_bw * params.pairwise_beta_scale
    base = (n - 1) * params.pairwise_alpha_s + ((n - 1) / n) * data_bytes / beta_pairwise
    contention = params.contention_factor if n > 4 else 1.0
    return {"pairwise": base * contention}


def alltoall_time_with_breakdown(
    data_bytes: float,
    n: int,
    hw: HardwareConfig,
    *,
    mode: CollectiveMode = "eager",
    topology_hint: TopologyHint = "concentrated",
    visible_devices: Optional[list] = None,
    algo: str = "auto",
) -> tuple[float, dict[str, Any]]:
    """Step 6-A: AllToAll wall-clock + algorithm breakdown.

    Returns:
        (latency_s, breakdown). breakdown keys 跟 allreduce_time_with_breakdown 对称:
        path / candidates / selected / algorithm_term / framework_overhead_s.
    """
    if n <= 1 or data_bytes <= 0:
        return 0.0, {
            "path": "v2" if hw.communication else "legacy_intra",
            "candidates": {}, "selected": None,
            "algorithm_term": 0.0, "framework_overhead_s": 0.0,
        }

    if _is_cross_node(n, hw):
        algo_term = _hierarchical_alltoall(data_bytes, n, hw)
        fo = _framework_overhead(hw, "alltoall", mode)
        return algo_term + fo, {
            "path": "legacy_hierarchical",
            "candidates": {},
            "selected": "hierarchical",
            "algorithm_term": algo_term,
            "framework_overhead_s": fo,
        }

    if (
        hw.communication is not None
        and "nccl" in hw.communication.backends
    ):
        candidates = _alltoall_candidates_v2(
            hw, n, data_bytes,
            topology_hint=topology_hint, visible_devices=visible_devices,
        )
        if candidates:
            if algo != "auto" and algo in candidates:
                selected, algo_term = algo, candidates[algo]
            else:
                selected = min(candidates, key=candidates.get)
                algo_term = candidates[selected]
            fo = _framework_overhead(hw, "alltoall", mode)
            total = (
                _optional_collective_floor(hw, "alltoall")
                + algo_term
                + fo
            )
            return total, {
                "path": "v2",
                "candidates": candidates,
                "selected": selected,
                "algorithm_term": algo_term,
                "framework_overhead_s": fo,
            }

    # Legacy fallback (hw 没配 communication)
    alpha = hw.comm_step_latency
    beta = hw.effective_intra_bw(n, topology_hint, visible_devices)
    if beta == 0:
        return float("inf"), {
            "path": "legacy_intra", "candidates": {}, "selected": None,
            "algorithm_term": float("inf"), "framework_overhead_s": 0.0,
        }
    algo_term = _alltoall_pairwise(n, data_bytes, alpha, beta)
    fo = _framework_overhead(hw, "alltoall", mode)
    return (
        _optional_collective_floor(hw, "alltoall") + algo_term + fo,
        {
            "path": "legacy_intra",
            "candidates": {"pairwise": algo_term},
            "selected": "pairwise",
            "algorithm_term": algo_term,
            "framework_overhead_s": fo,
        },
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
    """AllToAll wall-clock time (s). Step 6-A: thin wrapper over
    alltoall_time_with_breakdown(). data_bytes = per-rank total input bytes."""
    latency, _ = alltoall_time_with_breakdown(
        data_bytes, n, hw,
        mode=mode, topology_hint=topology_hint, visible_devices=visible_devices,
        algo=algo,
    )
    return latency


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
