"""Communication parameter profiles.

按 fabric (link) / backend / collective 三层组织通信参数, 替代
HardwareConfig.comm_step_latency 这种"万能 alpha"模式.

设计意图 (详 comm_plan.md §6):
  - p2p / allreduce / allgather / reducescatter / alltoall 参数互相独立
  - 每个 collective 按 algorithm/protocol 分子结构 (e.g., ll_tree / simple_ring)
  - fabric 层描述物理链路 (PCIe/NVLink/IB), backend 层 (nccl/pynccl/...) 在 fabric
    上跑自己的协议参数
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Fabric layer — physical link descriptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkProfile:
    """Single physical link (NVLink / PCIe / IB) — bandwidth + startup latency."""
    bandwidth_Bps: float          # one-direction nominal bandwidth, bytes/s
    startup_alpha_s: float        # link-level setup latency (P2P floor, seconds)
    topology: str = ""            # "pcie_shared_root" / "nvlink_full" / "ib_full"


@dataclass(frozen=True)
class FabricProfile:
    """Set of intra-node / inter-node links available on this hardware."""
    intra_node_links: dict[str, LinkProfile] = field(default_factory=dict)
    inter_node_links: dict[str, LinkProfile] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-collective params
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class P2PParams:
    """Point-to-point send/recv."""
    startup_alpha_s: float
    beta_scale: float = 1.0   # multiplier on link bandwidth (1.0 = nominal)


@dataclass(frozen=True)
class AllReduceParams:
    """AllReduce algorithm candidates per backend.

    候选算法 (Step 4 选 min):
      ll_tree       — small-msg LL protocol + tree topology
      ll128_tree    — mid-msg LL128 protocol + tree (待数据校准, 暂用 plan 默认)
      simple_ring   — large-msg simple protocol + ring (BW-optimized)
      simple_tree   — large-msg simple protocol + double-binary tree (fallback)
    """
    # LL (small-msg latency-optimized)
    ll_tree_alpha_s: float
    ll_tree_max_bytes: int
    # LL 协议带宽利用率低 (~50% of LL128); 但小消息区 alpha 主导, beta 影响 <5%.
    # 待大消息阶段独测细化.
    ll_tree_beta_scale: float = 0.4

    # LL128 (mid-msg). 默认值来自 plan, 待 8K-64K 区域细分数据校准.
    ll128_tree_alpha_s: float = 9.0e-6
    ll128_tree_max_bytes: int = 64 * 1024
    ll128_tree_beta_scale: float = 0.65

    # Simple ring (large-msg BW-optimized).
    # ring_beta_scale 0.625: Step 0 sweep (>8KB) best eff=0.625, log_rms=0.322;
    # docs/baselines/allreduce_roofline_gap CSV n=4 cudagraph 10-40MB gap ±2.5%
    # 确认 0.625 对大消息 standalone AR 已贴脸. plan §6 的 0.45 来自 ALL 区间
    # 联合 fit 平均, 大消息独立看时偏激.
    ring_startup_alpha_s: float = 9.6e-6
    ring_beta_scale: float = 0.625

    # Simple tree (large-msg fallback, e.g., double-binary tree).
    tree_startup_alpha_s: float = 9.6e-6
    tree_beta_scale: float = 0.625

    # Algorithm bias multiplier when selecting min(candidate × bias).
    # 默认空 = 纯 min. NCCL_ALGO override 时可注入。
    algorithm_bias: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AllGatherParams:
    """AllGather; Step 6 (后续 collective 展开) 细化."""
    ll_alpha_s: float = 7.0e-6
    ll_max_bytes: int = 8 * 1024
    ring_startup_alpha_s: float = 9.6e-6
    ring_beta_scale: float = 0.45


@dataclass(frozen=True)
class ReduceScatterParams:
    """ReduceScatter; Step 6 细化."""
    ll_alpha_s: float = 7.0e-6
    ll_max_bytes: int = 8 * 1024
    ring_startup_alpha_s: float = 9.6e-6
    ring_beta_scale: float = 0.45


@dataclass(frozen=True)
class AllToAllParams:
    """AllToAll; MoE EP 重点校准, Step 6 细化."""
    pairwise_alpha_s: float = 9.6e-6
    pairwise_beta_scale: float = 0.45
    # contention penalty: pairwise 算法下 n>4 时 root 抢带宽的额外 latency factor.
    # 1.0 = 无额外, 默认保守 1.2 (待 EP 数据校准).
    contention_factor: float = 1.2


# ---------------------------------------------------------------------------
# Backend layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackendCommunicationProfile:
    """单 backend (nccl / pynccl / custom_all_reduce) 的所有 collective 参数."""
    p2p: P2PParams
    allreduce: AllReduceParams
    allgather: AllGatherParams = field(default_factory=AllGatherParams)
    reducescatter: ReduceScatterParams = field(default_factory=ReduceScatterParams)
    alltoall: AllToAllParams = field(default_factory=AllToAllParams)


@dataclass(frozen=True)
class CommunicationProfile:
    """Top-level: fabric + backend 集合.

    用法:
        hw.communication.backends["nccl"].allreduce.ll_tree_alpha_s
        hw.communication.backends["nccl"].alltoall.pairwise_alpha_s
        hw.communication.backends["nccl"].p2p.startup_alpha_s
    """
    fabric: FabricProfile = field(default_factory=FabricProfile)
    backends: dict[str, BackendCommunicationProfile] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-built defaults
# ---------------------------------------------------------------------------

def rtx_4090_nccl_communication() -> CommunicationProfile:
    """RTX 4090 + NCCL backend 首版默认值.

    数据来源:
      - p2p.startup_alpha_s = 9.6us: scripts/measure_p2p_latency.py 2026-05-17 实测
        (GPU 0→1 cudagraph round-trip 19.3us / 2 hops)
      - allreduce.ll_tree_alpha_s = 7.0us: 2026-05-23 floor sweep,
        n=2 (7.5us) / n=4 (13us, ≈ 2×7) / n=8 (21us, ≈ 3×7) 符合 log2(n) × 7us
      - allreduce.ll_tree_max_bytes = 8KB: Step 0 sweep 显示 BW kick-in 在 ~5-8KB 区
      - allreduce.ring_beta_scale = 0.625: Step 0 sweep LARGE (>8KB) best eff=0.625
        (log_rms=0.322) + docs/baselines CSV n=4 cudagraph 10-40MB gap ±2.5%
        证实 0.625 跟 standalone AR 贴脸. 早期 plan §6 写 0.45 是 ALL 区间联合 fit,
        在 Step 5 全链路 bench 中导致 TTFT 退化 +27%, 故回归到 0.625.
      - 其它 (ll128 / allgather / alltoall): 待数据校准, 用 plan 默认
    """
    pcie_link = LinkProfile(
        bandwidth_Bps=32e9,         # PCIe 4.0 x16 单向 nominal
        startup_alpha_s=9.6e-6,
        topology="pcie_shared_root",
    )
    return CommunicationProfile(
        fabric=FabricProfile(
            intra_node_links={"pcie_4_x16": pcie_link},
            inter_node_links={},        # 单卡机, 不跨节点
        ),
        backends={
            "nccl": BackendCommunicationProfile(
                p2p=P2PParams(
                    startup_alpha_s=9.6e-6,
                    beta_scale=1.0,
                ),
                allreduce=AllReduceParams(
                    ll_tree_alpha_s=7.0e-6,
                    ll_tree_max_bytes=8 * 1024,
                    ll_tree_beta_scale=0.4,
                    ll128_tree_alpha_s=9.0e-6,
                    ll128_tree_max_bytes=64 * 1024,
                    ll128_tree_beta_scale=0.65,
                    ring_startup_alpha_s=9.6e-6,
                    ring_beta_scale=0.625,
                    tree_startup_alpha_s=9.6e-6,
                    tree_beta_scale=0.625,
                ),
            ),
        },
    )
