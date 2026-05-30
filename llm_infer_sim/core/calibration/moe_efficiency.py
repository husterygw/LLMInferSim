"""MoE calibration knobs — moe_plan §5.A.

跟 CommunicationProfile 模式一致, 独立 dataclass 挂在 HardwareConfig 上.

设计原则:
  - 首版只允许 3 类 knob (plan §4 Phase 5.A step 2):
      topk_overhead_us         - moe_topk op 的 fixed kernel overhead (us)
      local_dispatch_overhead_us - MoEDispatch pre/post local kernel overhead (us)
      grouped_gemm_efficiency  - MoE routed_experts compute multiplier (<1.0 = 实际比 roofline 慢)
  - moe_profile_id 用于 trace 调试, 标记当前用的是哪个 calibration set
  - 每个 knob 必须在 CostTraceEntry.metadata 可见 (plan §5.A step 3)
  - 修改 knob 只允许影响对应的 latency term (plan §5.A step 4 验收)
  - bucket 化 (per tokens / phase) 留 Phase 5.B 数据驱动后再加
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEEfficiencyProfile:
    """MoE 校准参数. moe_plan §5.A.

    profile_id: 当前 calibration set 标识 (trace metadata 用)
    topk_overhead_us: moe_topk op 加性 latency (us, after roofline)
    local_dispatch_overhead_us: MoEDispatch (pre / post) 加性 latency (us)
    grouped_gemm_efficiency: MoE.routed_experts compute term 乘数
                             实际 latency = roofline_latency / grouped_gemm_efficiency
                             1.0 = no calibration; <1.0 = sim 偏快, 调慢
    """
    profile_id: str = "default"
    topk_overhead_us: float = 0.0
    local_dispatch_overhead_us: float = 0.0
    grouped_gemm_efficiency: float = 1.0


def default_moe_efficiency() -> MoEEfficiencyProfile:
    """无 calibration placeholder (knob 全 0/1).

    用于非 RTX_4090 等还没采校准数据的 hw, 或者 unit test fixture.
    """
    return MoEEfficiencyProfile(
        profile_id="default_uncalibrated",
        topk_overhead_us=0.0,
        local_dispatch_overhead_us=0.0,
        grouped_gemm_efficiency=1.0,
    )


def rtx_4090_moe_efficiency_v1() -> MoEEfficiencyProfile:
    """RTX 4090 MoE calibration v1 — Phase 5.B 探查后回到中性, 仅保留 profile_id 标记.

    探查过程 (2026-05-25):
      初版尝试 grouped_gemm_efficiency=0.93 + topk/dispatch overhead 3us 让 sim 更慢,
      预期能 fit moe_ep_sweep TPOT +24% gap. 实测 driver case
      i2048_o128 tp4 ep4 TPOT 反而 +24.27% → +32.36%, 其它 EP case 全部退化 10%+.

    根因分析:
      - moe_ep_sweep TPOT gap "+24%" 实际方向是 sim 偏慢 (sim > real). plan §2 修正后已说明.
      - Phase 5 standalone MoE single-op measure 显示 sim 偏快 7.5% (gap_mean=1.075).
      - 两者方向相反 ⇒ +24% gap 主要来源 **不是 MoE compute**, 是 attention / scheduler /
        comm 等 step 其它部分 sim 估值偏慢.
      - MoEEfficiencyProfile 仅 knob MoE compute / dispatch / topk, 不覆盖 attention /
        scheduler / async pipeline 这条 root cause path.

    决策: revert 到中性值 (knob 全 0/1), profile_id 标 "phase5b_neutral" 表示
    已探查; 真正 fit +24% gap 留 follow-up (Phase 6 / dense attention recalibration).
    """
    return MoEEfficiencyProfile(
        profile_id="rtx_4090_phase5b_neutral",
        topk_overhead_us=0.0,
        local_dispatch_overhead_us=0.0,
        grouped_gemm_efficiency=1.0,
    )
