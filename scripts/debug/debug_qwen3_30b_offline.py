#!/usr/bin/env python3
"""离线调试 Qwen3-30B-A3B (MoE) 的 cost 估算 —— 不起 vLLM / 不用 GPU, 瞬间跑完.

走和生产完全一样的路径: build_roofline_engine → registry.get_model →
Qwen3MoeModel(build-once) → forward(step) → CostRouter → StepCostTrace.

用法:
    python scripts/debug_qwen3_30b_offline.py                 # prefill isl=128 + decode bs=8
    python scripts/debug_qwen3_30b_offline.py --isl 2048
    python scripts/debug_qwen3_30b_offline.py --tp 4 --ep 4
    python scripts/debug_qwen3_30b_offline.py --decode-n 32 --decode-ctx 4096
    python scripts/debug_qwen3_30b_offline.py --hw H100_SXM

在任意 op 上加断点调试: 直接在此脚本里 import pdb, 或在
Qwen3MoeModel.forward / 各 op 的 forward()/roofline_spec() 里下断点.
"""
from __future__ import annotations

import argparse

from llm_infer_sim.core.cost.engine import build_roofline_engine
from llm_infer_sim.core.metrics.breakdown import format_step_breakdown
from llm_infer_sim.core.operators import MoERoutingProfile
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)


def qwen3_30b_a3b() -> ModelConfig:
    """Qwen3-30B-A3B 离线 config (跟 tests/integration/core/test_moe_cost_consistency 一致)."""
    return ModelConfig(
        name="Qwen3-30B-A3B",
        hidden_dim=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        ffn_dim=0, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8,
        expert_dim=768, num_shared_experts=0,
        moe_layer_freq=1, first_moe_layer=0,
    )


def prefill_workload(isl: int) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(
            request_id="r0", phase=StepPhase.PREFILL,
            num_tokens=isl, context_len=0,
        )],
        num_prefill_tokens=isl, total_scheduled_tokens=isl,
        num_prefill_requests=1,
    )


def decode_workload(n: int, ctx: int) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(
            request_id=f"d{i}", phase=StepPhase.DECODE,
            num_tokens=1, context_len=ctx,
        ) for i in range(n)],
        num_decode_tokens=n, total_scheduled_tokens=n,
        num_decode_requests=n,
    )


def dump_ops(trace) -> None:
    """逐 op 表: count / latency / bottleneck / flops / mem_bytes (debug 用)."""
    hdr = f"{'op':24s} {'kind':12s} {'subtype':16s} {'cnt':>4s} {'lat_us':>10s} {'btl':>8s} {'GFLOP':>10s} {'MB':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for e in trace.entries:
        m = e.metadata
        cnt = m.get("count", 1)
        gflop = m.get("flops", 0) / 1e9
        mb = m.get("mem_bytes", 0) / 1e6
        print(f"{e.op_name:24s} {e.op_kind:12s} {e.op_subtype:16s} {cnt:>4} "
              f"{e.latency_s * 1e6:>10.2f} {m.get('bottleneck', '-'):>8s} "
              f"{gflop:>10.3f} {mb:>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isl", type=int, default=128, help="prefill input seq len")
    ap.add_argument("--decode-n", type=int, default=8, help="decode batch size")
    ap.add_argument("--decode-ctx", type=int, default=1024, help="decode context len")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--hw", default="RTX_4090", help="hardware profile (本机=RTX_4090)")
    ap.add_argument("--skew", type=float, default=0.0, help="MoE routing skew [0,1]")
    args = ap.parse_args()

    model = qwen3_30b_a3b()
    deployment = DeploymentProfile.flat(tp=args.tp, ep=args.ep)
    runtime = RuntimeProfile.flat()
    hw = get_hardware_profile(args.hw)
    routing = MoERoutingProfile(distribution="balanced", skew=args.skew)

    engine = build_roofline_engine(model, deployment, runtime, hw, routing=routing)
    print(f"model={model.name}  graph={type(engine.model).__name__}  "
          f"tp={args.tp} ep={args.ep} hw={args.hw} skew={args.skew}\n")

    for label, wl in (
        (f"PREFILL isl={args.isl}", prefill_workload(args.isl)),
        (f"DECODE  bs={args.decode_n} ctx={args.decode_ctx}",
         decode_workload(args.decode_n, args.decode_ctx)),
    ):
        trace = engine.estimate(wl)
        print(f"===== {label} =====")
        print(format_step_breakdown(trace))
        print()
        dump_ops(trace)
        print(f"\ntotal_latency={trace.total_latency_s * 1e6:.2f}us  "
              f"bottleneck={trace.bottleneck}  "
              f"(compute={trace.compute_time_s * 1e6:.1f} "
              f"mem={trace.memory_time_s * 1e6:.1f} "
              f"comm={trace.comm_time_s * 1e6:.1f}us)\n")


if __name__ == "__main__":
    main()
