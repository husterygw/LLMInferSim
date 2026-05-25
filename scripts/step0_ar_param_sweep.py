"""Step 0: AllReduce parameter sweep on existing formula.

不改代码,只 sweep (comm_step_latency, intra_node_protocol_efficiency, eager_extra)
找最贴 measured 的组合. 输出:
  (a) 全局最佳单组参数
  (b) 按 size 分段最佳 (small ≤8KB / large >8KB) - 看是否需要 ll_tree vs simple_ring 分裂
  (c) 当前默认值 vs 最佳的 gap 改善幅度

跑法:
    python scripts/step0_ar_param_sweep.py \
        --records /tmp/collector_ar_full/operator_db/RTX_4090/vllm-0.19.1/collective.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from llm_infer_sim.core.cost.roofline.communication import allreduce_time
from llm_infer_sim.core.profiles.hardware import get_hardware_profile


def load_ar_records(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("op_kind") != "collective":
            continue
        if r.get("params", {}).get("op_subtype") != "allreduce":
            continue
        out.append(r)
    return out


def sim_us(record: dict, hw, eager_extra_us: float) -> float:
    p = record["params"]
    t = allreduce_time(
        data_bytes=float(p["message_size_bytes"]),
        n=int(p["num_gpus"]),
        hw=hw,
        mode=p["execution_mode"],
        topology_hint=p["topology_hint"],
    ) * 1e6
    if p["execution_mode"] == "eager":
        t += eager_extra_us
    return t


def gap_metric(records: list[dict], hw, eager_extra_us: float) -> dict:
    """log-gap RMS as primary, plus median/p90/max for context."""
    log_gaps = []
    gaps = []
    for r in records:
        s = sim_us(r, hw, eager_extra_us)
        m = float(r["metrics"]["latency_us_p50"])
        if s <= 0:
            continue
        g = m / s
        gaps.append(g)
        log_gaps.append(math.log(g))
    if not gaps:
        return {}
    gaps_sorted = sorted(gaps)
    return {
        "n": len(gaps),
        "log_rms": math.sqrt(sum(x * x for x in log_gaps) / len(log_gaps)),
        "median": statistics.median(gaps),
        "p10": gaps_sorted[max(0, int(0.10 * len(gaps_sorted)))],
        "p90": gaps_sorted[min(len(gaps_sorted) - 1, int(0.90 * len(gaps_sorted)))],
        "max_dev": max(max(g, 1 / g) for g in gaps),
    }


def sweep(records: list[dict], alpha_grid, eff_grid, eager_grid, base_hw):
    """grid sweep, 返 list of (params, stats)."""
    out = []
    for alpha_us, eff, eager_us in itertools.product(alpha_grid, eff_grid, eager_grid):
        hw = dataclasses.replace(
            base_hw,
            comm_step_latency=alpha_us * 1e-6,
            intra_node_protocol_efficiency=eff,
        )
        stats = gap_metric(records, hw, eager_us)
        out.append({
            "alpha_us": alpha_us,
            "protocol_eff": eff,
            "eager_extra_us": eager_us,
            **stats,
        })
    return out


def fmt_combo(c):
    return (f"α={c['alpha_us']:.1f}us eff={c['protocol_eff']:.3f} "
            f"eager+{c['eager_extra_us']:.0f}us")


def print_topk(results, top_k=10, label=""):
    results = sorted(results, key=lambda r: r["log_rms"])
    print(f"\n=== {label} top-{top_k} by log_rms ===")
    print(f"{'rank':>4} {'combo':<42} {'log_rms':>8} {'median':>7} "
          f"{'p10':>6} {'p90':>6} {'max_dev':>8}")
    for i, r in enumerate(results[:top_k], 1):
        print(f"{i:>4} {fmt_combo(r):<42} {r['log_rms']:>8.3f} "
              f"{r['median']:>7.3f} {r['p10']:>6.3f} {r['p90']:>6.3f} "
              f"{r['max_dev']:>8.2f}")


def split_by_size(records, threshold_bytes):
    small = [r for r in records if r["params"]["message_size_bytes"] <= threshold_bytes]
    large = [r for r in records if r["params"]["message_size_bytes"] > threshold_bytes]
    return small, large


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--hardware", default="RTX_4090")
    args = ap.parse_args()

    records = load_ar_records(Path(args.records))
    print(f"loaded {len(records)} AR records")
    base_hw = get_hardware_profile(args.hardware)
    print(f"baseline hw: comm_step_latency={base_hw.comm_step_latency*1e6:.1f}us "
          f"intra_node_protocol_efficiency={base_hw.intra_node_protocol_efficiency:.3f}")

    # Baseline (current defaults, eager_extra=0)
    base_stats = gap_metric(records, base_hw, 0.0)
    print(f"\nBASELINE gap stats: log_rms={base_stats['log_rms']:.3f} "
          f"median={base_stats['median']:.3f} p10={base_stats['p10']:.3f} "
          f"p90={base_stats['p90']:.3f} max_dev={base_stats['max_dev']:.2f}")

    # Grids
    alpha_grid = [3.0, 3.5, 4.0, 5.0, 7.0, 9.6]
    eff_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.625]
    eager_grid = [0, 20.0, 25.0, 30.0]

    print(f"\nsweep grid: alpha={alpha_grid} eff={eff_grid} eager_extra={eager_grid}")
    print(f"  total combos: {len(alpha_grid)*len(eff_grid)*len(eager_grid)}")

    # (a) 全部数据 best
    print("\n" + "=" * 70)
    print("(a) ALL records sweep")
    print("=" * 70)
    all_results = sweep(records, alpha_grid, eff_grid, eager_grid, base_hw)
    print_topk(all_results, top_k=10, label="ALL data")

    # (b) split by 8KB
    print("\n" + "=" * 70)
    print("(b) SPLIT @ 8KB — small (≤8KB) vs large (>8KB) sweep")
    print("=" * 70)
    small, large = split_by_size(records, 8 * 1024)
    print(f"split: small={len(small)} large={len(large)}")
    small_results = sweep(small, alpha_grid, eff_grid, eager_grid, base_hw)
    large_results = sweep(large, alpha_grid, eff_grid, eager_grid, base_hw)
    print_topk(small_results, top_k=5, label="SMALL (≤8KB)")
    print_topk(large_results, top_k=5, label="LARGE (>8KB)")

    # (c) try thresholds 4KB / 16KB to compare
    print("\n" + "=" * 70)
    print("(c) split threshold sensitivity (best single combo per side)")
    print("=" * 70)
    for thresh_kb in (4, 8, 16, 32):
        s, l = split_by_size(records, thresh_kb * 1024)
        if not s or not l:
            print(f"\n  threshold {thresh_kb}KB: skip (small={len(s)} large={len(l)})")
            continue
        ss = sweep(s, alpha_grid, eff_grid, eager_grid, base_hw)
        ls = sweep(l, alpha_grid, eff_grid, eager_grid, base_hw)
        sb = min(ss, key=lambda r: r["log_rms"])
        lb = min(ls, key=lambda r: r["log_rms"])
        # Combined log_rms: weighted avg by sample count
        combined_rms = math.sqrt(
            (sb["log_rms"]**2 * sb["n"] + lb["log_rms"]**2 * lb["n"]) / (sb["n"] + lb["n"])
        )
        print(f"\n  threshold {thresh_kb}KB:")
        print(f"    small (n={sb['n']}): {fmt_combo(sb):<42} log_rms={sb['log_rms']:.3f}")
        print(f"    large (n={lb['n']}): {fmt_combo(lb):<42} log_rms={lb['log_rms']:.3f}")
        print(f"    combined log_rms = {combined_rms:.3f}")

    # (d) best single-combo at default eff=0.625 vs swept (改 alpha 单参数能多准)
    print("\n" + "=" * 70)
    print("(d) Single-knob sweep — change only alpha, hold eff=0.625, eager=25us")
    print("=" * 70)
    alpha_only = []
    for alpha_us in [3.0, 3.5, 4.0, 5.0, 7.0, 9.6, 12.0]:
        hw = dataclasses.replace(base_hw, comm_step_latency=alpha_us * 1e-6)
        s = gap_metric(records, hw, 25.0)
        s["alpha_us"] = alpha_us
        s["protocol_eff"] = 0.625
        s["eager_extra_us"] = 25.0
        alpha_only.append(s)
    print_topk(alpha_only, top_k=7, label="alpha-only (eff=0.625 fixed)")


if __name__ == "__main__":
    sys.exit(main())
