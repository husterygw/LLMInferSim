"""Report measured AllReduce latency vs sim roofline.

跟 report_operator_roofline_gap.py (GEMM) 同模式: 喂 collector AR 数据, sim 用同样
(n, topology, mode, size) 输入跑 allreduce_time(), 输出 measured/sim gap.

跑法:
    python scripts/report_allreduce_roofline_gap.py \
        --db-root collector/data/operator_db \
        --hardware RTX_4090 \
        --csv /tmp/ar_gap.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_infer_sim.core.cost.roofline.communication import allreduce_time
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile


FIELDNAMES = [
    "case_id",
    "execution_mode",
    "dtype",
    "num_gpus",
    "topology_hint",
    "message_size_bytes",
    "in_context",
    "kernel_source",
    "measured_us_p50",
    "measured_us_p10",
    "measured_us_p90",
    "sim_us",
    "gap_meas_over_sim",
    "delta_us",
    "source_profiles",
]


def load_records(path: Path) -> list[dict]:
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


def estimate_gap(record: dict, hw) -> dict[str, Any]:
    p = record["params"]
    m = record["metrics"]
    mode = p["execution_mode"]
    n = int(p["num_gpus"])
    size = int(p["message_size_bytes"])
    topo = p["topology_hint"]
    sim_s = allreduce_time(
        data_bytes=float(size),
        n=n,
        hw=hw,
        mode=mode,
        topology_hint=topo,
    )
    sim_us = sim_s * 1e6
    measured = float(m["latency_us_p50"])
    gap = measured / sim_us if sim_us > 0 else None
    return {
        "case_id": record["case_id"],
        "execution_mode": mode,
        "dtype": p["dtype"],
        "num_gpus": n,
        "topology_hint": topo,
        "message_size_bytes": size,
        "in_context": p.get("in_context", False),
        "kernel_source": record.get("kernel_source", ""),
        "measured_us_p50": measured,
        "measured_us_p10": m.get("latency_us_p10"),
        "measured_us_p90": m.get("latency_us_p90"),
        "sim_us": sim_us,
        "gap_meas_over_sim": gap,
        "delta_us": measured - sim_us,
        "source_profiles": ",".join(record.get("metadata", {}).get("source_profiles") or []),
    }


def _fmt_size(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)}M"
    if b >= 1024:
        return f"{b // 1024}K"
    return f"{b}B"


def print_summary(rows: list[dict]) -> None:
    n_rows = len(rows)
    print(f"rows={n_rows}")
    if not rows:
        return
    print()
    print("=== AR roofline gap by (n_gpus, topology, mode) ===")
    print(f"{'n':>3} {'topology':<14} {'mode':<10} {'N':>4} "
          f"{'gap_p50':>9} {'gap_p10':>9} {'gap_p90':>9} {'gap_max':>9} "
          f"{'meas_med':>10} {'sim_med':>10}")
    print("-" * 100)
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[(r["num_gpus"], r["topology_hint"], r["execution_mode"])].append(r)
    for (n, topo, mode), recs in sorted(by_combo.items()):
        gaps = [r["gap_meas_over_sim"] for r in recs if r["gap_meas_over_sim"] is not None]
        if not gaps:
            continue
        gaps.sort()
        meas_med = statistics.median(r["measured_us_p50"] for r in recs)
        sim_med = statistics.median(r["sim_us"] for r in recs)
        print(f"{n:>3} {topo:<14} {mode:<10} {len(recs):>4} "
              f"{statistics.median(gaps):>9.2f} {gaps[max(0, int(0.10*len(gaps)))]:>9.2f} "
              f"{gaps[min(len(gaps)-1, int(0.90*len(gaps)))]:>9.2f} {max(gaps):>9.2f} "
              f"{meas_med:>9.1f}us {sim_med:>9.1f}us")


def print_per_size(rows: list[dict]) -> None:
    print()
    print("=== AR gap per (size, n_gpus, topology) — cudagraph only ===")
    cg = [r for r in rows if r["execution_mode"] == "cudagraph"]
    by_key = defaultdict(dict)
    for r in cg:
        by_key[r["message_size_bytes"]][(r["num_gpus"], r["topology_hint"])] = r

    sizes = sorted(by_key.keys())
    combos = sorted({(r["num_gpus"], r["topology_hint"]) for r in cg})

    header = f"{'size':>8}"
    for n, t in combos:
        header += f"  {n}_{t[:3]:>7}"
    print(header)
    print("-" * (8 + len(combos) * 11))
    for s in sizes:
        row = f"{_fmt_size(s):>8}"
        for n, t in combos:
            r = by_key[s].get((n, t))
            if r and r["gap_meas_over_sim"]:
                row += f"  {r['gap_meas_over_sim']:>8.2f}"
            else:
                row += f"  {'?':>8}"
        print(row)

    print()
    print("=== Detailed: measured / sim by size for n_gpus=2 concentrated cudagraph ===")
    print(f"{'size':>8} {'meas_us':>10} {'sim_us':>10} {'gap':>8} {'delta_us':>10}")
    for s in sizes:
        r = by_key[s].get((2, "concentrated"))
        if r:
            print(f"{_fmt_size(s):>8} {r['measured_us_p50']:>9.1f}us {r['sim_us']:>9.1f}us "
                  f"{r['gap_meas_over_sim']:>8.2f} {r['delta_us']:>+9.1f}us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-root", default="collector/data/operator_db",
                    help="root containing <hardware>/<framework_version>/collective.jsonl")
    ap.add_argument("--hardware", default="RTX_4090")
    ap.add_argument("--framework-version", default="vllm-0.19.1")
    ap.add_argument("--csv", help="dump per-row CSV")
    args = ap.parse_args()

    db = Path(args.db_root) / args.hardware / args.framework_version / "collective.jsonl"
    if not db.exists():
        print(f"no DB: {db}", file=sys.stderr)
        return 1

    hw = get_hardware_profile(args.hardware)
    records = load_records(db)
    print(f"loaded {len(records)} AR records from {db}")

    rows = [estimate_gap(r, hw) for r in records]
    print_summary(rows)
    print_per_size(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
