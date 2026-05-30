"""Compare measured collective times vs Phase 5 cost model predictions.

Input:  /tmp/collective_bench.jsonl (output of measure_collectives.py)
Output: /tmp/collective_analysis.csv (one row per measurement, with gap%)
        + summary table to stdout

Phase 5 公式 (详 docs/COMMUNICATION_MODELING.md):
  T = algorithm_term(min over candidates) + framework_oh × [mode=="eager"]
  β = effective_intra_bw(n, topology_hint, visible_devices) — 拓扑感知

label → (topology_hint, visible_devices):
  same_numa_n2  → balanced=False, GPU [0,1]
  cross_numa_n2 → GPU [0,4]
  same_numa_n4  → GPU [0,1,2,3]
  cross_numa_n4 → GPU [0,1,4,5]
  full_n8       → GPU [0..7]

data_bytes 语义: 各 collective 都按 measure_collectives.py 实际传入的 tensor size
(即 per-rank input bytes), 跟 Phase 5 公式签名约定一致.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.cost.roofline.communication import (
    allreduce_time,
    allgather_time,
    reducescatter_time,
    alltoall_time,
    broadcast_time,
    p2p_time,
)


HW = get_hardware_profile("RTX_4090")


# label → visible_devices map (推 n_per_root)
LABEL_TO_DEVICES = {
    "same_numa_n2":  [0, 1],
    "cross_numa_n2": [0, 4],
    "same_numa_n4":  [0, 1, 2, 3],
    "cross_numa_n4": [0, 1, 4, 5],
    "full_n8":       [0, 1, 2, 3, 4, 5, 6, 7],
}


def predict_us(coll: str, n: int, data_bytes: int, mode: str, label: str) -> float:
    """Predict collective time (microseconds) using Phase 5 cost model."""
    visible = LABEL_TO_DEVICES.get(label)
    kwargs = dict(mode=mode, visible_devices=visible)

    if coll == "allreduce":
        return allreduce_time(data_bytes, n, HW, **kwargs) * 1e6
    if coll == "allgather":
        return allgather_time(data_bytes, n, HW, **kwargs) * 1e6
    if coll == "reducescatter":
        return reducescatter_time(data_bytes, n, HW, **kwargs) * 1e6
    if coll == "alltoall":
        return alltoall_time(data_bytes, n, HW, **kwargs) * 1e6
    if coll == "broadcast":
        return broadcast_time(data_bytes, n, HW, **kwargs) * 1e6
    if coll == "p2p":
        # measure_collectives.py 跑 round-trip (send+recv), 所以 ×2
        return 2 * p2p_time(data_bytes, HW, **kwargs) * 1e6
    raise ValueError(coll)


def main():
    jsonl = Path("/tmp/collective_bench.jsonl")
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}, run measure_collectives.py first")

    records = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        coll = r["collective"]
        n = r["n"]
        size = r["size_bytes"]
        med = r["median_us"]
        mode = r.get("mode", "eager")
        label = r.get("label", "")
        try:
            pred = predict_us(coll, n, size, mode, label)
        except Exception as e:
            pred = float("nan")
        gap_pct = (pred - med) / med * 100 if med > 0 else 0.0
        records.append({
            **r,
            "predicted_us": round(pred, 3) if pred == pred else None,
            "gap_pct": round(gap_pct, 1) if pred == pred else None,
        })

    out_csv = Path("/tmp/collective_analysis.csv")
    keys = ["collective", "label", "n", "mode", "size_bytes",
            "median_us", "p10_us", "p90_us", "predicted_us", "gap_pct"]
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(records)} rows)")

    # --- summary: avg abs gap by (collective, mode, label) ---
    print()
    print(f"{'collective':<14} {'label':<18} {'mode':<10} {'n':>3} "
          f"{'mean_abs_gap%':>14} {'median_gap%':>13} {'sample_n':>9}")
    print("-" * 95)

    groups = defaultdict(list)
    for r in records:
        if r["gap_pct"] is None:
            continue
        groups[(r["collective"], r["label"], r["mode"], r["n"])].append(r["gap_pct"])

    for (coll, label, mode, n), gaps in sorted(groups.items()):
        abs_gaps = [abs(g) for g in gaps]
        mean_abs = sum(abs_gaps) / len(abs_gaps)
        med_gap = statistics.median(gaps)
        print(f"{coll:<14} {label:<18} {mode:<10} {n:>3} "
              f"{mean_abs:>13.1f}% {med_gap:>12.1f}% {len(gaps):>9}")

    # --- overall summary ---
    print()
    all_gaps = [r["gap_pct"] for r in records if r["gap_pct"] is not None]
    abs_gaps = [abs(g) for g in all_gaps]
    print(f"OVERALL: n={len(all_gaps)} mean_abs_gap={sum(abs_gaps)/len(abs_gaps):.1f}% "
          f"median_gap={statistics.median(all_gaps):+.1f}%")


if __name__ == "__main__":
    main()
