"""Stage 校准结果汇总 + SLA 对比.

输入: bench_compare 跑完的 results_dir/<label>/Qwen3-4B*/sim_*.txt + real_*.txt
输出: 平均 abs gap, per-scenario gap, vs SLA 表 (CALIBRATION_METHODOLOGY §4.1)

跑法:
    python scripts/analyze_stage.py /tmp/stage_A      # 单 dir
    python scripts/analyze_stage.py /tmp/stage_A /tmp/stage_B  # 多 dir
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path


# CALIBRATION_METHODOLOGY §4.1 SLA 表
SLA = {
    "TPOT":         {1: 10, 2: 15, 4: 20, 8: 25},
    "TTFT_single":  {1: 15, 2: 20, 4: 25, 8: 30},
    "TTFT_multi":   {1: 20, 2: 25, 4: 30, 8: 35},
    "Throughput":   {1: 10, 2: 15, 4: 20, 8: 25},
}


def parse_bench_txt(path: Path) -> dict:
    """Parse Mean TTFT/TPOT/Throughput from vllm bench serve output."""
    out = {}
    text = path.read_text()
    for key, pattern in [
        ("TTFT_ms",       r"Mean TTFT \(ms\):\s+(\d+\.?\d*)"),
        ("TPOT_ms",       r"Mean TPOT \(ms\):\s+(\d+\.?\d*)"),
        ("throughput_tps", r"Output token throughput \(tok/s\):\s+(\d+\.?\d*)"),
    ]:
        m = re.search(pattern, text)
        if m: out[key] = float(m.group(1))
    return out


def collect_results(stage_dir: Path) -> list[dict]:
    """从 stage_dir 找所有 <label>/<model>/(real|sim)_<scen>.txt 配对."""
    records = []
    for label_dir in sorted(stage_dir.iterdir()):
        if not label_dir.is_dir(): continue
        label = label_dir.name
        # 推 TP from label
        m = re.search(r"TP(\d+)", label)
        tp = int(m.group(1)) if m else 1
        for model_dir in label_dir.iterdir():
            if not model_dir.is_dir(): continue
            model = model_dir.name
            # 找 real_xxx.txt + sim_xxx.txt 配对
            real_files = list(model_dir.glob("real_*.txt"))
            for rf in real_files:
                scen = rf.stem[len("real_"):]
                sf = model_dir / f"sim_{scen}.txt"
                if not sf.exists(): continue
                real_metrics = parse_bench_txt(rf)
                sim_metrics  = parse_bench_txt(sf)
                rec = {"label": label, "tp": tp, "scenario": scen, "model": model}
                for k in ("TTFT_ms", "TPOT_ms", "throughput_tps"):
                    if k in real_metrics and k in sim_metrics:
                        rec[f"real_{k}"] = real_metrics[k]
                        rec[f"sim_{k}"]  = sim_metrics[k]
                        rec[f"gap_{k}_pct"] = (
                            (sim_metrics[k] - real_metrics[k]) / real_metrics[k] * 100
                        )
                records.append(rec)
    return records


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 1
    all_records = []
    for d in sys.argv[1:]:
        all_records.extend(collect_results(Path(d)))
    if not all_records:
        print("no records found in:", sys.argv[1:]); return 1

    print(f"{'label':<22} {'scen':<12} {'TP':>3} "
          f"{'TTFT_gap%':>10} {'TPOT_gap%':>10} {'thru_gap%':>10}")
    print("-" * 75)
    for r in all_records:
        print(f"{r['label']:<22} {r['scenario']:<12} {r['tp']:>3} "
              f"{r.get('gap_TTFT_ms_pct', 'N/A'):>10.1f} "
              f"{r.get('gap_TPOT_ms_pct', 'N/A'):>10.1f} "
              f"{r.get('gap_throughput_tps_pct', 'N/A'):>10.1f}")

    # ---- per-(tp,label) avg abs gap + SLA check ----
    print()
    print("=== Aggregated by (label, TP) ===")
    print(f"{'label':<22} {'TP':>3} {'TTFT_avg':>10} {'TPOT_avg':>10} {'thru_avg':>10} "
          f"{'TTFT_SLA':>10} {'TPOT_SLA':>10} {'PASS':>6}")
    print("-" * 90)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_records:
        groups[(r["label"], r["tp"])].append(r)
    for (label, tp), recs in sorted(groups.items()):
        ttfts = [abs(r["gap_TTFT_ms_pct"]) for r in recs if "gap_TTFT_ms_pct" in r]
        tpots = [abs(r["gap_TPOT_ms_pct"]) for r in recs if "gap_TPOT_ms_pct" in r]
        thrs  = [abs(r["gap_throughput_tps_pct"]) for r in recs if "gap_throughput_tps_pct" in r]
        ttft_avg = sum(ttfts) / len(ttfts) if ttfts else 0
        tpot_avg = sum(tpots) / len(tpots) if tpots else 0
        thr_avg  = sum(thrs)  / len(thrs)  if thrs  else 0
        # SLA — 用 multi 因为大部分场景多请求; 若 label 含 'single' 用 single
        ttft_sla_key = "TTFT_single" if "single" in label.lower() else "TTFT_multi"
        ttft_sla = SLA[ttft_sla_key].get(tp, 30)
        tpot_sla = SLA["TPOT"].get(tp, 25)
        passes = "✓" if (ttft_avg <= ttft_sla and tpot_avg <= tpot_sla) else "✗"
        print(f"{label:<22} {tp:>3} {ttft_avg:>9.1f}% {tpot_avg:>9.1f}% {thr_avg:>9.1f}% "
              f"{ttft_sla:>9}% {tpot_sla:>9}% {passes:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
