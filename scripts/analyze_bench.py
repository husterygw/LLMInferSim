"""Analyze case-driven benchmark results.

输入: <out_root>/<suite>/<case_id>/metrics.json (bench_compare.sh 产出)
输出:
  1) 每 case 一行的对比表 (TTFT/TPOT/throughput mean+p99 + gap%)
  2) 按 (suite, tp) 聚合 abs gap 均值 + SLA 判定
  3) 可选 --csv / --jsonl dump

跑法:
  python scripts/analyze_bench.py /tmp/llm_infer_sim_bench                       # 全部 group
  python scripts/analyze_bench.py /tmp/llm_infer_sim_bench --suite single_tp1_roofline
  python scripts/analyze_bench.py /tmp/llm_infer_sim_bench --csv /tmp/bench.csv
  python scripts/analyze_bench.py /tmp/llm_infer_sim_bench --jsonl /tmp/bench.jsonl

SLA 表 ≡ docs/CALIBRATION_METHODOLOGY.md §4.1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# SLA 表: gap% (绝对值) 阈值, key by TP
SLA = {
    "TPOT":         {1: 15, 2: 15, 4: 20, 8: 25},   # CALIBRATION_METHODOLOGY §4.1
    "TTFT_single":  {1: 15, 2: 20, 4: 25, 8: 30},
    "TTFT_multi":   {1: 20, 2: 25, 4: 30, 8: 35},
    "Throughput":   {1: 15, 2: 20, 4: 25, 8: 30},
}


def suite_to_sla_key(suite: str) -> tuple[str, float]:
    """Return (TTFT SLA key, TPOT SLA multiplier).

    moe_* 暂时 TPOT 给 1.2× 松一点 (M-A AllReduce in-context floor 还没建模)."""
    if suite.startswith("moe_"):
        ttft_key = "TTFT_multi" if "batch" in suite else "TTFT_single"
        return ttft_key, 1.2
    if "batch" in suite or "multi_model" in suite:
        return "TTFT_multi", 1.0
    return "TTFT_single", 1.0


def load_cases(cases_path: Path) -> dict[str, dict]:
    """case_id → case meta dict."""
    out: dict[str, dict] = {}
    if not cases_path.exists():
        return out
    for line in cases_path.read_text().splitlines():
        c = json.loads(line)
        if "suite" not in c and "group" in c:
            c["suite"] = c["group"]
        out[c["case_id"]] = c
    return out


def collect(out_root: Path, suite_filter: str | None, cases: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    suite_dirs = [out_root / suite_filter] if suite_filter else \
        [p for p in out_root.iterdir() if p.is_dir() and p.name != "__pycache__"]
    for sdir in sorted(suite_dirs):
        if not sdir.is_dir():
            continue
        suite = sdir.name
        for case_dir in sorted(sdir.iterdir()):
            if not case_dir.is_dir() or case_dir.name.startswith("__"):
                continue
            mfile = case_dir / "metrics.json"
            if not mfile.exists():
                continue
            try:
                m = json.loads(mfile.read_text())
            except json.JSONDecodeError:
                continue
            case_id = m.get("case_id", case_dir.name)
            meta = cases.get(case_id, {})
            for s in m.get("scenarios", []):
                rows.append({
                    "suite": suite,
                    "case_id": case_id,
                    "model": meta.get("model_alias", "?"),
                    "tp": meta.get("tp", 1),
                    "ep": meta.get("ep", 1),
                    "hint": meta.get("topology_hint", "?"),
                    "concurrency": meta.get("concurrency", 1),
                    "scenario": s.get("scenario", "?"),
                    "real_TTFT_mean": s.get("real", {}).get("TTFT_ms_mean"),
                    "sim_TTFT_mean":  s.get("sim",  {}).get("TTFT_ms_mean"),
                    "real_TTFT_p99":  s.get("real", {}).get("TTFT_ms_p99"),
                    "sim_TTFT_p99":   s.get("sim",  {}).get("TTFT_ms_p99"),
                    "real_TPOT_mean": s.get("real", {}).get("TPOT_ms_mean"),
                    "sim_TPOT_mean":  s.get("sim",  {}).get("TPOT_ms_mean"),
                    "real_TPOT_p99":  s.get("real", {}).get("TPOT_ms_p99"),
                    "sim_TPOT_p99":   s.get("sim",  {}).get("TPOT_ms_p99"),
                    "real_thru_tps":  s.get("real", {}).get("throughput_output_tok_per_s"),
                    "sim_thru_tps":   s.get("sim",  {}).get("throughput_output_tok_per_s"),
                    "gap_TTFT_mean": s.get("gap_pct", {}).get("TTFT_mean"),
                    "gap_TPOT_mean": s.get("gap_pct", {}).get("TPOT_mean"),
                    "gap_TTFT_p99":  s.get("gap_pct", {}).get("TTFT_p99"),
                    "gap_TPOT_p99":  s.get("gap_pct", {}).get("TPOT_p99"),
                    "gap_thru":      s.get("gap_pct", {}).get("throughput_req"),
                })
    return rows


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v:+.1f}%"


def fmt_ms(v):
    if v is None:
        return "MISS"
    return f"{v:.1f}"


def print_table(rows: list[dict]) -> None:
    print(f"{'suite':<28} {'case_id':<64} {'TP':>3} {'C':>3} "
          f"{'real_TTFT':>10} {'sim_TTFT':>10} {'TTFT_gap':>10} "
          f"{'real_TPOT':>10} {'sim_TPOT':>10} {'TPOT_gap':>10}")
    print("-" * 155)
    for r in rows:
        print(f"{r['suite']:<28} {r['case_id']:<64} {r['tp']:>3} {r['concurrency']:>3} "
              f"{fmt_ms(r['real_TTFT_mean']):>10} {fmt_ms(r['sim_TTFT_mean']):>10} {fmt_pct(r['gap_TTFT_mean']):>10} "
              f"{fmt_ms(r['real_TPOT_mean']):>10} {fmt_ms(r['sim_TPOT_mean']):>10} {fmt_pct(r['gap_TPOT_mean']):>10}")


def aggregate(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["tp"])].append(r)

    print()
    print("=== Aggregated by (suite, TP) — abs gap mean vs SLA ===")
    print(f"{'suite':<28} {'TP':>3} {'N':>3} "
          f"{'TTFT_mean':>10} {'TPOT_mean':>10} {'thru':>8} "
          f"{'TTFT_SLA':>10} {'TPOT_SLA':>10} {'verdict':>8}")
    print("-" * 110)

    for (suite, tp), recs in sorted(groups.items()):
        ttfts = [abs(r["gap_TTFT_mean"]) for r in recs if r["gap_TTFT_mean"] is not None]
        tpots = [abs(r["gap_TPOT_mean"]) for r in recs if r["gap_TPOT_mean"] is not None]
        thrs  = [abs(r["gap_thru"])      for r in recs if r["gap_thru"]      is not None]
        ttft_avg = sum(ttfts) / len(ttfts) if ttfts else None
        tpot_avg = sum(tpots) / len(tpots) if tpots else None
        thr_avg  = sum(thrs)  / len(thrs)  if thrs  else None

        ttft_key, moe_mul = suite_to_sla_key(suite)
        ttft_sla = SLA[ttft_key].get(tp, 35)
        tpot_sla = SLA["TPOT"].get(tp, 25) * moe_mul

        verdict = "?"
        if ttft_avg is not None and tpot_avg is not None:
            verdict = "PASS" if (ttft_avg <= ttft_sla and tpot_avg <= tpot_sla) else "FAIL"

        print(f"{suite:<28} {tp:>3} {len(recs):>3} "
              f"{(f'{ttft_avg:6.1f}%' if ttft_avg is not None else '   N/A'):>10} "
              f"{(f'{tpot_avg:6.1f}%' if tpot_avg is not None else '   N/A'):>10} "
              f"{(f'{thr_avg:5.1f}%' if thr_avg is not None else ' N/A'):>8} "
              f"{ttft_sla:>9.0f}% {tpot_sla:>9.0f}% {verdict:>8}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", help="benchmark output root (e.g. /tmp/llm_infer_sim_bench)")
    ap.add_argument("--suite", help="filter to single suite")
    ap.add_argument("--group", help="deprecated alias for --suite")
    ap.add_argument("--cases", default="/tmp/llm_infer_sim_bench/cases.jsonl",
                    help="cases.jsonl path (for meta join)")
    ap.add_argument("--csv", help="dump per-case rows to CSV at this path")
    ap.add_argument("--jsonl", help="dump per-case rows to JSONL at this path")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_dir():
        print(f"not a dir: {out_root}", file=sys.stderr); return 1

    cases = load_cases(Path(args.cases))
    suite_filter = args.suite or args.group
    rows = collect(out_root, suite_filter, cases)
    if not rows:
        print(f"no metrics.json under {out_root}"
              f"{' (suite=' + suite_filter + ')' if suite_filter else ''}")
        return 1

    print_table(rows)
    aggregate(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote CSV: {args.csv}")
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote JSONL: {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
