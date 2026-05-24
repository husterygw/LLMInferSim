#!/usr/bin/env python3
"""Extract TTFT/TPOT/throughput metrics from a case_dir produced by bench_compare.sh.

Reads `<case_dir>/{real,sim}_<scenario>.txt` (vllm bench serve output),
emits a single JSON on stdout with structure:

    {
      "case_id": "...",
      "group": "...",
      "scenario": "<extracted from filename>",
      "real": {"TTFT_ms_mean": ..., "TTFT_ms_p99": ..., "TPOT_ms_mean": ..., ...},
      "sim":  {... same fields ...},
      "gap_pct": {"TTFT_mean": ..., "TPOT_mean": ...}
    }
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

# 想抽的字段 → 正则 (大小写敏感, vllm bench serve 输出格式稳定)
PATTERNS = {
    "throughput_req_per_s":      r"Request throughput \(req/s\):\s+([0-9.]+)",
    "throughput_output_tok_per_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "TTFT_ms_mean":              r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "TTFT_ms_median":            r"Median TTFT \(ms\):\s+([0-9.]+)",
    "TTFT_ms_p99":               r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "TPOT_ms_mean":              r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "TPOT_ms_median":            r"Median TPOT \(ms\):\s+([0-9.]+)",
    "TPOT_ms_p99":               r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "ITL_ms_median":             r"Median ITL \(ms\):\s+([0-9.]+)",
    "ITL_ms_p99":                r"P99 ITL \(ms\):\s+([0-9.]+)",
}


def parse_one(path: Path) -> dict:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    out = {}
    for field, pat in PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[field] = float(m.group(1))
    return out


def gap_pct(real: float | None, sim: float | None) -> float | None:
    if real is None or sim is None or real == 0:
        return None
    return round((sim - real) / real * 100.0, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--group", required=True, help="suite name; kept as group for JSON compatibility")
    ap.add_argument("--case-dir", required=True, help="case output dir; will glob real_*/sim_* recursively")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        print(f"case_dir not found: {case_dir}", file=sys.stderr)
        return 1

    # New executor writes <case_dir>/{real,sim}_case.txt. The recursive glob also
    # keeps compatibility with older <case_dir>/<model>/{real,sim}_*.txt layouts.
    real_files = sorted(case_dir.glob("**/real_*.txt"))
    sim_files = sorted(case_dir.glob("**/sim_*.txt"))

    if not real_files and not sim_files:
        print(f"no real_*.txt or sim_*.txt under {case_dir}", file=sys.stderr)
        return 2

    # 一个 case 应该只有 1 个 scenario.
    def scen_of(p: Path) -> str:
        # filename pattern: <mode>_<scenario>.txt
        stem = p.stem  # real_case 或 sim_case
        return stem.split("_", 1)[1] if "_" in stem else stem

    scenarios = sorted({scen_of(p) for p in real_files + sim_files})

    result = {
        "case_id": args.case_id,
        "group": args.group,
        "scenarios": [],
    }
    meta_path = case_dir / "block_metadata.json"
    if meta_path.exists():
        try:
            result.update(json.loads(meta_path.read_text()))
        except json.JSONDecodeError:
            pass

    # 多 scenario 兜底 (虽然 case-driven 通常 1 个)
    for scen in scenarios:
        real_path = next((p for p in real_files if scen_of(p) == scen), None)
        sim_path = next((p for p in sim_files if scen_of(p) == scen), None)
        real_metrics = parse_one(real_path) if real_path else {}
        sim_metrics = parse_one(sim_path) if sim_path else {}
        result["scenarios"].append({
            "scenario": scen,
            "real": real_metrics,
            "sim": sim_metrics,
            "gap_pct": {
                "TTFT_mean": gap_pct(real_metrics.get("TTFT_ms_mean"), sim_metrics.get("TTFT_ms_mean")),
                "TPOT_mean": gap_pct(real_metrics.get("TPOT_ms_mean"), sim_metrics.get("TPOT_ms_mean")),
                "TTFT_p99":  gap_pct(real_metrics.get("TTFT_ms_p99"),  sim_metrics.get("TTFT_ms_p99")),
                "TPOT_p99":  gap_pct(real_metrics.get("TPOT_ms_p99"),  sim_metrics.get("TPOT_ms_p99")),
                "throughput_req": gap_pct(real_metrics.get("throughput_req_per_s"),
                                          sim_metrics.get("throughput_req_per_s")),
            },
        })

    # back-compat top-level real/sim (1st scenario).
    if result["scenarios"]:
        first = result["scenarios"][0]
        result["real"] = first["real"]
        result["sim"] = first["sim"]
        result["gap_pct"] = first["gap_pct"]

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
