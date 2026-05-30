#!/usr/bin/env python3
"""Compare bench wall-clock TTFT/TPOT vs sim cost-engine raw step latency.

bench_compare.sh produces:
  - real_<scenario>.txt   (real wall-clock TTFT/TPOT)
  - sim_<scenario>.txt    (sim wall-clock TTFT/TPOT)
  - sim_server.log        (per-step [VirtualModelRunner] cost-engine latency)

sim wall-clock = cost-engine sleep + LLMInferSim Python glue overhead. The glue
shows up as a constant per-step tax that has nothing to do with the cost engine
公式. This script peels it out by reading raw step latencies from sim_server.log
and prints them side-by-side with bench wall-clock + real metrics, so the
cost-engine-vs-real gap can be judged independent of sim-side Python overhead.

Usage:
  python scripts/analyze_sim_step_latency.py <results_dir>

  results_dir = the per-model dir bench_compare.sh wrote, e.g.
    /tmp/stage_A/TP1_default/Qwen3-4B-Instruct-2507/
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

STEP_RE = re.compile(
    r"\[VirtualModelRunner\] step=(\d+) phase=(\w+) latency=\s*([\d.]+)us "
    r"\| compute=\s*([\d.]+)us memory=\s*([\d.]+)us comm=\s*([\d.]+)us "
    r"\| bottleneck=(\w+)"
)

METRIC_RE = {
    "TTFT": re.compile(r"Mean TTFT \(ms\):\s*([\d.]+)"),
    "TPOT": re.compile(r"Mean TPOT \(ms\):\s*([\d.]+)"),
}


def parse_steps(log_path: Path) -> list[dict]:
    out = []
    for line in log_path.read_text().splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step_id, phase, lat, comp, mem, comm, bn = m.groups()
        out.append({
            "step_id": int(step_id),
            "phase": phase,
            "latency_us": float(lat),
            "compute_us": float(comp),
            "memory_us": float(mem),
            "comm_us": float(comm),
            "bottleneck": bn,
        })
    return out


def parse_metric(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    m = METRIC_RE[key].search(path.read_text())
    return float(m.group(1)) if m else None


def stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.mean(s),
        "p10": s[int(0.1 * (n - 1))],
        "p50": s[int(0.5 * (n - 1))],
        "p90": s[int(0.9 * (n - 1))],
        "min": s[0],
        "max": s[-1],
    }


def fmt_stats_row(label: str, st: dict) -> str:
    if not st:
        return f"  {label}: no data"
    return (
        f"  {label}: n={st['n']:>4}  mean={st['mean']:>8.1f}  "
        f"p10={st['p10']:>8.1f}  p50={st['p50']:>8.1f}  "
        f"p90={st['p90']:>8.1f}  min={st['min']:>8.1f}  max={st['max']:>8.1f}"
    )


def gap_pct(s: float | None, r: float | None) -> str:
    if r is None or s is None or r == 0:
        return "  -"
    return f"{(s - r) / r * 100:+.1f}"


def fmt_num(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "MISS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path,
                    help="per-model dir containing real_*.txt, sim_*.txt, sim_server.log")
    args = ap.parse_args()
    d = args.results_dir
    if not d.is_dir():
        raise SystemExit(f"not a dir: {d}")

    log = d / "sim_server.log"
    if not log.exists():
        raise SystemExit(f"missing sim_server.log in {d}")

    steps = parse_steps(log)
    pf = [s["latency_us"] for s in steps if s["phase"] == "prefill"]
    dc = [s["latency_us"] for s in steps if s["phase"] == "decode"]
    pf_compute = [s["compute_us"] for s in steps if s["phase"] == "prefill"]
    pf_mem = [s["memory_us"] for s in steps if s["phase"] == "prefill"]
    dc_compute = [s["compute_us"] for s in steps if s["phase"] == "decode"]
    dc_mem = [s["memory_us"] for s in steps if s["phase"] == "decode"]

    scenarios = sorted({p.stem[len("real_"):] for p in d.glob("real_*.txt")})

    print(f"results: {d}")
    print(f"steps parsed: prefill={len(pf)}  decode={len(dc)}")
    print()
    print("== Wall-clock end-to-end (bench reported) ==")
    print(f"  {'scenario':<14}{'real_TTFT':>11}{'sim_TTFT':>11}{'gap%':>8}    "
          f"{'real_TPOT':>11}{'sim_TPOT':>11}{'gap%':>8}")
    for name in scenarios:
        r_t = parse_metric(d / f"real_{name}.txt", "TTFT")
        s_t = parse_metric(d / f"sim_{name}.txt", "TTFT")
        r_p = parse_metric(d / f"real_{name}.txt", "TPOT")
        s_p = parse_metric(d / f"sim_{name}.txt", "TPOT")
        print(
            f"  {name:<14}{fmt_num(r_t):>11}{fmt_num(s_t):>11}{gap_pct(s_t, r_t):>7}%    "
            f"{fmt_num(r_p):>11}{fmt_num(s_p):>11}{gap_pct(s_p, r_p):>7}%"
        )

    print()
    print("== Sim cost-engine raw step latency (us) — excludes sim Python glue overhead ==")
    print(fmt_stats_row("prefill latency ", stats(pf)))
    print(fmt_stats_row("prefill compute ", stats(pf_compute)))
    print(fmt_stats_row("prefill memory  ", stats(pf_mem)))
    print(fmt_stats_row("decode  latency ", stats(dc)))
    print(fmt_stats_row("decode  compute ", stats(dc_compute)))
    print(fmt_stats_row("decode  memory  ", stats(dc_mem)))

    if len(scenarios) == 1:
        name = scenarios[0]
        r_t = parse_metric(d / f"real_{name}.txt", "TTFT")
        r_p = parse_metric(d / f"real_{name}.txt", "TPOT")
        s_t_wc = parse_metric(d / f"sim_{name}.txt", "TTFT")
        s_p_wc = parse_metric(d / f"sim_{name}.txt", "TPOT")
        cost_ttft_ms = sum(pf) / 1000.0 if pf else None
        cost_tpot_ms = (sum(dc) / len(dc)) / 1000.0 if dc else None
        print()
        print(f"== Single-scenario direct comparison ({name}) ==")
        print(f"  {'metric':<6}{'real':>11}{'sim_wall':>11}{'sim_cost':>11}"
              f"{'cost vs real':>15}{'wc glue':>11}")
        if r_t is not None and cost_ttft_ms is not None and s_t_wc is not None:
            cost_gap_t = (cost_ttft_ms - r_t) / r_t * 100 if r_t else 0
            glue_t = s_t_wc - cost_ttft_ms
            print(f"  TTFT  {r_t:>11.2f}{s_t_wc:>11.2f}{cost_ttft_ms:>11.2f}"
                  f"{cost_gap_t:>14.1f}%{glue_t:>11.2f}")
        if r_p is not None and cost_tpot_ms is not None and s_p_wc is not None:
            cost_gap_p = (cost_tpot_ms - r_p) / r_p * 100 if r_p else 0
            glue_p = s_p_wc - cost_tpot_ms
            print(f"  TPOT  {r_p:>11.2f}{s_p_wc:>11.2f}{cost_tpot_ms:>11.2f}"
                  f"{cost_gap_p:>14.1f}%{glue_p:>11.2f}")
        print(
            "  (sim_cost = cost engine sum-of-prefill-steps / mean-of-decode-steps in ms;\n"
            "   wc glue = sim_wall - sim_cost = per-step LLMInferSim Python overhead)"
        )


if __name__ == "__main__":
    main()
