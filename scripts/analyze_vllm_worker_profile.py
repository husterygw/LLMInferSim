#!/usr/bin/env python3
"""Summarize vLLM worker profile JSONL emitted by LLM_INFER_VLLM_WORKER_PROFILE."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def load_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(path: Path) -> None:
    by_segment: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_record: dict[tuple[str, str], list[float]] = defaultdict(list)
    records = list(load_records(path))
    phase_by_step = {
        rec.get("step_id"): rec.get("phase")
        for rec in records
        if rec.get("record") == "execute_model" and rec.get("step_id") is not None
    }

    for rec in records:
        phase = rec.get("phase") or phase_by_step.get(rec.get("step_id")) or "unknown"
        record = rec.get("record") or "unknown"
        total_us = float(rec.get("total_us") or 0.0)
        by_record[(phase, record)].append(total_us)
        for seg in rec.get("segments", []):
            by_segment[(phase, record, seg.get("name", "unknown"))].append(
                float(seg.get("us") or 0.0)
            )

    print("== record totals ==")
    print("phase      record          count   mean_us    p50_us    p90_us")
    for (phase, record), values in sorted(by_record.items()):
        print(
            f"{phase:<10} {record:<15} {len(values):>5} "
            f"{statistics.mean(values):>9.1f} "
            f"{percentile(values, 0.50):>9.1f} "
            f"{percentile(values, 0.90):>9.1f}"
        )

    print("\n== segments ==")
    print("phase      record          segment                       count   mean_us    p50_us    p90_us")
    for (phase, record, segment), values in sorted(by_segment.items()):
        print(
            f"{phase:<10} {record:<15} {segment:<30} {len(values):>5} "
            f"{statistics.mean(values):>9.1f} "
            f"{percentile(values, 0.50):>9.1f} "
            f"{percentile(values, 0.90):>9.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_jsonl", type=Path)
    args = parser.parse_args()
    summarize(args.profile_jsonl)


if __name__ == "__main__":
    main()
