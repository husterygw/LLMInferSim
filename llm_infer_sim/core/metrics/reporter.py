"""ReportGenerator — 详设 §4.10.2。

阶段 3 范围:
  - console 文本报告 + JSON 结构化报告 + save_report 落盘
  - per-step CSV 推到阶段 X (calibration 触发)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_infer_sim.core.metrics.collector import MetricsCollector


class ReportGenerator:
    """从 MetricsCollector 生成可读 / 结构化报告。"""

    def __init__(self, collector: MetricsCollector) -> None:
        self.collector = collector

    def generate_console_report(self) -> str:
        s = self.collector.get_summary()
        bd = s.avg_step_breakdown
        lines = [
            "=" * 70,
            "  LLM Inference Perf Simulator — Performance Report",
            "=" * 70,
            "",
            f"  Total Requests:      {s.total_requests}",
            f"  Completed:           {s.completed_requests}",
            f"  Output Tokens:       {s.total_output_tokens}",
            f"  Total Steps:         {s.total_steps}",
            f"  Elapsed (sim time):  {s.elapsed_sim_time:.3f}s",
            "",
            "─── Throughput ───",
            f"  Requests / s:        {s.requests_per_second:.2f}",
            f"  Output Tokens / s:   {s.output_tokens_per_second:.2f}",
            "",
            "─── Latency (ms, sim time) ───",
            f"  TTFT  avg/p50/p90/p99:  "
            f"{s.avg_ttft*1e3:7.2f} / {s.p50_ttft*1e3:7.2f} / "
            f"{s.p90_ttft*1e3:7.2f} / {s.p99_ttft*1e3:7.2f}",
            f"  TPOT  avg/p50/p90/p99:  "
            f"{s.avg_tpot*1e3:7.2f} / {s.p50_tpot*1e3:7.2f} / "
            f"{s.p90_tpot*1e3:7.2f} / {s.p99_tpot*1e3:7.2f}",
            f"  E2E   avg/p50/p90/p99:  "
            f"{s.avg_e2e*1e3:7.2f} / {s.p50_e2e*1e3:7.2f} / "
            f"{s.p90_e2e*1e3:7.2f} / {s.p99_e2e*1e3:7.2f}",
            "",
            "─── Avg per-step breakdown (ms) ───",
        ]
        for k in ("total_latency", "compute_time", "memory_time", "comm_time"):
            lines.append(f"  {k:<20}: {bd.get(k, 0.0)*1e3:8.3f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_json_report(self) -> dict:
        return asdict(self.collector.get_summary())

    def save_report(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.generate_json_report(), f, indent=2)

        text_path = path.with_suffix(".txt")
        text_path.write_text(self.generate_console_report())
