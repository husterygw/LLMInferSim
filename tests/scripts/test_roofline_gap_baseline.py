"""Baseline: scripts/report_operator_roofline_gap.py 数据回归.

锁住:
  rows=144 ok=144 — 验证 V3 StepCostEngine 链路 GEMM gap 计算不漂.
  baseline CSV: docs/baselines/gemm_roofline_gap_RTX_4090_vllm-0.19.1.csv
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_operator_roofline_gap.py"
BASELINE = REPO_ROOT / "docs" / "baselines" / "gemm_roofline_gap_RTX_4090_vllm-0.19.1.csv"
DB_ROOT = REPO_ROOT / "collector" / "data" / "operator_db"
DB_FILE = DB_ROOT / "RTX_4090" / "vllm-0.19.1" / "gemm.jsonl"

pytestmark = pytest.mark.skipif(
    not DB_FILE.exists(),
    reason="real collector JSONL not present (RTX_4090 / vllm-0.19.1 / gemm)",
)


def _run_report(tmp_path: Path) -> tuple[str, Path]:
    """Run the report script, return (stdout, csv_path)."""
    csv_path = tmp_path / "gap.csv"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--db-root", str(DB_ROOT),
            "--hardware", "RTX_4090",
            "--framework", "vllm",
            "--framework-version", "0.19.1",
            "--op-kind", "gemm",
            "--csv", str(csv_path),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    return proc.stdout, csv_path


def test_report_runs_with_144_ok_rows(tmp_path):
    stdout, csv_path = _run_report(tmp_path)
    # 第一行应该是 rows=N ok=N
    first_line = stdout.splitlines()[0].strip()
    assert first_line.startswith("rows=144 ok=144")
    # CSV 必须 144 行 (excluding header)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 144
    assert all(r["status"] == "ok" for r in rows)


def test_baseline_csv_matches_current_run(tmp_path):
    """docs/baselines 里的 snapshot 应跟最新 run 数字一致 (defense against drift)."""
    if not BASELINE.exists():
        pytest.skip("baseline CSV not yet checked in")
    _, current_csv = _run_report(tmp_path)
    with current_csv.open() as f:
        current_rows = list(csv.DictReader(f))
    with BASELINE.open() as f:
        baseline_rows = list(csv.DictReader(f))
    assert len(current_rows) == len(baseline_rows), (
        f"row count mismatch: current={len(current_rows)} baseline={len(baseline_rows)}"
    )
    # 按 case_id 比 measured + roofline + gap; case_id 是 unique key.
    cur_map = {r["case_id"]: r for r in current_rows}
    for b in baseline_rows:
        c = cur_map.get(b["case_id"])
        assert c is not None, f"missing case_id {b['case_id']} in current run"
        # measured 来自 JSONL 静态数据, 跟 baseline 一字节 byte-for-byte
        assert c["measured_us_p50"] == b["measured_us_p50"], (
            f"{b['case_id']}: measured drift"
        )
        # roofline 是 cost engine 计算结果, 删除 legacy 后应保持一致
        assert c["roofline_us"] == b["roofline_us"], (
            f"{b['case_id']}: roofline drift "
            f"(current={c['roofline_us']} baseline={b['roofline_us']})"
        )
        assert c["roofline_gap"] == b["roofline_gap"], (
            f"{b['case_id']}: gap drift"
        )
