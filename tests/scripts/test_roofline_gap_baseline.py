"""Baseline: scripts/report_operator_roofline_gap.py 数据回归.

锁住:
  rows ≥ 144, ok == rows — 验证 V3 StepCostEngine 链路 GEMM gap 计算不漂.
  baseline CSV: docs/baselines/gemm_roofline_gap_RTX_4090_vllm-0.19.1.csv

注: 144 是 tp=1 GEMM case 的最小集; 后续可能加 tp>1 / 其他 profile 让 rows 增长,
本测试只保证 baseline 里的 case 全部仍 ok + 数字不漂, 不锁死总数.
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


def test_report_runs_with_at_least_baseline_rows(tmp_path):
    stdout, csv_path = _run_report(tmp_path)
    # 第一行 rows=N ok=N, 解析 N
    first_line = stdout.splitlines()[0].strip()
    assert first_line.startswith("rows="), f"unexpected first line: {first_line!r}"
    # 期望 rows ≥ 144 (tp=1 全集), all ok
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 144, f"rows shrunk: {len(rows)} < 144"
    assert all(r["status"] == "ok" for r in rows), "non-ok rows present"


def test_baseline_csv_subset_matches_current_run(tmp_path):
    """docs/baselines snapshot 里的每条 case 必须仍在当前 run 里, 数字 byte-for-byte 一致.

    允许当前 run 多出 case (e.g., 加了 tp>1), 但不允许 baseline 里的 case 数字漂.
    """
    if not BASELINE.exists():
        pytest.skip("baseline CSV not yet checked in")
    _, current_csv = _run_report(tmp_path)
    with current_csv.open() as f:
        current_rows = list(csv.DictReader(f))
    with BASELINE.open() as f:
        baseline_rows = list(csv.DictReader(f))
    assert len(current_rows) >= len(baseline_rows), (
        f"current run shrunk: current={len(current_rows)} baseline={len(baseline_rows)}"
    )
    cur_map = {r["case_id"]: r for r in current_rows}
    for b in baseline_rows:
        c = cur_map.get(b["case_id"])
        assert c is not None, f"missing case_id {b['case_id']} in current run"
        assert c["measured_us_p50"] == b["measured_us_p50"], (
            f"{b['case_id']}: measured drift"
        )
        assert c["roofline_us"] == b["roofline_us"], (
            f"{b['case_id']}: roofline drift "
            f"(current={c['roofline_us']} baseline={b['roofline_us']})"
        )
        assert c["roofline_gap"] == b["roofline_gap"], (
            f"{b['case_id']}: gap drift"
        )
