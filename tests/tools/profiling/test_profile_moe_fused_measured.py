from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/profile/profile_moe_fused_measured.py")


def _fake_trace(path: Path) -> None:
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "moe_align_block_size_kernel",
                "dur": 12.0,
            },
            {
                "ph": "X",
                "cat": "cuda_kernel",
                "name": "fused_moe_grouped_gemm_kernel",
                "dur": 100.0,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "moe_gather_scatter_kernel",
                "dur": 30.0,
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "ignored_cpu_event",
                "dur": 999.0,
            },
        ]
    }
    path.write_text(json.dumps(trace))


def test_parse_existing_trace_writes_measured_kernel_breakdown(tmp_path: Path):
    trace = tmp_path / "trace.json"
    csv_path = tmp_path / "measured.csv"
    json_path = tmp_path / "measured.json"
    _fake_trace(trace)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--parse",
            str(trace),
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "total kernels: 3" in result.stdout
    rows = list(csv.DictReader(csv_path.open()))
    by_cat = {row["category"]: row for row in rows}
    assert float(by_cat["moe_grouped_gemm"]["total_us"]) == 100.0
    assert float(by_cat["moe_align_sort"]["total_us"]) == 12.0
    assert float(by_cat["moe_permute_gather"]["total_us"]) == 30.0

    parsed = json.loads(json_path.read_text())
    assert parsed["total_kernel_us"] == 142.0
