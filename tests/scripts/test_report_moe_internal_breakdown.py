from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/report_moe_internal_breakdown.py")


def _moe_record() -> dict:
    return {
        "case_id": "moe__moe_n2048_motp1_moep4_balanced_cudagraph__test",
        "op_kind": "moe",
        "framework": "vllm",
        "framework_version": "0.19.1",
        "device": "NVIDIA GeForce RTX 4090",
        "execution_mode": "cudagraph",
        "kernel_source": "vllm_fused_moe",
        "params": {
            "moe_dtype": "bfloat16",
            "num_tokens": 2048,
            "hidden_size": 2048,
            "inter_size": 768,
            "topk": 8,
            "num_experts": 128,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "distribution": "balanced",
            "execution_mode": "cudagraph",
        },
        "metrics": {
            "latency_us_p50": 642.048,
            "latency_us_p10": 638.0,
            "latency_us_p90": 648.0,
            "used_cuda_graph": True,
            "n_warmups": 3,
            "n_iters": 10,
        },
        "metadata": {
            "source_profiles": ["qwen3_30b_a3b"],
            "fallback_reason": None,
        },
        "schema_version": "collector-v2",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _run_report(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--db-root",
            str(tmp_path / "operator_db"),
            "--hardware",
            "RTX_4090",
            "--framework",
            "vllm",
            "--framework-version",
            "0.19.1",
            *extra_args,
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        text=True,
        capture_output=True,
    )


def test_moe_internal_breakdown_writes_diagnostic_columns(tmp_path: Path):
    partition = tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"
    _write_jsonl(partition / "moe.jsonl", [_moe_record()])
    csv_path = tmp_path / "moe_internal.csv"

    result = _run_report(tmp_path, "--csv", str(csv_path))

    assert "MoE internal residual cases" in result.stdout
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "moe__moe_n2048_motp1_moep4_balanced_cudagraph__test"
    assert row["routing_distribution"] == "balanced"
    assert float(row["roofline_us"]) > 0
    assert float(row["residual_us"]) > 0
    assert float(row["gate_compute_us"]) > 0
    assert float(row["up_weight_mb"]) > 0
    assert float(row["down_weight_mb"]) > 0
    assert float(row["avg_rows_per_expert_rank"]) == 128.0
    assert float(row["extra_mem_equiv_mb"]) > 0
    assert float(row["materialized_intermediate_extra_us"]) > 0


def test_moe_internal_breakdown_case_filter(tmp_path: Path):
    partition = tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"
    wanted = _moe_record()
    other = _moe_record()
    other["case_id"] = "moe__other"
    _write_jsonl(partition / "moe.jsonl", [wanted, other])
    csv_path = tmp_path / "filtered.csv"

    _run_report(
        tmp_path,
        "--case-id",
        wanted["case_id"],
        "--csv",
        str(csv_path),
    )

    rows = list(csv.DictReader(csv_path.open()))
    assert [r["case_id"] for r in rows] == [wanted["case_id"]]
