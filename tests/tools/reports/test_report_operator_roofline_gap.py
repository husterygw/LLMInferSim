from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/report/report_operator_roofline_gap.py")


def _gemm_record(mode: str = "eager") -> dict:
    return {
        "case_id": f"gemm__qkv_proj_{mode}",
        "op_kind": "gemm",
        "framework": "vllm",
        "framework_version": "0.19.1",
        "device": "NVIDIA GeForce RTX 4090",
        "execution_mode": mode,
        "kernel_source": "vllm_row_parallel_linear",
        "params": {
            "op_subtype": "qkv_proj",
            "m": 128,
            "n": 6144,
            "k": 2560,
            "dtype": "bf16",
            "tp": 1,
            "execution_mode": mode,
        },
        "metrics": {
            "latency_us_p50": 100.0,
            "latency_us_p10": 90.0,
            "latency_us_p90": 120.0,
            "used_cuda_graph": mode == "cudagraph",
            "n_warmups": 3,
            "n_iters": 10,
        },
        "metadata": {
            "source_profiles": ["qwen3_4b"],
            "fallback_reason": None,
        },
        "schema_version": "collector-v2",
    }


def _attention_record() -> dict:
    return {
        "case_id": "attention__prefill",
        "op_kind": "attention",
        "framework": "vllm",
        "framework_version": "0.19.1",
        "device": "NVIDIA GeForce RTX 4090",
        "execution_mode": "eager",
        "kernel_source": "vllm_flash_attn",
        "params": {
            "phase": "prefill",
            "batch_size": 1,
            "isl": 128,
            "kv_prefill": 0,
            "n_decode": 0,
            "kv_decode": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype": "bf16",
            "tp": 1,
            "execution_mode": "eager",
        },
        "metrics": {
            "latency_us_p50": 20.0,
            "latency_us_p10": 18.0,
            "latency_us_p90": 24.0,
            "used_cuda_graph": False,
            "n_warmups": 3,
            "n_iters": 10,
        },
        "metadata": {"source_profiles": ["qwen3_4b"]},
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
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        text=True,
        capture_output=True,
    )


def test_gemm_report_writes_csv_and_jsonl(tmp_path: Path):
    partition = tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"
    _write_jsonl(partition / "gemm.jsonl", [_gemm_record()])
    csv_path = tmp_path / "gap.csv"
    jsonl_path = tmp_path / "gap.jsonl"

    result = _run_report(
        tmp_path,
        "--op-kind", "gemm",
        "--csv", str(csv_path),
        "--jsonl", str(jsonl_path),
    )

    assert "qkv_proj" in result.stdout
    assert "Top 12 GEMM roofline gap cases" in result.stdout
    assert "GEMM roofline gap by (op_subtype, m, execution_mode)" in result.stdout
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["op_kind"] == "gemm"
    assert row["op_subtype"] == "qkv_proj"
    assert row["execution_mode"] == "eager"
    assert float(row["measured_us_p50"]) > 0
    assert float(row["roofline_us"]) > 0
    assert float(row["roofline_gap"]) == float(row["measured_us_p50"]) / float(row["roofline_us"])
    assert row["source_profiles"] == "qwen3_4b"

    json_rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert json_rows[0]["case_id"] == "gemm__qkv_proj_eager"


def test_eager_and_cudagraph_modes_are_preserved(tmp_path: Path):
    partition = tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"
    _write_jsonl(partition / "gemm.jsonl", [
        _gemm_record("eager"),
        _gemm_record("cudagraph"),
    ])
    csv_path = tmp_path / "gap.csv"

    _run_report(tmp_path, "--op-kind", "gemm", "--csv", str(csv_path))

    rows = list(csv.DictReader(csv_path.open()))
    assert {row["execution_mode"] for row in rows} == {"eager", "cudagraph"}
    assert all(row["status"] == "ok" for row in rows)


def test_shape_analysis_prints_eager_graph_pair_delta(tmp_path: Path):
    partition = tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"
    eager = _gemm_record("eager")
    graph = _gemm_record("cudagraph")
    eager["metrics"]["latency_us_p50"] = 120.0
    graph["metrics"]["latency_us_p50"] = 80.0
    _write_jsonl(partition / "gemm.jsonl", [eager, graph])

    result = _run_report(tmp_path, "--op-kind", "gemm", "--top-n", "3")

    assert "Top 3 eager vs cudagraph paired deltas" in result.stdout
    assert "delta_us" in result.stdout
    assert "40.000" in result.stdout


def test_all_compares_non_gemm_roofline(tmp_path: Path):
    partition = tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"
    _write_jsonl(partition / "gemm.jsonl", [_gemm_record()])
    _write_jsonl(partition / "attention.jsonl", [_attention_record()])
    csv_path = tmp_path / "gap.csv"

    result = _run_report(tmp_path, "--op-kind", "all", "--csv", str(csv_path))

    assert "unsupported=0" in result.stdout.splitlines()[0]
    rows = list(csv.DictReader(csv_path.open()))
    by_kind = {row["op_kind"]: row for row in rows}
    assert by_kind["gemm"]["status"] == "ok"
    assert by_kind["attention"]["status"] == "ok"
    assert float(by_kind["attention"]["roofline_us"]) > 0
    assert float(by_kind["attention"]["roofline_gap"]) > 0
