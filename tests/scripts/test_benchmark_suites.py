from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_single_tp1_roofline_suite_generation(tmp_path: Path):
    out = tmp_path / "cases.jsonl"

    _run(sys.executable, "scripts/bench_cases.py", "--suite", "single_tp1_roofline", "--out", str(out))

    cases = _load_jsonl(out)
    assert len(cases) == 9
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["suite"] for case in cases} == {"single_tp1_roofline"}
    assert all(case["concurrency"] == 1 for case in cases)
    assert all(case["num_prompts"] == 1 for case in cases)
    assert all(case["request_rate"] == "inf" for case in cases)
    assert all(case["prefix_cache"] is False for case in cases)
    assert all(case["chunked_prefill"] is False for case in cases)
    assert all(case["max_model_len"] == 8192 for case in cases)
    assert all(case["max_num_batched_tokens"] == 8192 for case in cases)
    assert all(case["max_num_seqs"] is None for case in cases)


def test_batch_tp1_sweep_contains_expected_concurrency(tmp_path: Path):
    out = tmp_path / "cases.jsonl"

    _run(sys.executable, "scripts/bench_cases.py", "--suite", "batch_tp1_sweep", "--out", str(out))

    cases = _load_jsonl(out)
    assert len(cases) == 20
    assert sorted({case["concurrency"] for case in cases}) == [1, 4, 8, 16, 32]
    assert all(case["num_prompts"] == case["concurrency"] for case in cases)


def test_long_context_suite_bounds_vllm_scheduler_limits(tmp_path: Path):
    out = tmp_path / "cases.jsonl"

    _run(sys.executable, "scripts/bench_cases.py", "--suite", "long_context_sweep", "--out", str(out))

    cases = _load_jsonl(out)
    assert cases
    assert all(case["max_model_len"] == 32768 for case in cases)
    assert all(case["max_num_batched_tokens"] >= case["max_model_len"] for case in cases)
    assert all(
        case["max_num_batched_tokens"] >= case["concurrency"] * case["input_len"]
        for case in cases
    )
    assert max(case["input_len"] + case["output_len"] for case in cases) <= 32768


def test_bench_compare_dry_run_prints_server_and_bench_commands(tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    out = tmp_path / "out"
    _run(
        sys.executable,
        "scripts/bench_cases.py",
        "--suite",
        "single_tp1_roofline",
        "--filter-case",
        "*i128_o128*",
        "--out",
        str(cases),
    )

    result = _run(
        "bash",
        "scripts/bench_compare.sh",
        "--cases",
        str(cases),
        "--out",
        str(out),
        "--dry-run",
    )

    assert "SERVER[real]" in result.stdout
    assert "SERVER[sim]" in result.stdout
    assert "BENCH[real]" in result.stdout
    assert "BENCH[sim]" in result.stdout
    assert "--no-enable-prefix-caching" in result.stdout
    assert "--no-enable-chunked-prefill" in result.stdout
    assert "--max-model-len 8192" in result.stdout
    assert "--max-num-seqs" not in result.stdout


def test_run_bench_suite_legacy_alias_dry_run(tmp_path: Path):
    result = _run(
        "bash",
        "scripts/run_bench_suite.sh",
        "A",
        "--filter-case",
        "*i128_o128*",
        "--dry-run",
    )

    assert "suite=single_tp1_roofline" in result.stdout
    assert "BENCH[real]" in result.stdout
