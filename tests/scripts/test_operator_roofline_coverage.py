"""Real collector JSONL coverage: every collected operator gets roofline compare."""
from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_operator_roofline_gap.py"
DB_ROOT = REPO_ROOT / "collector" / "data" / "operator_db"
PARTITION = DB_ROOT / "RTX_4090" / "vllm-0.19.1"
OP_KINDS = ("gemm", "attention", "moe", "collective")

pytestmark = pytest.mark.skipif(
    not PARTITION.exists(),
    reason="real collector JSONL partition not present",
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location("report_operator_roofline_gap", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backend_cache():
    hw = get_hardware_profile("RTX_4090")
    backends: dict[str, RooflineBackend] = {}

    def get(mode: str) -> RooflineBackend:
        if mode not in backends:
            backends[mode] = RooflineBackend(hw, DeployConfig(execution_mode=mode))
        return backends[mode]

    return get


def _assert_ok_gap(row: dict):
    assert row["status"] == "ok", row
    roofline_us = float(row["roofline_us"])
    gap = float(row["roofline_gap"])
    assert math.isfinite(roofline_us) and roofline_us > 0, row
    assert math.isfinite(gap) and gap > 0, row


@pytest.mark.parametrize("op_kind", OP_KINDS)
def test_real_collector_records_all_have_roofline_compare(op_kind):
    path = PARTITION / f"{op_kind}.jsonl"
    if not path.exists():
        pytest.skip(f"{op_kind}.jsonl not present")
    report = _load_report_module()
    records = report.load_records(path, "RTX_4090")
    assert records, f"no records loaded from {path}"
    get_backend = _backend_cache()

    for record in records:
        row = report.estimate_gap(record, get_backend(record.execution_mode))
        _assert_ok_gap(row)


def test_record_factory_minimal_examples_are_estimable():
    report = _load_report_module()
    hw = get_hardware_profile("RTX_4090")
    wanted = {
        "gemm": lambda rec: True,
        "attention_prefill": lambda rec: rec.signature.op_kind == "attention"
        and rec.signature.op_subtype == "prefill",
        "attention_decode": lambda rec: rec.signature.op_kind == "attention"
        and rec.signature.op_subtype == "decode",
        "moe": lambda rec: rec.signature.op_kind == "moe",
        "collective_allreduce": lambda rec: rec.signature.op_kind == "collective"
        and rec.signature.op_subtype == "allreduce",
    }
    found = {}
    for op_kind in OP_KINDS:
        path = PARTITION / f"{op_kind}.jsonl"
        if not path.exists():
            continue
        for record in report.load_records(path, "RTX_4090"):
            for name, predicate in wanted.items():
                if name not in found and predicate(record):
                    found[name] = record
    missing = sorted(set(wanted) - set(found))
    assert not missing, f"missing sample records: {missing}"

    for name, record in found.items():
        op = report.record_to_operator(record, hw=hw)
        backend = RooflineBackend(hw, DeployConfig(execution_mode=record.execution_mode))
        entry = backend.estimate(op)
        assert entry.latency_s > 0, name
        assert math.isfinite(entry.latency_s), name


def test_report_op_kind_all_has_no_unsupported_rows(tmp_path):
    csv_path = tmp_path / "operator_gap_all.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--db-root",
            str(DB_ROOT),
            "--hardware",
            "RTX_4090",
            "--framework",
            "vllm",
            "--framework-version",
            "0.19.1",
            "--op-kind",
            "all",
            "--csv",
            str(csv_path),
            "--top-n",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    first_line = proc.stdout.splitlines()[0].strip()
    assert "unsupported=0" in first_line, first_line
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(row["status"] == "ok" for row in rows)
    for row in rows:
        _assert_ok_gap(row)
