"""Collector schemas — round-trip + invariants."""
from __future__ import annotations

import json

import pytest

from collector.schemas import (
    SCHEMA_VERSION,
    Case,
    CheckpointState,
    CollectorEntry,
    ErrorRecord,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    ProgressEntry,
    RawRecord,
    VersionRoute,
)


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

def test_case_id_is_deterministic():
    """同样 params 必须出同 case_id."""
    a = Case.make(OpKind.GEMM, {"m": 128, "n": 4096, "k": 2048, "dtype": "bf16"})
    b = Case.make(OpKind.GEMM, {"m": 128, "n": 4096, "k": 2048, "dtype": "bf16"})
    assert a.case_id == b.case_id


def test_case_id_independent_of_dict_order():
    """dict key 顺序不影响 case_id (json sort_keys)."""
    a = Case.make(OpKind.GEMM, {"m": 128, "n": 4096, "k": 2048})
    b = Case.make(OpKind.GEMM, {"k": 2048, "n": 4096, "m": 128})
    assert a.case_id == b.case_id


def test_case_id_changes_with_params():
    """不同 params 必须出不同 case_id."""
    a = Case.make(OpKind.GEMM, {"m": 128})
    b = Case.make(OpKind.GEMM, {"m": 256})
    assert a.case_id != b.case_id


def test_case_id_prefix_in_id():
    c = Case.make(OpKind.GEMM, {"m": 1}, prefix="qkv")
    assert "qkv" in c.case_id
    assert c.case_id.startswith("gemm__")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_serialization():
    m = Metrics(
        latency_us_p50=10.5,
        latency_us_p10=10.0,
        latency_us_p90=11.0,
        used_cuda_graph=True,
        n_warmups=3,
        n_iters=10,
    )
    # asdict 走得通
    from dataclasses import asdict
    d = asdict(m)
    assert d["latency_us_p50"] == 10.5
    assert d["power_w"] is None


# ---------------------------------------------------------------------------
# RawRecord round-trip
# ---------------------------------------------------------------------------

def _sample_record() -> RawRecord:
    return RawRecord(
        case_id="gemm__qkv__abc123",
        op_kind=OpKind.GEMM,
        framework=Framework.VLLM,
        framework_version="0.19.1",
        device="NVIDIA GeForce RTX 4090",
        execution_mode=ExecutionMode.EAGER,
        kernel_source="vllm_row_parallel_linear",
        params={"m": 128, "n": 4096, "k": 2048, "dtype": "bf16"},
        metrics=Metrics(
            latency_us_p50=120.5,
            latency_us_p10=118.0,
            latency_us_p90=125.0,
            used_cuda_graph=False,
            n_warmups=3,
            n_iters=10,
        ),
        metadata={
            "worker_id": 0, "git_sha": "abc123",
            "source_profiles": ["qwen3_4b"],   # provenance, 不进查询键
        },
    )


def test_raw_record_round_trip():
    """to_json_dict → json.dumps → json.loads → from_json_dict 必须等价."""
    r = _sample_record()
    s = json.dumps(r.to_json_dict())
    d = json.loads(s)
    r2 = RawRecord.from_json_dict(d)
    assert r2.case_id == r.case_id
    assert r2.op_kind == r.op_kind
    assert r2.framework == r.framework
    assert r2.execution_mode == r.execution_mode
    assert r2.params == r.params
    assert r2.metrics.latency_us_p50 == r.metrics.latency_us_p50
    assert r2.metadata == r.metadata


def test_raw_record_enum_to_string():
    """JSON 输出里 enum 必须是 str (不是 OpKind.GEMM)."""
    d = _sample_record().to_json_dict()
    assert d["op_kind"] == "gemm"
    assert d["framework"] == "vllm"
    assert d["execution_mode"] == "eager"
    # 必须能用 json.dumps (无 enum 残留)
    json.dumps(d)


def test_raw_record_schema_version_present():
    d = _sample_record().to_json_dict()
    assert d["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# ErrorRecord
# ---------------------------------------------------------------------------

def test_error_record_round_trip():
    e = ErrorRecord(
        case_id="moe__hash",
        op_kind=OpKind.MOE,
        framework=Framework.VLLM,
        error_type="CudaOOM",
        error_message="out of memory: tried to allocate ...",
        traceback="Traceback (most recent call last):\n  ...",
    )
    d = e.to_json_dict()
    json.dumps(d)  # JSON-serializable
    e2 = ErrorRecord.from_json_dict(d)
    assert e2.case_id == e.case_id
    assert e2.error_type == e.error_type


# ---------------------------------------------------------------------------
# CheckpointState
# ---------------------------------------------------------------------------

def test_checkpoint_state_round_trip():
    cp = CheckpointState(
        framework=Framework.VLLM,
        op_kind=OpKind.GEMM,
        done={"gemm__a", "gemm__b"},
        failed={"gemm__c"},
        updated_at="2026-05-19T12:00:00+0000",
    )
    d = cp.to_json_dict()
    json.dumps(d)
    cp2 = CheckpointState.from_json_dict(d)
    assert cp2.done == cp.done
    assert cp2.failed == cp.failed


def test_checkpoint_state_serialized_sets_sorted():
    """JSON 输出里 done/failed 必须排序 (diff-friendly)."""
    cp = CheckpointState(
        framework=Framework.VLLM,
        op_kind=OpKind.GEMM,
        done={"z", "a", "m"},
    )
    d = cp.to_json_dict()
    assert d["done"] == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# ProgressEntry
# ---------------------------------------------------------------------------

def test_progress_entry_to_json_dict():
    p = ProgressEntry(
        framework=Framework.VLLM,
        op_kind=OpKind.GEMM,
        total=200,
        done=198,
        failed=2,
    )
    d = p.to_json_dict()
    assert d["op_kind"] == "gemm"
    assert d["total"] == 200
    json.dumps(d)


# ---------------------------------------------------------------------------
# CollectorEntry + VersionRoute
# ---------------------------------------------------------------------------

def test_collector_entry_construct():
    """registry entry 字段齐全."""
    e = CollectorEntry(
        op=OpKind.GEMM,
        framework=Framework.VLLM,
        get_cases_module="collector.cases.qwen3_4b:get_gemm_cases",
        run_case_module="collector.runners.vllm_gemm",
        output_file="gemm.jsonl",
        versions=(VersionRoute("0.19.0", "collector.runners.vllm_gemm"),),
    )
    assert e.op == OpKind.GEMM
    assert e.framework == Framework.VLLM
    assert len(e.versions) == 1
    assert not e.multi_gpu


def test_collector_entry_multi_gpu_flag():
    """collective op 默认 multi_gpu=True 走 distributed/."""
    e = CollectorEntry(
        op=OpKind.COLLECTIVE,
        framework=Framework.VLLM,
        get_cases_module="...",
        run_case_module="collector.distributed.run_collective",
        output_file="nccl.jsonl",
        multi_gpu=True,
    )
    assert e.multi_gpu


# ---------------------------------------------------------------------------
# Enum invariants
# ---------------------------------------------------------------------------

def test_op_kind_values():
    """OpKind value 必须跟 文档/output 文件名对齐."""
    assert OpKind.GEMM.value == "gemm"
    assert OpKind.ATTENTION.value == "attention"
    assert OpKind.MOE.value == "moe"
    assert OpKind.COLLECTIVE.value == "collective"


def test_execution_mode_values():
    assert ExecutionMode.EAGER.value == "eager"
    assert ExecutionMode.CUDAGRAPH.value == "cudagraph"


def test_framework_values():
    assert Framework.VLLM.value == "vllm"
    assert Framework.SGLANG.value == "sglang"
