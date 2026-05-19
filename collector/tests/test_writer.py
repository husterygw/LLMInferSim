"""writer.py — JSONL append + 多进程安全."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from collector.schemas import (
    ErrorRecord,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    ProgressEntry,
    RawRecord,
)
from collector.writer import (
    append_error,
    append_progress,
    append_record,
    read_errors,
    read_records,
)


def _make_record(case_id: str = "case_a") -> RawRecord:
    return RawRecord(
        case_id=case_id,
        op_kind=OpKind.GEMM,
        framework=Framework.VLLM,
        framework_version="0.19.1",
        device="RTX 4090",
        execution_mode=ExecutionMode.EAGER,
        kernel_source="vllm_row_parallel_linear",
        params={"m": 128, "n": 4096, "k": 2048, "dtype": "bf16"},
        metrics=Metrics(
            latency_us_p50=100.0, latency_us_p10=99.0, latency_us_p90=101.0,
            used_cuda_graph=False, n_warmups=3, n_iters=10,
        ),
        metadata={"worker_id": 0},
    )


class TestAppendRecord:
    def test_writes_single_line(self, tmp_path):
        path = tmp_path / "gemm.jsonl"
        append_record(path, _make_record("case_a"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["case_id"] == "case_a"
        assert d["op_kind"] == "gemm"

    def test_appends_multiple(self, tmp_path):
        path = tmp_path / "gemm.jsonl"
        for i in range(5):
            append_record(path, _make_record(f"case_{i}"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "subdir" / "gemm.jsonl"
        append_record(path, _make_record("case_a"))
        assert path.exists()

    def test_read_round_trip(self, tmp_path):
        path = tmp_path / "gemm.jsonl"
        records_in = [_make_record(f"case_{i}") for i in range(3)]
        for r in records_in:
            append_record(path, r)
        records_out = read_records(path)
        assert len(records_out) == 3
        assert [r.case_id for r in records_out] == ["case_0", "case_1", "case_2"]

    def test_read_missing_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        assert read_records(path) == []


class TestAppendError:
    def test_writes_error_line(self, tmp_path):
        path = tmp_path / "errors" / "gemm.jsonl"
        e = ErrorRecord(
            case_id="case_x",
            op_kind=OpKind.GEMM,
            framework=Framework.VLLM,
            error_type="CudaOOM",
            error_message="out of memory",
        )
        append_error(path, e)
        errs = read_errors(path)
        assert len(errs) == 1
        assert errs[0].error_type == "CudaOOM"

    def test_errors_independent_of_main(self, tmp_path):
        """errors/<op>.jsonl 跟 <op>.jsonl 互不影响."""
        main_path = tmp_path / "gemm.jsonl"
        err_path = tmp_path / "errors" / "gemm.jsonl"
        append_record(main_path, _make_record("ok_case"))
        append_error(err_path, ErrorRecord(
            case_id="bad_case", op_kind=OpKind.GEMM, framework=Framework.VLLM,
            error_type="X", error_message="...",
        ))
        assert len(read_records(main_path)) == 1
        assert len(read_errors(err_path)) == 1


class TestAppendProgress:
    def test_progress_append_only(self, tmp_path):
        """progress.jsonl 是 append-only 审计日志, 每次 update 一行."""
        path = tmp_path / "progress.jsonl"
        for done in [10, 20, 50, 100]:
            append_progress(path, ProgressEntry(
                framework=Framework.VLLM, op_kind=OpKind.GEMM,
                total=100, done=done, failed=0,
            ))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 4
        # 最后一行 done=100
        last = json.loads(lines[-1])
        assert last["done"] == 100


# ---------------------------------------------------------------------------
# 多进程并发 (multiprocessing)
# ---------------------------------------------------------------------------

def _worker_write(args):
    """子进程: 写 N 条 record 进同一个 path."""
    path, n_records, worker_id = args
    from collector.writer import append_record
    from collector.schemas import (
        RawRecord, OpKind, Framework, ExecutionMode, Metrics,
    )
    for i in range(n_records):
        r = RawRecord(
            case_id=f"w{worker_id}_c{i}",
            op_kind=OpKind.GEMM,
            framework=Framework.VLLM,
            framework_version="0.19.1",
            device="RTX 4090",
            execution_mode=ExecutionMode.EAGER,
            kernel_source="vllm",
            params={"i": i, "worker": worker_id},
            metrics=Metrics(
                latency_us_p50=1.0, latency_us_p10=1.0, latency_us_p90=1.0,
                used_cuda_graph=False, n_warmups=1, n_iters=1,
            ),
        )
        append_record(Path(path), r)


@pytest.mark.skipif(
    os.environ.get("PYTEST_DISABLE_MP") == "1",
    reason="mp test disabled in this env",
)
def test_concurrent_writes_no_corruption(tmp_path):
    """4 个进程并发往同一 jsonl 写, 总行数 = N×K 且每行可解析."""
    path = tmp_path / "gemm.jsonl"
    n_workers = 4
    n_records = 10

    # 用 spawn (跟 CUDA 上下文兼容), fork 也行但 spawn 更安全
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        pool.map(
            _worker_write,
            [(str(path), n_records, i) for i in range(n_workers)],
        )

    records = read_records(path)
    assert len(records) == n_workers * n_records   # 没丢
    # 每条都能解析回 RawRecord (说明没 partial write 损坏)
    case_ids = {r.case_id for r in records}
    assert len(case_ids) == n_workers * n_records   # 每个 case_id 唯一
