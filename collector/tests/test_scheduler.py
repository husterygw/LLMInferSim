"""scheduler.py — 编排核心 (resume + 错误隔离 + dry_run + limit + multi_gpu skip)."""
from __future__ import annotations

import sys
import types

import pytest

from collector import checkpoint, scheduler, writer
from collector.paths import DataPaths
from collector.schemas import (
    Case,
    CollectorEntry,
    ErrorRecord,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    RawRecord,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_record(case_id: str, op: OpKind = OpKind.GEMM) -> RawRecord:
    return RawRecord(
        case_id=case_id,
        op_kind=op,
        framework=Framework.VLLM,
        framework_version="0.19.1",
        device="RTX 4090",
        execution_mode=ExecutionMode.EAGER,
        kernel_source="vllm_test",
        params={"placeholder": True},
        metrics=Metrics(
            latency_us_p50=100.0, latency_us_p10=99.0, latency_us_p90=101.0,
            used_cuda_graph=False, n_warmups=3, n_iters=10,
        ),
    )


def _make_cases(n: int, op: OpKind = OpKind.GEMM) -> list[Case]:
    return [Case.make(op, {"i": i}) for i in range(n)]


def _register_get_cases_module(monkeypatch, mod_name: str, cases: list[Case]) -> str:
    """动态创建一个 fake module 提供 get_cases(profiles), 返 'mod:get_cases' spec.

    生产 get_cases 签名: `get_cases(profiles) -> (cases, sources)`.
    测试 mock 忽略 profiles, 返固定 (cases, {}).
    """
    mod = types.ModuleType(mod_name)
    mod.get_cases = lambda profiles: (cases, {})   # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, mod_name, mod)
    return f"{mod_name}:get_cases"


def _entry(cases_spec: str, multi_gpu: bool = False) -> CollectorEntry:
    return CollectorEntry(
        op=OpKind.GEMM,
        framework=Framework.VLLM,
        get_cases_module=cases_spec,
        run_case_module="collector.runners.vllm_gemm",
        output_file="gemm.jsonl",
        multi_gpu=multi_gpu,
    )


@pytest.fixture
def paths(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    return p


# ---------------------------------------------------------------------------
# _resolve_callable
# ---------------------------------------------------------------------------

class TestResolveCallable:
    def test_module_colon_func(self, monkeypatch):
        mod = types.ModuleType("test_mod_a")
        mod.my_func = lambda: "hello"   # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "test_mod_a", mod)
        fn = scheduler._resolve_callable("test_mod_a:my_func")
        assert fn() == "hello"

    def test_missing_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            scheduler._resolve_callable("definitely_not_a_module:func")

    def test_missing_func_raises(self, monkeypatch):
        mod = types.ModuleType("test_mod_b")
        monkeypatch.setitem(sys.modules, "test_mod_b", mod)
        with pytest.raises(AttributeError):
            scheduler._resolve_callable("test_mod_b:no_such_func")


# ---------------------------------------------------------------------------
# _run_one_case
# ---------------------------------------------------------------------------

class TestRunOneCase:
    def test_success_returns_record(self):
        case = Case.make(OpKind.GEMM, {"i": 0})
        run = lambda c, d: _make_record(c.case_id)  # noqa: E731
        rec, err = scheduler._run_one_case(case, 0, Framework.VLLM, run)
        assert rec is not None
        assert err is None
        assert rec.case_id == case.case_id

    def test_exception_returns_error_record(self):
        case = Case.make(OpKind.GEMM, {"i": 0})

        def run(c, d):
            raise RuntimeError("boom!")

        rec, err = scheduler._run_one_case(case, 0, Framework.VLLM, run)
        assert rec is None
        assert err is not None
        assert err.case_id == case.case_id
        assert err.error_type == "RuntimeError"
        assert "boom" in err.error_message
        assert "Traceback" in err.traceback


# ---------------------------------------------------------------------------
# run_op — main orchestration
# ---------------------------------------------------------------------------

class TestRunOpSuccess:
    def test_all_succeed(self, monkeypatch, paths):
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_succ", cases)
        entry = _entry(spec)

        run_calls = []

        def run(c, d):
            run_calls.append(c.case_id)
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        assert result.total_cases == 5
        assert result.done == 5
        assert result.failed == 0
        assert not result.skipped_multi_gpu
        assert len(run_calls) == 5

        # 数据真落盘
        records = writer.read_records(paths.op_jsonl(OpKind.GEMM))
        assert len(records) == 5

        # checkpoint 标记
        state = checkpoint.load(
            paths.checkpoint_json(OpKind.GEMM), Framework.VLLM, OpKind.GEMM,
        )
        assert len(state.done) == 5


class TestRunOpFailures:
    def test_some_fail(self, monkeypatch, paths):
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_fail", cases)
        entry = _entry(spec)

        def run(c, d):
            i = c.params["i"]
            if i in (1, 3):
                raise RuntimeError(f"fail_{i}")
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        assert result.done == 3
        assert result.failed == 2

        # successful 写到 main, fail 写到 errors
        records = writer.read_records(paths.op_jsonl(OpKind.GEMM))
        errors = writer.read_errors(paths.errors_jsonl(OpKind.GEMM))
        assert len(records) == 3
        assert len(errors) == 2

        # checkpoint
        state = checkpoint.load(
            paths.checkpoint_json(OpKind.GEMM), Framework.VLLM, OpKind.GEMM,
        )
        assert len(state.done) == 3
        assert len(state.failed) == 2

    def test_failure_does_not_stop_remaining(self, monkeypatch, paths):
        """前面失败不阻塞后面 case."""
        cases = _make_cases(10)
        spec = _register_get_cases_module(monkeypatch, "tmod_fail2", cases)
        entry = _entry(spec)

        def run(c, d):
            if c.params["i"] == 0:
                raise RuntimeError("first fails")
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        assert result.done == 9
        assert result.failed == 1


class TestRunOpResume:
    def test_resume_skips_done(self, monkeypatch, paths, tmp_path):
        """先跑 3 个 → checkpoint 标 done → 重跑应该跳过."""
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_resume", cases)
        entry = _entry(spec)

        # 预先 mark done 前 3 个
        cp_path = paths.checkpoint_json(OpKind.GEMM)
        paths.ensure_dirs()
        from collector.schemas import CheckpointState
        state = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            done={cases[0].case_id, cases[1].case_id, cases[2].case_id},
        )
        checkpoint.save(cp_path, state)

        run_calls = []

        def run(c, d):
            run_calls.append(c.case_id)
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        # 只应跑后 2 个
        assert result.total_cases == 2
        assert result.done == 2
        assert len(run_calls) == 2
        assert run_calls[0] == cases[3].case_id
        assert run_calls[1] == cases[4].case_id

    def test_retry_failed_reruns_failed(self, monkeypatch, paths):
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_retry", cases)
        entry = _entry(spec)

        # 先 mark failed
        cp_path = paths.checkpoint_json(OpKind.GEMM)
        paths.ensure_dirs()
        from collector.schemas import CheckpointState
        state = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            failed={cases[1].case_id},
        )
        checkpoint.save(cp_path, state)

        run_calls = []

        def run(c, d):
            run_calls.append(c.case_id)
            return _make_record(c.case_id)

        # 默认 retry_failed=False, failed 跳过 → 跑 4 个
        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        assert result.total_cases == 4
        assert cases[1].case_id not in run_calls

    def test_retry_failed_true(self, monkeypatch, paths):
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_retry2", cases)
        entry = _entry(spec)

        cp_path = paths.checkpoint_json(OpKind.GEMM)
        paths.ensure_dirs()
        from collector.schemas import CheckpointState
        state = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            failed={cases[1].case_id},
        )
        checkpoint.save(cp_path, state)

        run_calls = []

        def run(c, d):
            run_calls.append(c.case_id)
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[], retry_failed=True)
        assert result.total_cases == 5
        assert cases[1].case_id in run_calls


class TestRunOpLimit:
    def test_limit_caps_total(self, monkeypatch, paths):
        cases = _make_cases(10)
        spec = _register_get_cases_module(monkeypatch, "tmod_limit", cases)
        entry = _entry(spec)

        def run(c, d):
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[], limit=3)
        assert result.total_cases == 3
        assert result.done == 3


class TestRunOpDryRun:
    def test_dry_run_counts_but_no_files(self, monkeypatch, paths):
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_dry", cases)
        entry = _entry(spec)

        run_calls = []

        def run(c, d):
            run_calls.append(c.case_id)
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[], dry_run=True)
        assert result.total_cases == 5
        assert result.done == 0      # 没真跑
        assert len(run_calls) == 0
        # 没产 jsonl
        assert not paths.op_jsonl(OpKind.GEMM).exists()
        # checkpoint 也不动
        assert not paths.checkpoint_json(OpKind.GEMM).exists()


class TestRunOpMultiGpu:
    def test_multi_gpu_skipped(self, monkeypatch, paths):
        cases = _make_cases(5)
        spec = _register_get_cases_module(monkeypatch, "tmod_mgpu", cases)
        entry = _entry(spec, multi_gpu=True)

        run_calls = []

        def run(c, d):
            run_calls.append(c.case_id)
            return _make_record(c.case_id)

        result = scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        assert result.skipped_multi_gpu
        assert result.total_cases == 0
        assert len(run_calls) == 0


class TestCaseFilter:
    def test_case_filter_excludes(self, monkeypatch, paths):
        cases = _make_cases(10)
        spec = _register_get_cases_module(monkeypatch, "tmod_filter", cases)
        entry = _entry(spec)

        def run(c, d):
            return _make_record(c.case_id)

        # filter: 只跑 i % 2 == 0
        result = scheduler.run_op(
            entry, paths, "0.19.1", run,
            profiles=[],
            case_filter=lambda c: c.params["i"] % 2 == 0,
        )
        assert result.total_cases == 5
        assert result.done == 5


# ---------------------------------------------------------------------------
# run_all — 跨多 entries
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_iterates_entries(self, monkeypatch, paths):
        gemm_cases = _make_cases(3, OpKind.GEMM)
        moe_cases = _make_cases(2, OpKind.MOE)

        spec_g = _register_get_cases_module(monkeypatch, "tmod_runall_gemm", gemm_cases)
        spec_m = _register_get_cases_module(monkeypatch, "tmod_runall_moe", moe_cases)

        entries = [
            _entry(spec_g),
            CollectorEntry(
                op=OpKind.MOE, framework=Framework.VLLM,
                get_cases_module=spec_m,
                run_case_module="x",
                output_file="moe.jsonl",
            ),
        ]

        def run(c, d):
            return _make_record(c.case_id, op=c.op_kind)

        results = scheduler.run_all(entries, paths, "0.19.1", run, profiles=[])
        assert (OpKind.GEMM, Framework.VLLM) in results
        assert (OpKind.MOE, Framework.VLLM) in results
        assert results[(OpKind.GEMM, Framework.VLLM)].done == 3
        assert results[(OpKind.MOE, Framework.VLLM)].done == 2

    def test_one_op_multi_gpu_skipped_others_run(self, monkeypatch, paths):
        gemm_cases = _make_cases(3, OpKind.GEMM)
        coll_cases = _make_cases(2, OpKind.COLLECTIVE)

        spec_g = _register_get_cases_module(monkeypatch, "tmod_runall_gemm2", gemm_cases)
        spec_c = _register_get_cases_module(monkeypatch, "tmod_runall_coll", coll_cases)

        entries = [
            _entry(spec_g),    # 普通
            CollectorEntry(
                op=OpKind.COLLECTIVE, framework=Framework.VLLM,
                get_cases_module=spec_c,
                run_case_module="x",
                output_file="nccl.jsonl",
                multi_gpu=True,
            ),
        ]

        def run(c, d):
            return _make_record(c.case_id, op=c.op_kind)

        results = scheduler.run_all(entries, paths, "0.19.1", run, profiles=[])
        assert results[(OpKind.GEMM, Framework.VLLM)].done == 3
        assert results[(OpKind.COLLECTIVE, Framework.VLLM)].skipped_multi_gpu


# ---------------------------------------------------------------------------
# Progress log
# ---------------------------------------------------------------------------

class TestSourceProfilesInjection:
    """source_profiles 写入 record.metadata."""

    def test_metadata_gets_source_profiles(self, monkeypatch, paths):
        cases = _make_cases(3)
        # mock get_cases 返 (cases, sources)
        sources = {cases[0].case_id: ["prof_a"], cases[1].case_id: ["prof_a", "prof_b"]}
        mod = types.ModuleType("tmod_src")
        mod.get_cases = lambda p: (cases, sources)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tmod_src", mod)
        entry = _entry("tmod_src:get_cases")

        def run(c, d):
            return _make_record(c.case_id)

        scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        records = writer.read_records(paths.op_jsonl(OpKind.GEMM))
        # 按 case_id 索引
        recs_by_id = {r.case_id: r for r in records}
        assert recs_by_id[cases[0].case_id].metadata["source_profiles"] == ["prof_a"]
        assert recs_by_id[cases[1].case_id].metadata["source_profiles"] == ["prof_a", "prof_b"]
        # cases[2] 没在 sources 里 → metadata 不应有 source_profiles 字段
        assert "source_profiles" not in recs_by_id[cases[2].case_id].metadata

    def test_error_record_also_gets_source_profiles(self, monkeypatch, paths):
        cases = _make_cases(2)
        sources = {cases[0].case_id: ["prof_x"]}
        mod = types.ModuleType("tmod_src_err")
        mod.get_cases = lambda p: (cases, sources)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tmod_src_err", mod)
        entry = _entry("tmod_src_err:get_cases")

        def run(c, d):
            raise RuntimeError("boom")

        scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        errors = writer.read_errors(paths.errors_jsonl(OpKind.GEMM))
        errs_by_id = {e.case_id: e for e in errors}
        assert errs_by_id[cases[0].case_id].metadata["source_profiles"] == ["prof_x"]


class TestProgress:
    def test_progress_written_after_run(self, monkeypatch, paths):
        cases = _make_cases(3)
        spec = _register_get_cases_module(monkeypatch, "tmod_progress", cases)
        entry = _entry(spec)

        def run(c, d):
            return _make_record(c.case_id)

        scheduler.run_op(entry, paths, "0.19.1", run, profiles=[])
        prog_lines = paths.progress_jsonl.read_text().strip().splitlines()
        assert len(prog_lines) == 1
        import json
        d = json.loads(prog_lines[0])
        assert d["total"] == 3
        assert d["done"] == 3
