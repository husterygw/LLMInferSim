"""checkpoint.py — per-op resume state."""
from __future__ import annotations

import json

import pytest

from collector.checkpoint import (
    filter_cases,
    load,
    mark_done,
    mark_failed,
    save,
)
from collector.schemas import (
    Case,
    CheckpointState,
    Framework,
    OpKind,
)


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------

class TestLoadSave:
    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "missing.json"
        s = load(path, Framework.VLLM, OpKind.GEMM)
        assert s.framework == Framework.VLLM
        assert s.op_kind == OpKind.GEMM
        assert s.done == set()
        assert s.failed == set()

    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "ckpt.json"
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            done={"a", "b"}, failed={"c"},
        )
        save(path, s)
        loaded = load(path, Framework.VLLM, OpKind.GEMM)
        assert loaded.done == {"a", "b"}
        assert loaded.failed == {"c"}
        assert loaded.updated_at != ""

    def test_save_atomic_via_tmp_rename(self, tmp_path):
        """save 用 .tmp 文件 + rename, 中途 crash 不会留半截."""
        path = tmp_path / "ckpt.json"
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM, done={"a"},
        )
        save(path, s)
        # tmp 应该已 rename 不剩
        tmp_file = path.with_suffix(path.suffix + ".tmp")
        assert not tmp_file.exists()
        assert path.exists()

    def test_load_corrupted_returns_empty(self, tmp_path):
        """损坏的 JSON 不让 scheduler 挂, 返空 state."""
        path = tmp_path / "bad.json"
        path.write_text("{ this is not valid json")
        s = load(path, Framework.VLLM, OpKind.GEMM)
        assert s.done == set()

    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "ckpt.json"
        s = CheckpointState(framework=Framework.VLLM, op_kind=OpKind.GEMM)
        save(path, s)
        assert path.exists()


# ---------------------------------------------------------------------------
# mark_done / mark_failed
# ---------------------------------------------------------------------------

class TestMark:
    def test_mark_done_adds_to_set(self, tmp_path):
        path = tmp_path / "ckpt.json"
        s = CheckpointState(framework=Framework.VLLM, op_kind=OpKind.GEMM)
        mark_done(path, s, "case_a")
        assert "case_a" in s.done
        # 持久化
        loaded = load(path, Framework.VLLM, OpKind.GEMM)
        assert "case_a" in loaded.done

    def test_mark_failed_adds_to_set(self, tmp_path):
        path = tmp_path / "ckpt.json"
        s = CheckpointState(framework=Framework.VLLM, op_kind=OpKind.GEMM)
        mark_failed(path, s, "case_b")
        loaded = load(path, Framework.VLLM, OpKind.GEMM)
        assert "case_b" in loaded.failed

    def test_mark_done_removes_from_failed(self, tmp_path):
        """case 之前 fail, 后来重跑成功, done 加, failed 应该清."""
        path = tmp_path / "ckpt.json"
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            failed={"case_x"},
        )
        mark_done(path, s, "case_x")
        loaded = load(path, Framework.VLLM, OpKind.GEMM)
        assert "case_x" in loaded.done
        assert "case_x" not in loaded.failed

    def test_mark_failed_does_not_touch_done(self, tmp_path):
        """case 之前 done, 不应该被 mark_failed 推倒."""
        path = tmp_path / "ckpt.json"
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            done={"case_y"},
        )
        mark_failed(path, s, "case_z")
        loaded = load(path, Framework.VLLM, OpKind.GEMM)
        assert "case_y" in loaded.done
        assert "case_z" in loaded.failed


# ---------------------------------------------------------------------------
# filter_cases
# ---------------------------------------------------------------------------

def _cases(n: int) -> list[Case]:
    return [Case.make(OpKind.GEMM, {"i": i}) for i in range(n)]


class TestFilterCases:
    def test_no_state_returns_all(self):
        cases = _cases(5)
        s = CheckpointState(framework=Framework.VLLM, op_kind=OpKind.GEMM)
        assert filter_cases(s, cases) == cases

    def test_done_skipped(self):
        cases = _cases(5)
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            done={cases[0].case_id, cases[2].case_id},
        )
        kept = filter_cases(s, cases)
        kept_ids = {c.case_id for c in kept}
        assert cases[0].case_id not in kept_ids
        assert cases[2].case_id not in kept_ids
        assert cases[1].case_id in kept_ids
        assert len(kept) == 3

    def test_failed_skipped_by_default(self):
        cases = _cases(5)
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            failed={cases[3].case_id},
        )
        kept = filter_cases(s, cases)
        assert cases[3].case_id not in {c.case_id for c in kept}

    def test_retry_failed_includes_failed(self):
        cases = _cases(5)
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            done={cases[0].case_id},
            failed={cases[3].case_id},
        )
        kept = filter_cases(s, cases, retry_failed=True)
        kept_ids = {c.case_id for c in kept}
        # done 还是跳, failed 现在不跳
        assert cases[0].case_id not in kept_ids
        assert cases[3].case_id in kept_ids

    def test_order_preserved(self):
        """filter 保留输入顺序 (resume 后续 case 进度可预期)."""
        cases = _cases(5)
        s = CheckpointState(
            framework=Framework.VLLM, op_kind=OpKind.GEMM,
            done={cases[1].case_id, cases[3].case_id},
        )
        kept = filter_cases(s, cases)
        assert [c.case_id for c in kept] == [
            cases[0].case_id, cases[2].case_id, cases[4].case_id,
        ]


# ---------------------------------------------------------------------------
# JSON 输出格式稳定性
# ---------------------------------------------------------------------------

def test_save_produces_sorted_lists(tmp_path):
    """JSON 输出里 done/failed 排序, 便于 diff."""
    path = tmp_path / "ckpt.json"
    s = CheckpointState(
        framework=Framework.VLLM, op_kind=OpKind.GEMM,
        done={"z", "a", "m"}, failed={"q", "b"},
    )
    save(path, s)
    d = json.loads(path.read_text())
    assert d["done"] == ["a", "m", "z"]
    assert d["failed"] == ["b", "q"]
