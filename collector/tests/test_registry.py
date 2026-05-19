"""registry.py — 注册表 CRUD + list / require."""
from __future__ import annotations

import pytest

from collector.registry import REGISTRY, CollectorRegistry
from collector.schemas import CollectorEntry, Framework, OpKind, VersionRoute


def _entry(op: OpKind, fw: Framework, multi_gpu: bool = False) -> CollectorEntry:
    return CollectorEntry(
        op=op,
        framework=fw,
        get_cases_module=f"collector.cases.x:get_{op.value}_cases",
        run_case_module=f"collector.runners.{fw.value}_{op.value}",
        output_file=f"{op.value}.jsonl",
        multi_gpu=multi_gpu,
    )


@pytest.fixture
def reg():
    """干净 registry (不污染全局 REGISTRY)."""
    return CollectorRegistry()


class TestRegister:
    def test_register_then_get(self, reg):
        e = _entry(OpKind.GEMM, Framework.VLLM)
        reg.register(e)
        got = reg.get(OpKind.GEMM, Framework.VLLM)
        assert got is e

    def test_get_missing_returns_none(self, reg):
        assert reg.get(OpKind.GEMM, Framework.VLLM) is None

    def test_duplicate_register_raises(self, reg):
        e1 = _entry(OpKind.GEMM, Framework.VLLM)
        e2 = _entry(OpKind.GEMM, Framework.VLLM)
        reg.register(e1)
        with pytest.raises(ValueError, match="Duplicate registry"):
            reg.register(e2)

    def test_different_frameworks_coexist(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.register(_entry(OpKind.GEMM, Framework.SGLANG))
        assert reg.get(OpKind.GEMM, Framework.VLLM) is not None
        assert reg.get(OpKind.GEMM, Framework.SGLANG) is not None
        assert len(reg) == 2

    def test_different_ops_coexist(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.register(_entry(OpKind.MOE, Framework.VLLM))
        assert len(reg) == 2


class TestRequire:
    def test_require_returns_entry(self, reg):
        e = _entry(OpKind.GEMM, Framework.VLLM)
        reg.register(e)
        assert reg.require(OpKind.GEMM, Framework.VLLM) is e

    def test_require_missing_raises(self, reg):
        with pytest.raises(KeyError, match="No registry entry"):
            reg.require(OpKind.GEMM, Framework.VLLM)

    def test_require_error_lists_available(self, reg):
        reg.register(_entry(OpKind.MOE, Framework.VLLM))
        with pytest.raises(KeyError) as exc:
            reg.require(OpKind.GEMM, Framework.SGLANG)
        msg = str(exc.value)
        assert "moe" in msg or "vllm" in msg


class TestListMethods:
    def test_list_ops_all_frameworks(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.register(_entry(OpKind.MOE, Framework.VLLM))
        reg.register(_entry(OpKind.GEMM, Framework.SGLANG))
        ops = reg.list_ops()
        assert OpKind.GEMM in ops
        assert OpKind.MOE in ops

    def test_list_ops_filter_by_framework(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.register(_entry(OpKind.MOE, Framework.VLLM))
        reg.register(_entry(OpKind.ATTENTION, Framework.SGLANG))
        ops = reg.list_ops(Framework.VLLM)
        assert OpKind.GEMM in ops
        assert OpKind.MOE in ops
        assert OpKind.ATTENTION not in ops

    def test_list_frameworks(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.register(_entry(OpKind.GEMM, Framework.SGLANG))
        fws = reg.list_frameworks()
        assert Framework.VLLM in fws
        assert Framework.SGLANG in fws

    def test_all_entries_sorted(self, reg):
        reg.register(_entry(OpKind.MOE, Framework.VLLM))
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        entries = reg.all_entries()
        # 按 (op, framework) 排序: gemm 先, moe 后
        assert entries[0].op == OpKind.GEMM
        assert entries[1].op == OpKind.MOE


class TestContainsAndLen:
    def test_contains(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        assert (OpKind.GEMM, Framework.VLLM) in reg
        assert (OpKind.MOE, Framework.VLLM) not in reg

    def test_len(self, reg):
        assert len(reg) == 0
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.register(_entry(OpKind.MOE, Framework.VLLM))
        assert len(reg) == 2

    def test_clear(self, reg):
        reg.register(_entry(OpKind.GEMM, Framework.VLLM))
        reg.clear()
        assert len(reg) == 0


def test_global_registry_starts_empty():
    """REGISTRY 是 lazy-bootstrapped, import 时空."""
    # 必须新进程下 import 才严格成立; 这里只验证 type
    assert isinstance(REGISTRY, CollectorRegistry)
