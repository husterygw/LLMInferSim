"""operator-DB framework-version precheck (#3)."""
from __future__ import annotations

import pytest

from llm_infer_sim.core.operator_db.version_check import (
    OperatorDBVersionMismatch,
    normalize_version,
    verify_db_framework_version,
)


def test_normalize_strips_build_and_v_prefix():
    assert normalize_version("0.19.1+cu128") == "0.19.1"
    assert normalize_version(" v0.19.1 ") == "0.19.1"
    assert normalize_version(None) is None


def test_match_returns_runtime_version():
    assert verify_db_framework_version("0.19.1", "0.19.1") == "0.19.1"
    # build metadata normalized away
    assert verify_db_framework_version("0.19.1", "0.19.1+cu128") == "0.19.1+cu128"


def test_mismatch_fails_closed_by_default():
    with pytest.raises(OperatorDBVersionMismatch):
        verify_db_framework_version("0.19.1", "0.19.0")


def test_mismatch_ignored_downgrades_to_warning():
    with pytest.warns(UserWarning):
        out = verify_db_framework_version("0.19.1", "0.19.0", ignore_mismatch=True)
    assert out == "0.19.0"


def test_unknown_runtime_warns_and_skips(monkeypatch):
    """When the runtime vLLM version can't be determined: warn, don't raise."""
    monkeypatch.setattr(
        "llm_infer_sim.core.operator_db.version_check.current_runtime_vllm_version",
        lambda: None,
    )
    with pytest.warns(UserWarning):
        out = verify_db_framework_version("0.19.1", None)
    assert out is None


def test_load_partition_verifies_version(tmp_path):
    """load_partition raises on version skew unless ignored."""
    from llm_infer_sim.core.operator_db.stores.jsonl import JsonlOperatorStore

    part = tmp_path / "RTX_4090" / "vllm-9.9.9"
    part.mkdir(parents=True)
    (part / "gemm.jsonl").write_text("")  # empty partition is fine for the check

    store = JsonlOperatorStore()
    with pytest.raises(OperatorDBVersionMismatch):
        store.load_partition(tmp_path, hardware="RTX_4090", framework="vllm",
                             framework_version="9.9.9", runtime_version="0.19.0")

    counts = store.load_partition(
        tmp_path, hardware="RTX_4090", framework="vllm", framework_version="9.9.9",
        runtime_version="0.19.0", ignore_version_mismatch=True,
    )
    assert store.framework_version == "9.9.9"
    assert counts["gemm"] == 0
