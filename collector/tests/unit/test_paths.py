"""paths.DataPaths 路径计算 + 目录创建."""
from __future__ import annotations

from pathlib import Path

from collector.paths import DataPaths
from collector.schemas import Framework, OpKind


def test_from_args_builds_correct_root(tmp_path):
    p = DataPaths.from_args(
        data_root=tmp_path,
        hardware="RTX_4090",
        framework=Framework.VLLM,
        framework_version="0.19.1",
    )
    assert p.base == tmp_path / "operator_db" / "RTX_4090" / "vllm-0.19.1"


def test_op_jsonl_path(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    assert p.op_jsonl(OpKind.GEMM).name == "gemm.jsonl"
    assert p.op_jsonl(OpKind.MOE).name == "moe.jsonl"


def test_errors_jsonl_path(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    err = p.errors_jsonl(OpKind.GEMM)
    assert err.parent.name == "errors"
    assert err.name == "gemm.jsonl"


def test_checkpoint_json_path(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    cp = p.checkpoint_json(OpKind.MOE)
    assert cp.parent.name == "checkpoints"
    assert cp.name == "moe.json"


def test_progress_and_manifest_paths(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    assert p.progress_jsonl.name == "progress.jsonl"
    assert p.manifest_yaml.name == "manifest.yaml"


def test_ensure_dirs_creates_all(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    assert not p.base.exists()
    p.ensure_dirs()
    assert p.base.exists()
    assert (p.base / "errors").exists()
    assert (p.base / "checkpoints").exists()


def test_ensure_dirs_idempotent(tmp_path):
    p = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    p.ensure_dirs()
    p.ensure_dirs()    # 再调一次不应 raise
    assert p.base.exists()


def test_different_frameworks_different_paths(tmp_path):
    p_vllm = DataPaths.from_args(tmp_path, "RTX_4090", Framework.VLLM, "0.19.1")
    p_sglang = DataPaths.from_args(tmp_path, "RTX_4090", Framework.SGLANG, "0.4.0")
    assert p_vllm.base != p_sglang.base
    assert "vllm-0.19.1" in str(p_vllm.base)
    assert "sglang-0.4.0" in str(p_sglang.base)
