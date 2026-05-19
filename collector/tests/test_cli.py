"""cli.py — argparse + 子命令端到端."""
from __future__ import annotations

import sys
import types
from io import StringIO

import pytest

from collector import _bootstrap, cli
from collector.registry import REGISTRY
from collector.schemas import CollectorEntry, Framework, OpKind


@pytest.fixture(autouse=True)
def clean_registry():
    """每个 test 用干净 registry."""
    REGISTRY.clear()
    yield
    REGISTRY.clear()


# ---------------------------------------------------------------------------
# argparse build
# ---------------------------------------------------------------------------

def test_parser_help_runs():
    """--help 不挂."""
    parser = cli._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_parser_default_framework():
    parser = cli._build_parser()
    args = parser.parse_args([])
    assert args.frameworks == ["vllm"]


def test_parser_multiple_ops():
    parser = cli._build_parser()
    args = parser.parse_args(["--ops", "gemm", "moe"])
    assert args.ops == ["gemm", "moe"]


def test_parser_limit_int():
    parser = cli._build_parser()
    args = parser.parse_args(["--limit", "5"])
    assert args.limit == 5


# ---------------------------------------------------------------------------
# --list-ops
# ---------------------------------------------------------------------------

def test_list_ops_empty_registry(monkeypatch, capsys):
    """空 registry case — patch bootstrap 不 register."""
    monkeypatch.setattr(_bootstrap, "register_defaults", lambda: REGISTRY.clear())
    rc = cli.main(["--list-ops"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "registry 空" in out or "registry empty" in out.lower() or "暂未" in out


def test_list_ops_populated_default_bootstrap(capsys):
    """默认 bootstrap 注册了 GEMM + ATTENTION (vLLM)."""
    rc = cli.main(["--list-ops"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemm" in out
    assert "attention" in out


def test_list_ops_populated(capsys, monkeypatch):
    def fake_bootstrap():
        REGISTRY.clear()
        REGISTRY.register(CollectorEntry(
            op=OpKind.GEMM, framework=Framework.VLLM,
            get_cases_module="collector.cases.x:get_cases",
            run_case_module="collector.runners.vllm_gemm:run_case",
            output_file="gemm.jsonl",
        ))
    monkeypatch.setattr(_bootstrap, "register_defaults", fake_bootstrap)
    rc = cli.main(["--list-ops"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemm" in out
    assert "vllm" in out


# ---------------------------------------------------------------------------
# --show-env
# ---------------------------------------------------------------------------

def test_show_env_runs(capsys):
    rc = cli.main(["--show-env"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "python_version" in out
    assert "torch_version" in out
    assert "gpu_count" in out


# ---------------------------------------------------------------------------
# No entries matched
# ---------------------------------------------------------------------------

def test_no_entries_matched_returns_2(monkeypatch, capsys):
    """空 registry + 选 op → 返非 0 exit code + stderr 信息."""
    monkeypatch.setattr(_bootstrap, "register_defaults", lambda: REGISTRY.clear())
    rc = cli.main(["--frameworks", "vllm", "--ops", "gemm"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "No entries matched" in err


def test_unregistered_framework_returns_2(capsys):
    """选未注册 framework (sglang 当前没 entries) → 没匹配 → exit 2."""
    rc = cli.main(["--frameworks", "sglang", "--ops", "gemm"])
    assert rc == 2


# ---------------------------------------------------------------------------
# --dry-run 端到端 (用 mock entry + mock cases)
# ---------------------------------------------------------------------------

def test_dry_run_end_to_end(monkeypatch, capsys, tmp_path):
    """--dry-run + entry 注册 → scheduler.run_op 跑通, 不产文件."""
    from collector.schemas import Case
    from collector.paths import DataPaths

    # 1. 注入 fake cases module — params 不含 model (新设计), get_cases 签名带 profiles
    mod = types.ModuleType("test_cli_cases")
    mod.get_cases = lambda profiles: (   # type: ignore[attr-defined]
        [Case.make(OpKind.GEMM, {"i": i, "dtype": "bf16"}) for i in range(5)],
        {},
    )
    monkeypatch.setitem(sys.modules, "test_cli_cases", mod)

    # 2. fake bootstrap 注册一条
    def fake_bootstrap():
        REGISTRY.clear()
        REGISTRY.register(CollectorEntry(
            op=OpKind.GEMM, framework=Framework.VLLM,
            get_cases_module="test_cli_cases:get_cases",
            run_case_module="collector.runners.vllm_gemm:run_case",
            output_file="gemm.jsonl",
        ))
    monkeypatch.setattr(_bootstrap, "register_defaults", fake_bootstrap)

    # 3. 跑 dry-run
    rc = cli.main([
        "--frameworks", "vllm",
        "--ops", "gemm",
        "--out", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemm" in out
    assert "DRY RUN" in out
    # 没产真 jsonl
    assert not list(tmp_path.glob("**/gemm.jsonl"))


# ---------------------------------------------------------------------------
# 不带 dry-run 真跑 → NotImplementedError (走 _make_run_case_fn 占位)
# ---------------------------------------------------------------------------

def test_real_run_unresolvable_runner_raises(monkeypatch, capsys, tmp_path):
    """真跑模式: entry.run_case_module 指向不存在的模块 → ModuleNotFoundError loudly.

    Config 错应该 crash loudly (而不是 silently 把每 case 写 errors.jsonl),
    让用户立刻看到 registry 配错了.
    """
    from collector.schemas import Case

    mod = types.ModuleType("test_cli_real_cases")
    mod.get_cases = lambda profiles: ([Case.make(OpKind.GEMM, {"i": 0})], {})   # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "test_cli_real_cases", mod)

    def fake_bootstrap():
        REGISTRY.clear()
        REGISTRY.register(CollectorEntry(
            op=OpKind.GEMM, framework=Framework.VLLM,
            get_cases_module="test_cli_real_cases:get_cases",
            run_case_module="nonexistent.module:run_case",   # ← unresolvable
            output_file="gemm.jsonl",
        ))
    monkeypatch.setattr(_bootstrap, "register_defaults", fake_bootstrap)

    with pytest.raises(ModuleNotFoundError):
        cli.main([
            "--frameworks", "vllm",
            "--ops", "gemm",
            "--out", str(tmp_path),
        ])


# ---------------------------------------------------------------------------
# --shape-profiles
# ---------------------------------------------------------------------------

def test_shape_profiles_default_loads_all():
    """没传 --shape-profiles → load 全部已知 profile."""
    profiles = cli._resolve_profiles(None)
    names = [p.profile_name for p in profiles]
    assert "qwen3_4b" in names
    assert "qwen3_30b_a3b" in names


def test_shape_profiles_explicit_subset():
    profiles = cli._resolve_profiles(["qwen3_4b"])
    names = [p.profile_name for p in profiles]
    assert names == ["qwen3_4b"]


def test_shape_profiles_unknown_exits_2(capsys):
    """未知 profile → 退出 code 2 + stderr 提示."""
    rc = cli.main([
        "--shape-profiles", "totally_not_a_profile",
        "--ops", "gemm",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Unknown profile" in err or "No entries matched" in err
