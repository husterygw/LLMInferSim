"""version_resolver.py — 按 framework_version 选 runner module."""
from __future__ import annotations

import pytest

from collector.schemas import CollectorEntry, Framework, OpKind, VersionRoute
from collector.version_resolver import parse_version, resolve_runner


# ---------------------------------------------------------------------------
# parse_version
# ---------------------------------------------------------------------------

class TestParseVersion:
    @pytest.mark.parametrize("input_str,expected", [
        ("0.19.1", (0, 19, 1)),
        ("0.20.0", (0, 20, 0)),
        ("1.0", (1, 0)),
        ("1", (1,)),
        ("", ()),
        ("0.19.0rc1", (0, 19, 0)),       # rc strip
        ("0.20.0+abc", (0, 20, 0)),      # build meta strip
        ("0.21.0.dev123", (0, 21, 0)),   # dev part 解析到第三段就 break (因为 dev 不是 digits)
    ])
    def test_parses(self, input_str, expected):
        assert parse_version(input_str) == expected

    def test_compare_correctly(self):
        assert parse_version("0.19.0") < parse_version("0.19.1")
        assert parse_version("0.19.1") < parse_version("0.20.0")
        assert parse_version("0.19.0") == parse_version("0.19.0")
        assert parse_version("0.19.0") < parse_version("1.0.0")


# ---------------------------------------------------------------------------
# resolve_runner
# ---------------------------------------------------------------------------

def _entry_with_versions(*versions: VersionRoute) -> CollectorEntry:
    return CollectorEntry(
        op=OpKind.GEMM,
        framework=Framework.VLLM,
        get_cases_module="collector.cases.x:get_gemm_cases",
        run_case_module="collector.runners.vllm_gemm.DEFAULT",
        output_file="gemm.jsonl",
        versions=versions,
    )


class TestResolveRunner:
    def test_no_versions_uses_default(self):
        e = _entry_with_versions()
        assert resolve_runner(e, "0.19.1") == "collector.runners.vllm_gemm.DEFAULT"

    def test_single_version_matched(self):
        """actual >= min_version → 用对应 runner."""
        e = _entry_with_versions(
            VersionRoute("0.19.0", "collector.runners.vllm_gemm.v19"),
        )
        assert resolve_runner(e, "0.19.1") == "collector.runners.vllm_gemm.v19"

    def test_single_version_too_low_fallback_default(self):
        """actual < min_version → fallback default."""
        e = _entry_with_versions(
            VersionRoute("0.20.0", "collector.runners.vllm_gemm.v20"),
        )
        assert resolve_runner(e, "0.19.1") == "collector.runners.vllm_gemm.DEFAULT"

    def test_multiple_versions_picks_highest_matching(self):
        """多版本时取最大的 min_version <= actual 那条."""
        e = _entry_with_versions(
            VersionRoute("0.17.0", "collector.runners.vllm_gemm.v17"),
            VersionRoute("0.19.0", "collector.runners.vllm_gemm.v19"),
            VersionRoute("0.20.0", "collector.runners.vllm_gemm.v20"),
        )
        # actual=0.19.5 → 应取 v19 (0.20.0 > 0.19.5 排除)
        assert resolve_runner(e, "0.19.5") == "collector.runners.vllm_gemm.v19"

    def test_actual_above_all_picks_highest(self):
        e = _entry_with_versions(
            VersionRoute("0.17.0", "v17"),
            VersionRoute("0.19.0", "v19"),
        )
        # actual=0.21.0 → max(<=0.21) = 0.19 → v19
        assert resolve_runner(e, "0.21.0") == "v19"

    def test_exact_match(self):
        e = _entry_with_versions(
            VersionRoute("0.19.0", "v19"),
        )
        assert resolve_runner(e, "0.19.0") == "v19"

    def test_order_independent(self):
        """versions tuple 顺序不影响选取."""
        e_asc = _entry_with_versions(
            VersionRoute("0.17.0", "v17"),
            VersionRoute("0.19.0", "v19"),
            VersionRoute("0.20.0", "v20"),
        )
        e_desc = _entry_with_versions(
            VersionRoute("0.20.0", "v20"),
            VersionRoute("0.19.0", "v19"),
            VersionRoute("0.17.0", "v17"),
        )
        assert resolve_runner(e_asc, "0.19.5") == resolve_runner(e_desc, "0.19.5")

    def test_with_rc_version(self):
        """actual 是 rc 版本时 strip 后正确路由."""
        e = _entry_with_versions(
            VersionRoute("0.19.0", "v19"),
            VersionRoute("0.20.0", "v20"),
        )
        # 0.19.5rc1 → parse_version (0, 19, 5) → match v19 (0.19.0)
        assert resolve_runner(e, "0.19.5rc1") == "v19"
