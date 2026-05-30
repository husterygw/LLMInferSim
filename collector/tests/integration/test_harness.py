"""harness.py — timing 框架测试.

全部 CPU-only. 用 timer_fn inject 假 latency, 不依赖 GPU.
GPU 路径放 runners/test_*_smoke.py 跟实测一起跑.
"""
from __future__ import annotations

import math

import pytest

from collector.harness import (
    BenchConfig,
    BenchResult,
    _summarize_latencies,
    measure,
)


# ---------------------------------------------------------------------------
# _summarize_latencies (pure function)
# ---------------------------------------------------------------------------

class TestSummarizeLatencies:
    def test_single_value(self):
        s = _summarize_latencies([100.0])
        assert s["p10"] == 100.0
        assert s["p50"] == 100.0
        assert s["p90"] == 100.0

    def test_two_values_median_is_avg(self):
        s = _summarize_latencies([100.0, 200.0])
        # median = 150
        assert s["p50"] == 150.0

    def test_uniform_distribution(self):
        """1..100 等距,p50 应在中间附近,p10 / p90 在两端."""
        s = _summarize_latencies(list(range(1, 101)))
        assert 49 <= s["p50"] <= 51       # median ≈ 50
        assert s["p10"] < 20                # 10th percentile 在低端
        assert s["p90"] > 80                # 90th percentile 在高端

    def test_p10_le_p50_le_p90(self):
        """percentile 必须有序."""
        s = _summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
        assert s["p10"] <= s["p50"] <= s["p90"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _summarize_latencies([])

    def test_outlier_doesnt_dominate(self):
        """1 个 outlier 不应让 p50 跑偏 — median 抗 outlier."""
        s = _summarize_latencies([10.0] * 9 + [10000.0])
        assert s["p50"] == 10.0      # median = 10, outlier 在 p90 那边


# ---------------------------------------------------------------------------
# measure() with injected timer_fn (no GPU needed)
# ---------------------------------------------------------------------------

class TestMeasureInjectedTimer:
    def test_constant_latency(self):
        """timer 每次返 100µs, 期待 p10=p50=p90=100."""
        cfg = BenchConfig(n_warmups=2, n_iters=10, use_cuda_graph=False)
        timer = lambda fn: 100.0  # noqa: E731
        result = measure(lambda: None, cfg, timer_fn=timer)
        assert result.latency_us_p50 == 100.0
        assert result.latency_us_p10 == 100.0
        assert result.latency_us_p90 == 100.0
        assert not result.used_cuda_graph
        assert result.fallback_reason is None or "cuda_unavailable" in result.fallback_reason

    def test_warmup_runs_first(self):
        """warmup phase 实际跑 kernel n_warmups 次, 然后 measure n_iters 次."""
        call_counts = {"kernel": 0}

        def kernel():
            call_counts["kernel"] += 1

        cfg = BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=False)
        timer = lambda fn: (fn() or 50.0)  # noqa: E731 — 调一次 + 返 50

        measure(kernel, cfg, timer_fn=timer)
        # warmup 3 次 (eager path) + measure 10 次 (timer 内调一次 = 10 次)
        assert call_counts["kernel"] == 3 + 10

    def test_iters_count_reflected(self):
        cfg = BenchConfig(n_warmups=1, n_iters=5, use_cuda_graph=False)
        result = measure(lambda: None, cfg, timer_fn=lambda fn: 10.0)
        assert result.n_iters == 5
        assert result.n_warmups == 1

    def test_varying_latency_percentiles_correct(self):
        """timer 返一系列已知 µs, p50/p10/p90 跟手算一致."""
        latencies = iter([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cfg = BenchConfig(n_warmups=0, n_iters=10, use_cuda_graph=False)
        result = measure(lambda: None, cfg, timer_fn=lambda fn: float(next(latencies)))
        assert 50 <= result.latency_us_p50 <= 60   # median of 1..10 ≈ 55
        assert result.latency_us_p10 < 30
        assert result.latency_us_p90 > 70

    def test_cuda_graph_disabled_flag(self):
        cfg = BenchConfig(n_warmups=1, n_iters=3, use_cuda_graph=False)
        result = measure(lambda: None, cfg, timer_fn=lambda fn: 1.0)
        assert not result.used_cuda_graph
        assert result.fallback_reason is None     # 没尝试 graph, 不是 fallback

    def test_cuda_graph_fallback_records_reason(self, monkeypatch):
        """capture 失败时 fallback eager + 记 reason."""
        # mock _try_capture_cuda_graph 返失败
        from collector import harness
        monkeypatch.setattr(
            harness, "_try_capture_cuda_graph",
            lambda fn, n_warmups: (None, "capture_failed: mocked"),
        )
        cfg = BenchConfig(n_warmups=1, n_iters=3, use_cuda_graph=True,
                          allow_graph_fail=True)
        result = measure(lambda: None, cfg, timer_fn=lambda fn: 1.0)
        assert not result.used_cuda_graph
        assert result.fallback_reason == "capture_failed: mocked"

    def test_cuda_graph_disallowed_fail_raises(self, monkeypatch):
        """allow_graph_fail=False 时 capture 失败应该 raise."""
        from collector import harness
        monkeypatch.setattr(
            harness, "_try_capture_cuda_graph",
            lambda fn, n_warmups: (None, "capture_failed: mocked"),
        )
        cfg = BenchConfig(n_warmups=1, n_iters=3, use_cuda_graph=True,
                          allow_graph_fail=False)
        with pytest.raises(RuntimeError, match="cudagraph capture failed"):
            measure(lambda: None, cfg, timer_fn=lambda fn: 1.0)

    def test_cuda_graph_success_path(self, monkeypatch):
        """capture 成功时 used_cuda_graph=True + fallback_reason=None."""
        from collector import harness

        class FakeGraph:
            def __init__(self):
                self.replay_count = 0
            def replay(self):
                self.replay_count += 1

        fake_graph = FakeGraph()
        monkeypatch.setattr(
            harness, "_try_capture_cuda_graph",
            lambda fn, n_warmups: (fake_graph, None),
        )
        cfg = BenchConfig(n_warmups=2, n_iters=5, use_cuda_graph=True)
        result = measure(lambda: None, cfg, timer_fn=lambda fn: (fn(), 1.0)[1])
        assert result.used_cuda_graph
        assert result.fallback_reason is None
        # replay 5 次 (timer 调一次 → fake_graph.replay)
        assert fake_graph.replay_count == 5


# ---------------------------------------------------------------------------
# BenchResult dataclass
# ---------------------------------------------------------------------------

class TestBenchResult:
    def test_fields_present(self):
        r = BenchResult(
            latency_us_p50=10.0, latency_us_p10=9.0, latency_us_p90=11.0,
            used_cuda_graph=False, n_warmups=3, n_iters=10,
        )
        assert r.latency_us_p50 == 10.0
        assert r.fallback_reason is None

    def test_fallback_reason_optional(self):
        r = BenchResult(
            latency_us_p50=10.0, latency_us_p10=9.0, latency_us_p90=11.0,
            used_cuda_graph=False, n_warmups=3, n_iters=10,
            fallback_reason="capture_failed",
        )
        assert r.fallback_reason == "capture_failed"
