"""单 case timing 框架.

设计:
  - measure() 高层 API: warmup → cudagraph capture (可选) → 多次 timing → p10/50/90
  - _summarize_latencies() 纯函数, 不依赖 torch, 单元测可直接覆盖
  - _time_with_cuda_events() / _capture_cuda_graph() 是 GPU-only 部分, 用 try/except
    保护; 测试时通过 inject timer_fn 绕过

时间单位: 全部 µs (微秒). float.

参考: AIC `benchmark_with_power` 关键行为, 但实现独立(零依赖 AIC).
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    """measure() 入参."""
    n_warmups: int = 3
    n_iters: int = 10
    use_cuda_graph: bool = True          # capture cudagraph if possible
    allow_graph_fail: bool = True        # cudagraph fail → fallback eager (不报错)


@dataclass
class BenchResult:
    """measure() 返回."""
    latency_us_p50: float
    latency_us_p10: float
    latency_us_p90: float
    used_cuda_graph: bool
    n_warmups: int
    n_iters: int
    fallback_reason: Optional[str] = None  # cudagraph 失败原因, 成功则 None


# ---------------------------------------------------------------------------
# Pure functions (CPU-only, 单元测可直接覆盖)
# ---------------------------------------------------------------------------

def _summarize_latencies(latencies_us: list[float]) -> dict[str, float]:
    """list of latency µs → p10/p50/p90 字典. 纯函数."""
    if not latencies_us:
        raise ValueError("empty latency list")
    if len(latencies_us) == 1:
        v = latencies_us[0]
        return {"p10": v, "p50": v, "p90": v}
    sorted_lat = sorted(latencies_us)
    # statistics.quantiles 用 method="inclusive" 跟 numpy 的 percentile linear 接近
    qs = statistics.quantiles(sorted_lat, n=10, method="inclusive")
    return {
        "p10": qs[0],       # 10th percentile
        "p50": statistics.median(sorted_lat),
        "p90": qs[8],       # 90th percentile
    }


# ---------------------------------------------------------------------------
# GPU timing primitives (用 torch.cuda.Event, 失败 fallback time.perf_counter)
# ---------------------------------------------------------------------------

def _time_one_iter_cuda(kernel_func: Callable[[], None]) -> float:
    """单次 kernel 执行 µs (CUDA Event). 失败 raise."""
    import torch  # lazy import
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    kernel_func()
    end.record()
    torch.cuda.synchronize()
    # elapsed_time returns ms
    return start.elapsed_time(end) * 1000.0  # ms → µs


def _time_one_iter_cpu_fallback(kernel_func: Callable[[], None]) -> float:
    """CPU fallback timer. 用于 CPU-only test 或 mock 路径."""
    t0 = time.perf_counter_ns()
    kernel_func()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1000.0   # ns → µs


def _try_capture_cuda_graph(kernel_func: Callable[[], None],
                            n_warmups: int) -> tuple[object | None, Optional[str]]:
    """尝试 capture CUDA graph. 失败返 (None, reason)."""
    try:
        import torch
    except ImportError:
        return None, "torch_unavailable"
    if not torch.cuda.is_available():
        return None, "cuda_unavailable"
    try:
        # warmup on side stream (AIC 风格)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                kernel_func()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            kernel_func()
        return g, None
    except Exception as e:  # noqa: BLE001
        return None, f"capture_failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Public measure() — high level entry
# ---------------------------------------------------------------------------

def measure(
    kernel_func: Callable[[], None],
    config: BenchConfig | None = None,
    *,
    timer_fn: Callable[[Callable[[], None]], float] | None = None,
) -> BenchResult:
    """跑 warmup + (可选 cudagraph) + n_iters 次 timing, 返 BenchResult.

    Args:
        kernel_func: 一次 kernel 调用 (无参 callable). 内部要保证 GPU 工作真发起.
        config: BenchConfig. None 用 default.
        timer_fn: 单 iter timer override (注入用, 单元测时绕开 CUDA).
                  默认 None → 用 _time_one_iter_cuda (要求 GPU).

    Returns:
        BenchResult.
    """
    cfg = config or BenchConfig()
    timer = timer_fn or _time_one_iter_cuda

    # -------- cudagraph 路径 --------
    used_graph = False
    fallback_reason: Optional[str] = None
    iter_target = kernel_func

    if cfg.use_cuda_graph:
        graph, reason = _try_capture_cuda_graph(kernel_func, cfg.n_warmups)
        if graph is not None:
            used_graph = True
            iter_target = graph.replay   # type: ignore[attr-defined]
        else:
            if not cfg.allow_graph_fail:
                raise RuntimeError(f"cudagraph capture failed: {reason}")
            fallback_reason = reason

    # -------- warmup (eager 路径, cudagraph 路径已 warmup 过) --------
    if not used_graph:
        for _ in range(cfg.n_warmups):
            iter_target()
        # 同步, 不让 warmup 残留 work 进 measurement
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    # -------- timing --------
    latencies_us: list[float] = []
    for _ in range(cfg.n_iters):
        latencies_us.append(timer(iter_target))

    stats = _summarize_latencies(latencies_us)
    return BenchResult(
        latency_us_p50=stats["p50"],
        latency_us_p10=stats["p10"],
        latency_us_p90=stats["p90"],
        used_cuda_graph=used_graph,
        n_warmups=cfg.n_warmups,
        n_iters=cfg.n_iters,
        fallback_reason=fallback_reason,
    )
