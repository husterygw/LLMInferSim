"""vLLM GEMM runner — 用 vllm RowParallelLinear 测真实 kernel latency.

跟 AIC `collector/vllm/collect_gemm.py` 同思路: 用 vLLM 自己的 linear layer
保证测的就是 vLLM 实际跑的 kernel (包括 dispatch / dtype-specific code path).

第一版只 BF16 + tp=1, FP8 / TP>1 后续扩.

接口契约 (registry.run_case_module):
    run_case(case: Case, device: int) -> RawRecord
"""
from __future__ import annotations

from typing import Optional

from collector.harness import BenchConfig, BenchResult, measure
from collector.runners._vllm_dist import ensure_initialized, torch_dtype
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    RawRecord,
)


# ---------------------------------------------------------------------------
# Pure helper (no GPU needed) — testable
# ---------------------------------------------------------------------------

def build_record(
    case: Case,
    bench: BenchResult,
    *,
    framework_version: str,
    device_name: str,
    kernel_source: str = "vllm_row_parallel_linear",
) -> RawRecord:
    """从 bench result 组装 RawRecord. 纯函数 (无 GPU 依赖), 单元测可覆盖."""
    return RawRecord(
        case_id=case.case_id,
        op_kind=OpKind.GEMM,
        framework=Framework.VLLM,
        framework_version=framework_version,
        device=device_name,
        execution_mode=(
            ExecutionMode.CUDAGRAPH if bench.used_cuda_graph else ExecutionMode.EAGER
        ),
        kernel_source=kernel_source,
        params=dict(case.params),
        metrics=Metrics(
            latency_us_p50=bench.latency_us_p50,
            latency_us_p10=bench.latency_us_p10,
            latency_us_p90=bench.latency_us_p90,
            used_cuda_graph=bench.used_cuda_graph,
            n_warmups=bench.n_warmups,
            n_iters=bench.n_iters,
        ),
        metadata={
            "fallback_reason": bench.fallback_reason,
        },
    )


# ---------------------------------------------------------------------------
# Real GPU runner
# ---------------------------------------------------------------------------

def run_case(case: Case, device: int) -> RawRecord:
    """跑单 case GEMM 测量, 返 RawRecord.

    Steps:
      1. 初始化 vLLM distributed env (idempotent)
      2. 构造 vLLM `RowParallelLinear(disable_tp=True)` matching (n, k) 形状
      3. warmup + cuda graph capture + 中位数 timing
      4. 包成 RawRecord

    Args:
        case: must have params: {op_subtype, m, n, k, dtype, tp}
              tp=1 only (TP>1 走 distributed/, 不在这个 runner)
        device: cuda:N
    """
    params = case.params

    # TP>1: cases/gemm.py 已经把 n/k 切到 per-rank (qkv/gate_up 切 n, o/down 切 k),
    # 这里直接跑切分后的 shape (单 GPU 模拟 1 rank 的工作量), 不需要多 GPU runner.
    # 当前只 bf16, 后续按 dtype 加 FP8 path
    dtype_str = params["dtype"]
    if dtype_str != "bf16":
        raise NotImplementedError(
            f"vllm_gemm runner: dtype={dtype_str} not yet supported (BF16 only)"
        )

    import torch
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import RowParallelLinear

    dtype = torch_dtype(dtype_str)
    device_str = f"cuda:{device}"

    m = int(params["m"])
    n = int(params["n"])
    k = int(params["k"])

    # 整个 run_case 在 vllm config context 里:
    #   - vllm.distributed init 需要 current vllm config
    #   - RowParallelLinear.forward 调 CustomOp 也校验 current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        ensure_initialized(device)

        x = torch.randn((m, k), dtype=dtype, device=device_str)
        linear = RowParallelLinear(
            input_size=k,
            output_size=n,
            bias=False,
            skip_bias_add=True,
            params_dtype=dtype,
            quant_config=None,
            prefix="",
            return_bias=True,
            disable_tp=True,
        )
        linear.to(device_str)
        _ = linear.forward(x)   # dry run init

        def kernel_func() -> None:
            linear.forward(x)

        # case.params['execution_mode'] 决定模式;
        # 默认 cudagraph (向后兼容老 case schema 没这个字段).
        # eager case: 不 capture; cudagraph case: 必须 capture 成功, 失败 raise (写 errors)
        mode = params.get("execution_mode", "cudagraph")
        if mode == "eager":
            cfg = BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=False)
        elif mode == "cudagraph":
            cfg = BenchConfig(n_warmups=3, n_iters=10,
                              use_cuda_graph=True, allow_graph_fail=False)
        else:
            raise NotImplementedError(f"execution_mode {mode!r} not supported")
        bench = measure(kernel_func, cfg)

    device_name = torch.cuda.get_device_name(device)
    return build_record(
        case, bench,
        framework_version=str(vllm.__version__),
        device_name=device_name,
        kernel_source="vllm_row_parallel_linear",
    )
