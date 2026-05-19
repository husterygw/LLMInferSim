"""vLLM single-GPU distributed env setup.

vLLM `RowParallelLinear` 等 model_executor.layers 调用前需要 torch.distributed
+ vllm.distributed 初始化 (即使 world_size=1). 这里做最小初始化, 整个进程只做一次.

AIC 等价代码: aiconfigurator/collector/vllm/utils.py:setup_distributed
"""
from __future__ import annotations

import os

_INITIALIZED = False


def ensure_initialized(device: int) -> None:
    """初始化 torch + vllm distributed env (world_size=1). 幂等."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(device)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", str(device))
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    # vLLM-specific (parallel state)
    from vllm.distributed.parallel_state import (  # noqa: PLC0415
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    init_distributed_environment(
        world_size=1, rank=0,
        distributed_init_method="env://",
        local_rank=device,
    )
    ensure_model_parallel_initialized(1, 1)

    _INITIALIZED = True


def torch_dtype(dtype_str: str):
    """case.params['dtype'] str → torch.dtype."""
    import torch
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise NotImplementedError(f"unsupported dtype: {dtype_str}")
    return mapping[dtype_str]
