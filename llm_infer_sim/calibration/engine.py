"""vLLM 引擎 lifecycle for calibration (详设 §9.4.2 B.2).

calibration **必须走真 CUDA 路径**: 显式 unset VLLM_VIRTUAL_BACKEND, 让 vLLM
platform 探测走到 nvidia CUDA backend, 加载真模型到 GPU。

入口:
    engine = spin_up(model="Qwen/Qwen3-4B", dtype="bf16", tp=1)
    # ... 跑 shot 们 ...
    spin_down(engine)

实现注意:
  - 强制 `enforce_eager=True` 防 torch.compile / cudagraph 引入额外开销, 干净校 op
  - `gpu_memory_utilization=0.7` 给点 buffer 防 OOM
  - `max_logprobs=0` 跟我们 VirtualPlatform feature gate 对齐 (其实 calibration 不用
    gate, 但显式设防 sampler 路径多走一份没用的 logprobs 计算)
  - `worker_extension_cls` 注入 LayerwiseProfileExtension
"""
from __future__ import annotations

import os
from typing import Any


# vLLM 0.20.1 LLM() 接受的 dtype 字符串
_VALID_DTYPES = ("auto", "float16", "bfloat16", "float32", "half", "bf16", "fp16")


def spin_up(
    model: str,
    dtype: str = "bfloat16",
    tp: int = 1,
    max_model_len: int = 20480,    # 容纳最大 attention shot (kv_decode=16384 + 1)
    max_num_seqs: int = 16,         # 跟 ATTENTION_SHOTS 最大 n_decode 对齐
    gpu_memory_utilization: float = 0.85,   # 留 KV 充足空间 (4090 24GB 紧)
    extra_engine_kwargs: dict[str, Any] | None = None,
) -> Any:
    """启动 vLLM LLM 引擎, 注入 LayerwiseProfileExtension.

    Args:
        model: HF id 或本地路径.
        dtype: bf16 / fp16 等.
        tp: tensor_parallel_size, 4090 single GPU 时 = 1.
        max_model_len: max sequence length (含 prompt + history + new).
        max_num_seqs: max concurrent sequences. 用足 attention shot 网格的 n_decode 上限.
        gpu_memory_utilization: 留 buffer 防 OOM.
        extra_engine_kwargs: 透传 LLM() 其他 kwargs (例 quantization 量化).

    Returns:
        vllm.LLM 实例.

    Raises:
        RuntimeError: VLLM_VIRTUAL_BACKEND=1 时 (calibration 必须真 GPU).
    """
    if os.environ.get("VLLM_VIRTUAL_BACKEND") == "1":
        raise RuntimeError(
            "calibration.engine.spin_up: VLLM_VIRTUAL_BACKEND=1 时不能跑 calibration. "
            "显式 `unset VLLM_VIRTUAL_BACKEND` 或在子进程 env 排除该变量."
        )
    if dtype not in _VALID_DTYPES:
        raise ValueError(f"dtype {dtype!r} 不在 {_VALID_DTYPES}")

    from vllm import LLM

    kwargs: dict[str, Any] = dict(
        model=model,
        tensor_parallel_size=tp,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,            # 关 torch.compile / cudagraph, 测纯 op
        max_logprobs=0,                # 跟 §7.5.2 feature gate 一致
        disable_log_stats=True,        # calibration 不需要 stat
        worker_extension_cls=(
            "llm_infer_sim.calibration.extension.LayerwiseProfileExtension"
        ),
    )
    if extra_engine_kwargs:
        kwargs.update(extra_engine_kwargs)
    return LLM(**kwargs)


def spin_down(engine: Any) -> None:
    """清理 vLLM 引擎. v0.20.1 没显式 close API, 删引用让 GC 回收即可。"""
    # 显式 deinit 路径 (如果有): vllm v0.20.1 LLM 没 .close(), workers 在
    # __del__ 收尾。这里 placeholder, 后续 vLLM 加 close 接口可对接。
    del engine


def fire_shot(
    engine: Any,
    shot_dict: dict[str, Any],
    catalog_slice: dict[str, dict[str, Any]],
    kind: str,
    iterations: int = 3,
) -> list[list[dict[str, Any]]]:
    """Host-side wrapper: 通过 collective_rpc 触发 LayerwiseProfileExtension.fire.

    Returns:
        list[list[TimingSample dict]] — 每 rank 一个 list.
        single-GPU (tp=1) 只有 1 个 rank, list 长度 1.
    """
    return engine.collective_rpc(
        "fire",
        args=(shot_dict, catalog_slice, kind, iterations),
    )
