"""Collective (NCCL) cases — AllReduce / AllToAll, profile-derived message size.

NCCL collective 性能 = f(op_subtype, num_gpus, message_size_bytes, dtype, topology).
不依赖模型, 但 message_size 取值由 profile 决定: 一般 = tokens × hidden × dtype_bytes.

跨 profile dedup 自然发生:
  - Qwen3-4B  hidden=2560: 128 tok × 2560 × 2 = 640KB
  - Qwen3-30B hidden=2048: 128 tok × 2048 × 2 = 512KB → 不同 case
  - 但同 hidden 模型给同 size, 自动 dedup

主键 (case_id 来源):
  op_subtype, num_gpus, message_size_bytes, dtype, topology_hint, in_context

注意:
  - 这些 case 必须走多 GPU + torchrun, 主 scheduler 不能直接跑.
  - CollectorEntry.multi_gpu=True 让主 scheduler skip, 由 distributed/run_collective.py
    通过 torchrun --nproc_per_node=N 启动跑.
  - in_context=True 的 case 后续需要定义 disruption pattern (router / fused_moe 等
    kernel 插在 AR 之间, 重现 §13.7 vLLM pynccl current_stream 行为). 第一版 only False.
"""
from __future__ import annotations

from collector.cases._dedup import merge_and_dedup
from collector.profiles import ProfileSpec
from collector.schemas import Case, OpKind


# ---------------------------------------------------------------------------
# Sweep defaults
# ---------------------------------------------------------------------------

# 跟 GEMM/MoE 保持 token 序列一致 (decode=1 → prefill=8192)
DEFAULT_NUM_TOKENS: list[int] = [1, 4, 16, 32, 128, 512, 2048, 4096, 8192]

# NCCL 并行度. AllReduce 跟 AllToAll 都按这个扫.
DEFAULT_NUM_GPUS: list[int] = [2, 4, 8]

# RTX 4090 dual NUMA: concentrated 同 root, balanced 跨 root
DEFAULT_TOPOLOGY_HINTS: list[str] = ["concentrated", "balanced"]

DEFAULT_DTYPES: list[str] = ["bf16"]
DEFAULT_EXECUTION_MODES: list[str] = ["eager", "cudagraph"]

# dtype → bytes per element
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "fp4": 0.5}


# ---------------------------------------------------------------------------
# Per profile
# ---------------------------------------------------------------------------

def get_cases_for_profile(
    profile: ProfileSpec,
    *,
    num_tokens_values: list[int] | None = None,
    num_gpus_values: list[int] | None = None,
    topology_hints: list[str] | None = None,
    dtypes: list[str] | None = None,
    execution_modes: list[str] | None = None,
    include_allreduce: bool = True,
    include_alltoall: bool | None = None,
) -> list[Case]:
    """从单 profile 派生 collective cases.

    AllReduce: TP attention/FFN output reduce 用. 所有 profile 都跑.
    AllToAll:  MoE EP dispatch/combine 用. 默认 only `profile.has_moe=True` 跑.
    """
    num_tokens_values = num_tokens_values or DEFAULT_NUM_TOKENS
    num_gpus_values = num_gpus_values or DEFAULT_NUM_GPUS
    topology_hints = topology_hints or DEFAULT_TOPOLOGY_HINTS
    dtypes = dtypes or DEFAULT_DTYPES
    execution_modes = execution_modes or DEFAULT_EXECUTION_MODES
    if include_alltoall is None:
        include_alltoall = profile.has_moe

    cases: list[Case] = []

    for mode in execution_modes:
        if include_allreduce:
            for num_gpus in num_gpus_values:
                for dtype in dtypes:
                    for tokens in num_tokens_values:
                        size = int(tokens * profile.hidden * _DTYPE_BYTES[dtype])
                        for topology in topology_hints:
                            cases.append(_collective_case(
                                op_subtype="allreduce",
                                num_gpus=num_gpus,
                                message_size_bytes=size,
                                dtype=dtype,
                                topology_hint=topology,
                                in_context=False,
                                execution_mode=mode,
                            ))

        if include_alltoall:
            for num_gpus in num_gpus_values:
                if num_gpus > profile.moe_num_experts and profile.has_moe:
                    continue
                for dtype in dtypes:
                    for tokens in num_tokens_values:
                        size = int(tokens * profile.hidden * _DTYPE_BYTES[dtype])
                        for topology in topology_hints:
                            cases.append(_collective_case(
                                op_subtype="alltoall",
                                num_gpus=num_gpus,
                                message_size_bytes=size,
                                dtype=dtype,
                                topology_hint=topology,
                                in_context=False,
                                execution_mode=mode,
                            ))

    return cases


# ---------------------------------------------------------------------------
# Multi-profile dedup
# ---------------------------------------------------------------------------

def get_cases(
    profiles: list[ProfileSpec],
    **opts,
) -> tuple[list[Case], dict[str, list[str]]]:
    """跨 profile dedup. 同 message_size 不同 profile 自然 dedup."""
    per_profile = [
        (p.profile_name, get_cases_for_profile(p, **opts))
        for p in profiles
    ]
    return merge_and_dedup(per_profile)


# ---------------------------------------------------------------------------
# Case constructor
# ---------------------------------------------------------------------------

def _collective_case(
    *,
    op_subtype: str,
    num_gpus: int,
    message_size_bytes: int,
    dtype: str,
    topology_hint: str,
    in_context: bool,
    execution_mode: str,
) -> Case:
    """构造单个 collective Case. multi_gpu=True 让主 scheduler skip."""
    return Case.make(
        OpKind.COLLECTIVE,
        params={
            "op_subtype": op_subtype,
            "num_gpus": num_gpus,
            "message_size_bytes": message_size_bytes,
            "dtype": dtype,
            "topology_hint": topology_hint,
            "in_context": in_context,
            "execution_mode": execution_mode,
        },
        multi_gpu=True,
        prefix=_collective_prefix(op_subtype, num_gpus, message_size_bytes,
                                   topology_hint, execution_mode),
    )


def _collective_prefix(op_subtype: str, num_gpus: int,
                       size_bytes: int, topology: str,
                       execution_mode: str) -> str:
    if size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes // (1024 * 1024)}M"
    elif size_bytes >= 1024:
        size_str = f"{size_bytes // 1024}K"
    else:
        size_str = f"{size_bytes}B"
    return f"{op_subtype}_n{num_gpus}_{size_str}_{topology}_{execution_mode}"
