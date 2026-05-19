"""注册默认 CollectorEntry. 由 cli 在 startup 时显式调.

约定:
  - get_cases_module 函数签名: `get_cases(profiles: list[ProfileSpec]) -> (cases, sources)`
    scheduler.run_op 拿到 entry 后调用, 注入 profiles 列表 + 用 source 写 metadata.
  - run_case_module 函数签名: `run_case(case, device_id) -> RawRecord`
    实现在 runners/ 下 (commit #10+ 真接入 vLLM, 当前 placeholder).
"""
from __future__ import annotations

from collector.registry import REGISTRY
from collector.schemas import CollectorEntry, Framework, OpKind, VersionRoute


def register_defaults() -> None:
    """注册全部默认 entries. 幂等 (重复调先 clear)."""
    REGISTRY.clear()

    # ------------------------------------------------------------------
    # vLLM × {GEMM, Attention}
    # MoE / Collective 后续 commit 加 cases module 后再注册.
    # ------------------------------------------------------------------
    REGISTRY.register(CollectorEntry(
        op=OpKind.GEMM, framework=Framework.VLLM,
        get_cases_module="collector.cases.gemm:get_cases",
        run_case_module="collector.runners.vllm_gemm:run_case",   # TODO commit #10+
        output_file="gemm.jsonl",
    ))

    REGISTRY.register(CollectorEntry(
        op=OpKind.ATTENTION, framework=Framework.VLLM,
        get_cases_module="collector.cases.attention:get_cases",
        run_case_module="collector.runners.vllm_attention:run_case",  # TODO
        output_file="attention.jsonl",
    ))

    REGISTRY.register(CollectorEntry(
        op=OpKind.MOE, framework=Framework.VLLM,
        get_cases_module="collector.cases.moe:get_cases",
        run_case_module="collector.runners.vllm_moe:run_case",   # TODO
        output_file="moe.jsonl",
    ))

    REGISTRY.register(CollectorEntry(
        op=OpKind.COLLECTIVE, framework=Framework.VLLM,
        get_cases_module="collector.cases.collective:get_cases",
        run_case_module="collector.distributed.run_collective:run_case",   # TODO
        output_file="nccl.jsonl",
        multi_gpu=True,         # 主 scheduler skip, 由 distributed/ 走 torchrun
    ))

    # Collective: 走 distributed/, multi_gpu=True 让主 scheduler skip
    # REGISTRY.register(CollectorEntry(
    #     op=OpKind.COLLECTIVE, framework=Framework.VLLM,
    #     get_cases_module="collector.cases.collective:get_cases",
    #     run_case_module="collector.distributed.run_collective:run_case",
    #     output_file="nccl.jsonl",
    #     multi_gpu=True,
    # ))
