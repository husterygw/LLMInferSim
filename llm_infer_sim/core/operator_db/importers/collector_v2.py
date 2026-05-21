"""collector RawRecord (JSONL row) → OperatorRecord.

输入 dict 来自 `collector.schemas.RawRecord.to_json_dict()` (即 JSONL 一行).
按 op_kind dispatch 到对应 canonicalizer 生成 signature.
Stage 3 主要支持 GEMM; attention/moe/collective 也接, 但实测使用阶段 4/5/6.
"""
from __future__ import annotations

from typing import Any

from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operator_schema.attention import (
    attention_case_params_to_signature,
)
from llm_infer_sim.core.operator_schema.collective import (
    collective_case_params_to_signature,
)
from llm_infer_sim.core.operator_schema.gemm import gemm_case_params_to_signature
from llm_infer_sim.core.operator_schema.moe import moe_case_params_to_signature
from llm_infer_sim.core.operator_schema.signature import OperatorSignature


def raw_record_to_signature(record: dict[str, Any]) -> OperatorSignature:
    """JSONL row -> OperatorSignature, 按 op_kind dispatch."""
    op_kind = record["op_kind"]
    params = record["params"]
    ctx = dict(
        framework=record["framework"],
        framework_version=record["framework_version"],
        kernel_source=record["kernel_source"],
    )
    if op_kind == "gemm":
        return gemm_case_params_to_signature(params, **ctx)
    if op_kind == "attention":
        # collector params 不带 attention_backend/kv_dtype/block_size,
        # 从 kernel_source 推断 backend; kv_dtype 默认跟 dtype; block_size 默认 16.
        attention_backend = _infer_attention_backend(record["kernel_source"])
        return attention_case_params_to_signature(
            params,
            **ctx,
            attention_backend=attention_backend,
            kv_dtype=params["dtype"],
            block_size=16,
        )
    if op_kind == "moe":
        return moe_case_params_to_signature(params, **ctx)
    if op_kind == "collective":
        return collective_case_params_to_signature(params, **ctx)
    raise ValueError(f"unknown op_kind in RawRecord: {op_kind!r}")


def _infer_attention_backend(kernel_source: str) -> str | None:
    """从 collector kernel_source 推 attention_backend (flash_attn / flashinfer / ...)."""
    ks = kernel_source.lower()
    if "flash_attn" in ks or "flashattn" in ks:
        return "flash_attn"
    if "flashinfer" in ks:
        return "flashinfer"
    if "triton" in ks:
        return "triton"
    return None


def import_record(record: dict[str, Any], *, hardware: str) -> OperatorRecord:
    """JSONL row → OperatorRecord. hardware 由 partition 路径提供, 不在 RawRecord 里."""
    signature = raw_record_to_signature(record)
    metrics = record["metrics"]
    metadata = record.get("metadata", {})
    return OperatorRecord(
        signature=signature,
        hardware=hardware,
        framework=record["framework"],
        framework_version=record["framework_version"],
        execution_mode=record["execution_mode"],
        kernel_source=record["kernel_source"],
        latency_us_p50=float(metrics["latency_us_p50"]),
        latency_us_p10=float(metrics["latency_us_p10"]),
        latency_us_p90=float(metrics["latency_us_p90"]),
        n_iters=int(metrics["n_iters"]),
        n_warmups=int(metrics["n_warmups"]),
        confidence=1.0,
        source={
            "case_id": record["case_id"],
            "source_profiles": metadata.get("source_profiles", []),
            "device": record.get("device", ""),
            "fallback_reason": metadata.get("fallback_reason"),
        },
    )
