"""Collector 数据 schema — 唯一 source of truth.

所有 read/write 经过这里. 跟 sim runtime 零依赖, 只用标准库.

JSONL 写出格式见 RawRecord.to_json_dict / ErrorRecord.to_json_dict.
Schema 演进通过 SCHEMA_VERSION 字段标记, importer 按版本路由.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "collector-v2"
# v2 (2026-05-19): RawRecord 移除 top-level `model` 字段.
#   profile_name 改走 metadata["source_profiles"] (provenance, 不是 DB 主键).
#   OperatorDB 查询键 = (hw, framework, version, mode, op_kind, dtype, shape, parallel, ...);
#   model 不是查询键, 同 shape 跨模型 dedup.


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OpKind(str, Enum):
    """采集的算子种类. 首版 4 类, 后续可扩."""
    GEMM = "gemm"
    ATTENTION = "attention"
    MOE = "moe"
    COLLECTIVE = "collective"


class ExecutionMode(str, Enum):
    """采集执行模式. cudagraph 失败时 fallback eager."""
    EAGER = "eager"
    CUDAGRAPH = "cudagraph"


class Framework(str, Enum):
    """首版只 vllm, 加 sglang/trtllm 时 enum 扩."""
    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM = "trtllm"


# ---------------------------------------------------------------------------
# Case (调度单元)
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """一个待跑的测量任务. case_id 必须稳定 (params hash), 作 resume key.

    `params` 是 framework-agnostic 的 shape 描述 (m/n/k/dtype/tp 等).
    `multi_gpu = True` 时主 scheduler 跳过, 交 distributed/ 走 torchrun.
    """
    case_id: str
    op_kind: OpKind
    params: dict[str, Any]
    multi_gpu: bool = False
    description: str = ""

    @classmethod
    def make(cls, op_kind: OpKind, params: dict[str, Any],
             prefix: str = "", multi_gpu: bool = False,
             description: str = "") -> "Case":
        """生成 case_id 稳定哈希. params 内 dict/list 顺序无关 (json sort_keys)."""
        canonical = json.dumps(params, sort_keys=True, default=str)
        h = hashlib.sha1(canonical.encode()).hexdigest()[:12]
        case_id = f"{op_kind.value}__{prefix}__{h}" if prefix else f"{op_kind.value}__{h}"
        return cls(
            case_id=case_id,
            op_kind=op_kind,
            params=params,
            multi_gpu=multi_gpu,
            description=description,
        )


# ---------------------------------------------------------------------------
# Metrics (单 case 测量结果)
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """单 case timing 统计. 时间单位统一 µs (微秒)."""
    latency_us_p50: float
    latency_us_p10: float
    latency_us_p90: float
    used_cuda_graph: bool
    n_warmups: int
    n_iters: int
    # 可选: power_w / power_limit_w 等, 默认 None
    power_w: float | None = None
    power_limit_w: float | None = None


# ---------------------------------------------------------------------------
# RawRecord (主输出, 1 case 1 行 JSONL)
# ---------------------------------------------------------------------------

@dataclass
class RawRecord:
    """一个 case 跑完后的 JSONL record.

    schema (collector-v2):
      schema_version, case_id, op_kind, framework, framework_version,
      device, execution_mode, kernel_source,
      params {op-specific shape fields — OperatorDB 主键的一部分},
      metrics {latency / used_cuda_graph / ...},
      metadata {timestamp / worker_id / git_sha / fallback_reason /
                source_profiles / 等 — provenance, 不是查询键}
    """
    case_id: str
    op_kind: OpKind
    framework: Framework
    framework_version: str
    device: str                          # e.g. "NVIDIA GeForce RTX 4090"
    execution_mode: ExecutionMode
    kernel_source: str                   # e.g. "vllm_row_parallel_linear", "vllm_fused_moe"
    params: dict[str, Any]
    metrics: Metrics
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        """转 JSON-serializable dict (供 writer.append). Enum → str."""
        d = asdict(self)
        d["op_kind"] = self.op_kind.value
        d["framework"] = self.framework.value
        d["execution_mode"] = self.execution_mode.value
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "RawRecord":
        """从 JSON dict 反序列化. 用于 importer / 测试 round-trip."""
        return cls(
            case_id=d["case_id"],
            op_kind=OpKind(d["op_kind"]),
            framework=Framework(d["framework"]),
            framework_version=d["framework_version"],
            device=d["device"],
            execution_mode=ExecutionMode(d["execution_mode"]),
            kernel_source=d["kernel_source"],
            params=d["params"],
            metrics=Metrics(**d["metrics"]),
            metadata=d.get("metadata", {}),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# ErrorRecord (失败 case 隔离)
# ---------------------------------------------------------------------------

@dataclass
class ErrorRecord:
    """失败 case 写 errors/<op>.jsonl. 不阻塞主流程."""
    case_id: str
    op_kind: OpKind
    framework: Framework
    error_type: str                      # e.g. "CudaOOM", "RuntimeError", "Timeout"
    error_message: str
    traceback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["op_kind"] = self.op_kind.value
        d["framework"] = self.framework.value
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "ErrorRecord":
        return cls(
            case_id=d["case_id"],
            op_kind=OpKind(d["op_kind"]),
            framework=Framework(d["framework"]),
            error_type=d["error_type"],
            error_message=d["error_message"],
            traceback=d.get("traceback", ""),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# CheckpointState (per-op resume state)
# ---------------------------------------------------------------------------

@dataclass
class CheckpointState:
    """per-(framework, op) checkpoint. 中断恢复用."""
    framework: Framework
    op_kind: OpKind
    done: set[str] = field(default_factory=set)        # case_id set
    failed: set[str] = field(default_factory=set)
    updated_at: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "op_kind": self.op_kind.value,
            "done": sorted(self.done),
            "failed": sorted(self.failed),
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "CheckpointState":
        return cls(
            framework=Framework(d["framework"]),
            op_kind=OpKind(d["op_kind"]),
            done=set(d.get("done", [])),
            failed=set(d.get("failed", [])),
            updated_at=d.get("updated_at", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# ProgressEntry (跨 op 总进度, 一行一 op)
# ---------------------------------------------------------------------------

@dataclass
class ProgressEntry:
    """progress.jsonl 一行. 给外部脚本 / CI 看总体状态."""
    framework: Framework
    op_kind: OpKind
    total: int
    done: int
    failed: int
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "op_kind": self.op_kind.value,
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# CollectorEntry (registry 注册项)
# ---------------------------------------------------------------------------

@dataclass
class VersionRoute:
    """按 framework_version 路由到不同 runner. AIC 风格."""
    min_version: str                     # e.g. "0.19.0" (semver, 含等)
    runner_module: str                   # e.g. "collector.runners.vllm_gemm"


@dataclass
class CollectorEntry:
    """registry 一条目. (op, framework) → runner + case 函数."""
    op: OpKind
    framework: Framework
    get_cases_module: str                # e.g. "collector.cases.qwen3_4b:get_gemm_cases"
    run_case_module: str                 # default runner if no version match
    output_file: str                     # e.g. "gemm.jsonl"
    versions: tuple[VersionRoute, ...] = ()
    multi_gpu: bool = False              # collective 类标 True, 走 distributed/
