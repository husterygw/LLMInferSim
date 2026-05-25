"""Report measured operator latency vs roofline lower bound.

Compares real collector JSONL rows with the roofline backend for every op kind
that can be reconstructed as a runtime operator.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.operator_db.importers.collector_v2 import import_record
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.operators import (
    AllGather,
    AllReduce,
    AllToAll,
    Attention,
    Collective,
    FusedMoE,
    GEMM,
    MoERoutingProfile,
    P2P,
    ReduceScatter,
)
from llm_infer_sim.core.operators.context import OperatorContext
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import HardwareConfig, get_hardware_profile
from llm_infer_sim.core.profiles.model_config import ModelConfig
from llm_infer_sim.core.operators.gemm import dtype_to_bytes


FIELDNAMES = [
    "status",
    "op_kind",
    "op_subtype",
    "execution_mode",
    "dtype",
    "m",
    "n",
    "k",
    "tp",
    "kernel_source",
    "case_id",
    "measured_us_p50",
    "measured_us_p10",
    "measured_us_p90",
    "roofline_us",
    "roofline_gap",
    "roofline_bottleneck",
    "t_compute_us",
    "t_memory_us",
    "arithmetic_intensity",
    "source_profiles",
]

SUPPORTED_ROOFLINE_OP_KINDS = {"gemm", "attention", "moe", "collective"}


def load_records(path: Path | str, hardware: str) -> list[OperatorRecord]:
    records: list[OperatorRecord] = []
    p = Path(path)
    with p.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                records.append(import_record(row, hardware=hardware))
            except Exception as exc:
                raise ValueError(f"failed to import {p}:{line_no}: {exc}") from exc
    return records


def _dtype_bytes(dtype: str) -> float:
    return dtype_to_bytes(dtype)


def _deploy_from_record(record: OperatorRecord) -> DeployConfig:
    parallel = dict(record.signature.parallel)
    runtime = dict(record.signature.runtime)
    return DeployConfig(
        tp_size=int(parallel.get("tp", 1) or 1),
        ep_size=int(parallel.get("ep", 1) or 1),
        execution_mode=runtime.get("execution_mode", record.execution_mode),
        backend=runtime.get("framework", record.framework),
        backend_version=runtime.get("framework_version", record.framework_version),
        block_size=int(runtime.get("block_size", 16) or 16),
    )


def _ctx_from_record(
    record: OperatorRecord,
    *,
    hw: HardwareConfig,
    model: ModelConfig | None = None,
) -> OperatorContext:
    elem_bytes = _dtype_bytes(record.signature.dtype)
    return OperatorContext(
        model=model or ModelConfig(),
        deploy=_deploy_from_record(record),
        hw=hw,
        w_byte=elem_bytes,
        a_byte=elem_bytes,
        kv_byte=elem_bytes,
        dtype=record.signature.dtype,
    )


def _attention_from_record(record: OperatorRecord, *, hw: HardwareConfig) -> Attention:
    shape = dict(record.signature.shape)
    runtime = dict(record.signature.runtime)
    ctx = _ctx_from_record(record, hw=hw)
    common = dict(
        layer_idx=0,
        bs=int(shape["num_seqs"]),
        n_q=int(shape["num_q_heads"]),
        n_kv=int(shape["num_kv_heads"]),
        head_dim=int(shape["head_dim"]),
        ctx=ctx,
        name=f"attention_{record.signature.op_subtype}_{record.source.get('case_id')}",
        op_subtype=record.signature.op_subtype,
        kernel_source=runtime.get("kernel_source", record.kernel_source),
    )
    if record.signature.op_subtype == "prefill":
        return Attention.flash_prefill(seqlen=int(shape["q_len"]), **common)
    if record.signature.op_subtype == "decode":
        return Attention.flash_decode(ctx_len=int(shape["kv_len"]), **common)
    raise ValueError(
        f"unsupported attention subtype: {record.signature.op_subtype!r}"
    )


def _moe_from_record(record: OperatorRecord, *, hw: HardwareConfig) -> FusedMoE:
    shape = dict(record.signature.shape)
    runtime = dict(record.signature.runtime)
    model = ModelConfig(
        name="collector_moe",
        hidden_dim=int(shape["hidden"]),
        is_moe=True,
        num_experts=int(shape["num_experts"]),
        num_activated_experts=int(shape["topk"]),
        expert_dim=int(shape["moe_intermediate"]),
    )
    ctx = _ctx_from_record(record, hw=hw, model=model)
    distribution = str(shape["routing_distribution"])
    alpha = float(shape["power_law_alpha"])
    routing = (
        MoERoutingProfile.power_law(alpha)
        if distribution == "power_law"
        else MoERoutingProfile.balanced()
    )
    return FusedMoE.routed_experts(
        layer_idx=0,
        tokens=int(shape["num_tokens"]),
        ctx=ctx,
        routing=routing,
        name=f"moe_{record.source.get('case_id')}",
        op_subtype=record.signature.op_subtype,
        kernel_source=runtime.get("kernel_source", record.kernel_source),
    )


def _collective_from_record(
    record: OperatorRecord,
    *,
    hw: HardwareConfig,
) -> Collective:
    shape = dict(record.signature.shape)
    parallel = dict(record.signature.parallel)
    runtime = dict(record.signature.runtime)
    cls_by_subtype: dict[str, type[Collective]] = {
        "allreduce": AllReduce,
        "allgather": AllGather,
        "reducescatter": ReduceScatter,
        "reduce_scatter": ReduceScatter,
        "alltoall": AllToAll,
        "p2p": P2P,
    }
    cls = cls_by_subtype.get(record.signature.op_subtype)
    if cls is None:
        raise ValueError(
            f"unsupported collective subtype: {record.signature.op_subtype!r}"
        )
    return cls(
        name=f"collective_{record.source.get('case_id')}",
        phase="operator_report",
        layer_idx=None,
        message_bytes=int(shape["message_bytes"]),
        world_size=int(parallel.get("world_size", 1) or 1),
        ctx=_ctx_from_record(record, hw=hw),
        dtype_override=record.signature.dtype,
        kernel_source=runtime.get("kernel_source", record.kernel_source),
        comm_backend=runtime.get("backend", "nccl") or "nccl",
        algo=runtime.get("algo") or "",
        protocol=runtime.get("protocol") or "",
        topology=runtime.get("topology") or "",
    )


def record_to_operator(record: OperatorRecord, *, hw: HardwareConfig) -> Any:
    op_kind = record.signature.op_kind
    if op_kind == "gemm":
        return GEMM.from_record(record, hw=hw)
    if op_kind == "attention":
        return _attention_from_record(record, hw=hw)
    if op_kind == "moe":
        return _moe_from_record(record, hw=hw)
    if op_kind == "collective":
        return _collective_from_record(record, hw=hw)
    raise ValueError(f"unsupported roofline op_kind: {op_kind!r}")


def estimate_gap(
    record: OperatorRecord,
    roofline_backend: RooflineBackend | None,
) -> dict[str, Any]:
    shape = dict(record.signature.shape)
    parallel = dict(record.signature.parallel)
    row = {
        "status": "unsupported_roofline",
        "op_kind": record.signature.op_kind,
        "op_subtype": record.signature.op_subtype,
        "execution_mode": record.execution_mode,
        "dtype": record.signature.dtype,
        "m": shape.get("m"),
        "n": shape.get("n"),
        "k": shape.get("k"),
        "tp": parallel.get("tp"),
        "kernel_source": record.kernel_source,
        "case_id": record.source.get("case_id"),
        "measured_us_p50": record.latency_us_p50,
        "measured_us_p10": record.latency_us_p10,
        "measured_us_p90": record.latency_us_p90,
        "roofline_us": None,
        "roofline_gap": None,
        "roofline_bottleneck": None,
        "t_compute_us": None,
        "t_memory_us": None,
        "arithmetic_intensity": None,
        "source_profiles": ",".join(record.source.get("source_profiles") or []),
    }

    if record.signature.op_kind not in SUPPORTED_ROOFLINE_OP_KINDS:
        return row
    if roofline_backend is None:
        raise ValueError(f"{record.signature.op_kind} record requires a RooflineBackend")

    op = record_to_operator(record, hw=roofline_backend.hw)
    entry = roofline_backend.estimate(op)
    roofline_us = entry.latency_s * 1e6
    row.update({
        "status": "ok",
        "roofline_us": roofline_us,
        "roofline_gap": (
            record.latency_us_p50 / roofline_us if roofline_us > 0 else None
        ),
        "roofline_bottleneck": entry.metadata.get("bottleneck"),
        "t_compute_us": float(entry.metadata.get("t_compute", 0.0)) * 1e6,
        "t_memory_us": float(entry.metadata.get("t_memory", 0.0)) * 1e6,
        "arithmetic_intensity": entry.metadata.get("arithmetic_intensity"),
    })
    return row


def write_csv(rows: list[dict[str, Any]], path: Path | str) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return float(ordered[idx])


def print_summary(rows: list[dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    unsupported = [r for r in rows if r["status"] != "ok"]

    print(f"rows={len(rows)} ok={len(ok_rows)} unsupported={len(unsupported)}")
    if unsupported:
        by_kind = defaultdict(int)
        for row in unsupported:
            by_kind[row["op_kind"]] += 1
        print("unsupported_roofline:", ", ".join(
            f"{kind}={count}" for kind, count in sorted(by_kind.items())
        ))
    if not ok_rows:
        return

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        groups[(row["op_kind"], row["op_subtype"], row["execution_mode"])].append(row)

    print()
    print("=== Roofline gap by (op_kind, op_subtype, execution_mode) ===")
    print(
        f"{'op_kind':<12} {'op_subtype':<16} {'mode':<10} {'N':>4} "
        f"{'meas_p50_us':>12} {'roof_us':>12} "
        f"{'gap_p50':>10} {'gap_p90':>10} {'gap_max':>10} "
        f"{'compute':>8} {'memory':>8}"
    )
    print("-" * 125)

    for (op_kind, subtype, mode), recs in sorted(groups.items()):
        measured = [float(r["measured_us_p50"]) for r in recs]
        roofline = [float(r["roofline_us"]) for r in recs]
        gaps = [float(r["roofline_gap"]) for r in recs]
        compute_count = sum(1 for r in recs if r["roofline_bottleneck"] == "compute")
        memory_count = sum(1 for r in recs if r["roofline_bottleneck"] == "memory")
        print(
            f"{op_kind:<12} {subtype:<16} {mode:<10} {len(recs):>4} "
            f"{_median(measured):>12.3f} {_median(roofline):>12.3f} "
            f"{_median(gaps):>10.2f} {_quantile(gaps, 0.9):>10.2f} "
            f"{max(gaps):>10.2f} {compute_count:>8} {memory_count:>8}"
        )


def print_shape_analysis(rows: list[dict[str, Any]], *, top_n: int = 12) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok" and r["op_kind"] == "gemm"]
    if not ok_rows:
        return

    print()
    print(f"=== Top {top_n} GEMM roofline gap cases ===")
    print(
        f"{'op_subtype':<16} {'mode':<10} {'m':>6} {'n':>7} {'k':>7} "
        f"{'measured_us':>12} {'roof_us':>10} {'gap':>8} {'bottleneck':>10}"
    )
    print("-" * 98)
    for row in sorted(ok_rows, key=lambda r: float(r["roofline_gap"]), reverse=True)[:top_n]:
        print(
            f"{row['op_subtype']:<16} {row['execution_mode']:<10} "
            f"{int(row['m']):>6} {int(row['n']):>7} {int(row['k']):>7} "
            f"{float(row['measured_us_p50']):>12.3f} "
            f"{float(row['roofline_us']):>10.3f} "
            f"{float(row['roofline_gap']):>8.2f} "
            f"{row['roofline_bottleneck']:>10}"
        )

    print()
    print("=== GEMM roofline gap by (op_subtype, m, execution_mode) ===")
    print(
        f"{'op_subtype':<16} {'m':>6} {'mode':<10} {'N':>4} "
        f"{'gap_p50':>10} {'gap_p90':>10} {'gap_max':>10} "
        f"{'meas_p50_us':>12} {'roof_us':>10}"
    )
    print("-" * 98)
    by_m: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_m[(row["op_subtype"], int(row["m"]), row["execution_mode"])].append(row)
    for (subtype, m, mode), recs in sorted(by_m.items()):
        gaps = [float(r["roofline_gap"]) for r in recs]
        measured = [float(r["measured_us_p50"]) for r in recs]
        roofline = [float(r["roofline_us"]) for r in recs]
        print(
            f"{subtype:<16} {m:>6} {mode:<10} {len(recs):>4} "
            f"{_median(gaps):>10.2f} {_quantile(gaps, 0.9):>10.2f} "
            f"{max(gaps):>10.2f} {_median(measured):>12.3f} "
            f"{_median(roofline):>10.3f}"
        )

    paired: dict[tuple[str, str, int, int, int, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in ok_rows:
        key = (
            row["op_subtype"],
            row["dtype"],
            int(row["m"]),
            int(row["n"]),
            int(row["k"]),
            row["tp"],
        )
        paired[key][row["execution_mode"]] = row
    pairs = []
    for key, mode_rows in paired.items():
        eager = mode_rows.get("eager")
        graph = mode_rows.get("cudagraph")
        if eager is None or graph is None:
            continue
        eager_us = float(eager["measured_us_p50"])
        graph_us = float(graph["measured_us_p50"])
        pairs.append({
            "key": key,
            "eager_us": eager_us,
            "graph_us": graph_us,
            "delta_us": eager_us - graph_us,
            "ratio": eager_us / graph_us if graph_us > 0 else None,
            "eager_gap": float(eager["roofline_gap"]),
            "graph_gap": float(graph["roofline_gap"]),
        })

    if pairs:
        print()
        print(f"=== Top {top_n} eager vs cudagraph paired deltas ===")
        print(
            f"{'op_subtype':<16} {'m':>6} {'n':>7} {'k':>7} "
            f"{'eager_us':>10} {'graph_us':>10} {'delta_us':>10} "
            f"{'ratio':>8} {'gap_e':>8} {'gap_g':>8}"
        )
        print("-" * 102)
        for item in sorted(pairs, key=lambda r: r["delta_us"], reverse=True)[:top_n]:
            subtype, _dtype, m, n, k, _tp = item["key"]
            print(
                f"{subtype:<16} {m:>6} {n:>7} {k:>7} "
                f"{item['eager_us']:>10.3f} {item['graph_us']:>10.3f} "
                f"{item['delta_us']:>10.3f} {item['ratio']:>8.2f} "
                f"{item['eager_gap']:>8.2f} {item['graph_gap']:>8.2f}"
            )

    below_one = [r for r in ok_rows if float(r["roofline_gap"]) < 1.0]
    if below_one:
        print()
        print(f"=== Top {min(top_n, len(below_one))} gap < 1 cases (check formula/hardware params) ===")
        print(
            f"{'op_subtype':<16} {'mode':<10} {'m':>6} {'n':>7} {'k':>7} "
            f"{'measured_us':>12} {'roof_us':>10} {'gap':>8}"
        )
        print("-" * 88)
        for row in sorted(below_one, key=lambda r: float(r["roofline_gap"]))[:top_n]:
            print(
                f"{row['op_subtype']:<16} {row['execution_mode']:<10} "
                f"{int(row['m']):>6} {int(row['n']):>7} {int(row['k']):>7} "
                f"{float(row['measured_us_p50']):>12.3f} "
                f"{float(row['roofline_us']):>10.3f} "
                f"{float(row['roofline_gap']):>8.2f}"
            )


def _resolve_input_paths(
    db_root: Path,
    hardware: str,
    framework: str,
    framework_version: str,
    op_kind: str,
) -> list[Path]:
    partition = db_root / hardware / f"{framework}-{framework_version}"
    if op_kind == "all":
        return [partition / f"{kind}.jsonl" for kind in (
            "gemm", "attention", "moe", "collective"
        )]
    return [partition / f"{op_kind}.jsonl"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare collector operator latency with roofline lower bound.",
    )
    parser.add_argument("--db-root", default="collector/data/operator_db")
    parser.add_argument("--hardware", default="RTX_4090")
    parser.add_argument("--framework", default="vllm")
    parser.add_argument("--framework-version", default="0.19.1")
    parser.add_argument(
        "--op-kind",
        default="gemm",
        choices=("gemm", "attention", "moe", "collective", "all"),
    )
    parser.add_argument("--csv", help="write per-record CSV")
    parser.add_argument("--jsonl", help="write per-record JSONL")
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="number of shape-level rows to print in top-case sections",
    )
    args = parser.parse_args()

    paths = _resolve_input_paths(
        Path(args.db_root),
        args.hardware,
        args.framework,
        args.framework_version,
        args.op_kind,
    )

    records: list[OperatorRecord] = []
    for path in paths:
        if not path.exists():
            if args.op_kind == "all":
                continue
            print(f"missing input JSONL: {path}", file=sys.stderr)
            return 1
        records.extend(load_records(path, args.hardware))

    if not records:
        print("no records loaded", file=sys.stderr)
        return 1

    hw = get_hardware_profile(args.hardware)
    backends: dict[str, RooflineBackend] = {}

    rows: list[dict[str, Any]] = []
    for record in records:
        mode = record.execution_mode
        if mode not in backends:
            backends[mode] = RooflineBackend(
                hw, DeployConfig(execution_mode=mode),
            )
        backend = backends[mode]
        rows.append(estimate_gap(record, backend))

    print_summary(rows)
    print_shape_analysis(rows, top_n=args.top_n)

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote CSV: {args.csv}")
    if args.jsonl:
        write_jsonl(rows, args.jsonl)
        print(f"wrote JSONL: {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
