"""Report diagnostic internal breakdown for measured MoE collector cases.

This script keeps the main MoE latency model unchanged.  It reconstructs each
collector MoE record, reuses the existing roofline metadata, and adds a
logical breakdown that helps explain where measured fused_moe time exceeds the
modeled lower bound.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from llm_infer_sim.core.cost.backends.roofline import RooflineBackend
from llm_infer_sim.core.operator_db.schema import OperatorRecord
from llm_infer_sim.core.profiles.deploy import DeployConfig
from llm_infer_sim.core.profiles.hardware import get_hardware_profile

from report_operator_roofline_gap import (
    _resolve_input_paths,
    load_records,
    record_to_operator,
)


FIELDNAMES = [
    "case_id",
    "execution_mode",
    "dtype",
    "kernel_source",
    "num_tokens",
    "hidden",
    "moe_intermediate",
    "topk",
    "num_experts",
    "tp",
    "ep",
    "routing_distribution",
    "power_law_alpha",
    "measured_us_p50",
    "roofline_us",
    "residual_us",
    "residual_fraction",
    "roofline_gap",
    "roofline_bottleneck",
    "t_compute_us",
    "t_memory_us",
    "gate_compute_us",
    "up_compute_us",
    "down_compute_us",
    "gate_weight_mb",
    "up_weight_mb",
    "down_weight_mb",
    "modeled_weight_mb",
    "modeled_act_io_mb",
    "modeled_total_mem_mb",
    "extra_mem_equiv_mb",
    "materialized_intermediate_extra_mb",
    "materialized_intermediate_extra_us",
    "tokens_per_device",
    "distinct_experts_global",
    "distinct_experts_per_rank",
    "avg_rows_per_expert_rank",
    "effective_compute_peak_ratio",
    "effective_mem_bw_ratio",
    "source_profiles",
]


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _mb(value: float) -> float:
    return value / 1_000_000.0


def _case_distribution(shape: dict[str, Any]) -> str:
    dist = str(shape.get("routing_distribution", ""))
    alpha = _float(shape.get("power_law_alpha"), 0.0)
    if dist == "power_law":
        return f"power_law_{alpha:g}"
    return dist


def estimate_moe_breakdown(
    record: OperatorRecord,
    backend: RooflineBackend,
) -> dict[str, Any]:
    if record.signature.op_kind != "moe":
        raise ValueError(f"expected op_kind=moe, got {record.signature.op_kind!r}")

    shape = dict(record.signature.shape)
    parallel = dict(record.signature.parallel)
    op = record_to_operator(record, hw=backend.hw)
    entry = backend.estimate(op)
    spec = op.roofline_spec()
    md = entry.metadata

    measured_us = record.latency_us_p50
    roofline_us = entry.latency_s * 1e6
    residual_us = measured_us - roofline_us
    residual_fraction = residual_us / measured_us if measured_us > 0 else 0.0
    t_compute_us = _float(md.get("t_compute")) * 1e6
    t_memory_us = _float(md.get("t_memory")) * 1e6

    ep = int(parallel.get("ep", 1) or 1)
    a_byte = getattr(op.ctx, "a_byte", 2.0)
    tokens_per_device = _float(md.get("tokens_per_device"))
    expert_dim = int(shape["moe_intermediate"])
    distinct_global = _float(md.get("distinct_experts"))
    distinct_per_rank = distinct_global / ep if ep > 0 else distinct_global
    avg_rows = (
        tokens_per_device / distinct_per_rank
        if distinct_per_rank > 0 else 0.0
    )

    # Logical gate/up/down shares.  vLLM may fuse gate+up, but this split is
    # useful for attributing the lower-bound terms without changing latency.
    gate_compute_us = t_compute_us / 3.0
    up_compute_us = t_compute_us / 3.0
    down_compute_us = t_compute_us / 3.0
    gate_weight = spec.load_weight / 3.0
    up_weight = spec.load_weight / 3.0
    down_weight = spec.load_weight / 3.0

    bandwidth = backend.hw.effective_mem_bandwidth
    extra_mem_equiv = max(residual_us, 0.0) * 1e-6 * bandwidth

    # A diagnostic upper bound for the bytes added if gate/up intermediates and
    # the activation output were materialized in HBM instead of staying fused.
    materialized_extra = (
        tokens_per_device * expert_dim * a_byte * 2.0  # gate/up output write
        + tokens_per_device * expert_dim * a_byte * 2.0  # silu_mul read
        + tokens_per_device * expert_dim * a_byte       # activation write
        + tokens_per_device * expert_dim * a_byte       # down input read
    )
    materialized_extra_us = (
        materialized_extra / bandwidth * 1e6 if bandwidth > 0 else 0.0
    )

    source_profiles = ",".join(record.source.get("source_profiles") or [])
    return {
        "case_id": record.source.get("case_id"),
        "execution_mode": record.execution_mode,
        "dtype": record.signature.dtype,
        "kernel_source": record.kernel_source,
        "num_tokens": int(shape["num_tokens"]),
        "hidden": int(shape["hidden"]),
        "moe_intermediate": expert_dim,
        "topk": int(shape["topk"]),
        "num_experts": int(shape["num_experts"]),
        "tp": int(parallel.get("tp", 1) or 1),
        "ep": ep,
        "routing_distribution": _case_distribution(shape),
        "power_law_alpha": _float(shape.get("power_law_alpha")),
        "measured_us_p50": measured_us,
        "roofline_us": roofline_us,
        "residual_us": residual_us,
        "residual_fraction": residual_fraction,
        "roofline_gap": measured_us / roofline_us if roofline_us > 0 else None,
        "roofline_bottleneck": md.get("bottleneck"),
        "t_compute_us": t_compute_us,
        "t_memory_us": t_memory_us,
        "gate_compute_us": gate_compute_us,
        "up_compute_us": up_compute_us,
        "down_compute_us": down_compute_us,
        "gate_weight_mb": _mb(gate_weight),
        "up_weight_mb": _mb(up_weight),
        "down_weight_mb": _mb(down_weight),
        "modeled_weight_mb": _mb(spec.load_weight),
        "modeled_act_io_mb": _mb(spec.load_act + spec.store_act),
        "modeled_total_mem_mb": _mb(spec.mem_bytes),
        "extra_mem_equiv_mb": _mb(extra_mem_equiv),
        "materialized_intermediate_extra_mb": _mb(materialized_extra),
        "materialized_intermediate_extra_us": materialized_extra_us,
        "tokens_per_device": tokens_per_device,
        "distinct_experts_global": distinct_global,
        "distinct_experts_per_rank": distinct_per_rank,
        "avg_rows_per_expert_rank": avg_rows,
        "effective_compute_peak_ratio": (
            t_compute_us / measured_us if measured_us > 0 else None
        ),
        "effective_mem_bw_ratio": (
            t_memory_us / measured_us if measured_us > 0 else None
        ),
        "source_profiles": source_profiles,
    }


def write_csv(rows: list[dict[str, Any]], path: Path | str) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(rows: list[dict[str, Any]], *, top_n: int) -> None:
    print(f"rows={len(rows)}")
    if not rows:
        return

    print()
    print(f"=== Top {top_n} MoE internal residual cases ===")
    print(
        f"{'case_id':<54} {'meas_us':>10} {'roof_us':>10} {'resid_us':>10} "
        f"{'gap':>6} {'rows/expert':>12} {'extra_mem_mb':>13}"
    )
    print("-" * 122)
    for row in sorted(rows, key=lambda r: float(r["residual_us"]), reverse=True)[:top_n]:
        print(
            f"{str(row['case_id'])[:54]:<54} "
            f"{float(row['measured_us_p50']):>10.3f} "
            f"{float(row['roofline_us']):>10.3f} "
            f"{float(row['residual_us']):>10.3f} "
            f"{float(row['roofline_gap']):>6.2f} "
            f"{float(row['avg_rows_per_expert_rank']):>12.1f} "
            f"{float(row['extra_mem_equiv_mb']):>13.1f}"
        )

    print()
    print("=== Interpretation columns ===")
    print("residual_us = measured_us_p50 - roofline_us")
    print("extra_mem_equiv_mb = residual converted to bytes at hardware bandwidth")
    print("materialized_intermediate_extra_* estimates cost if fused intermediates hit HBM")


def _resolve_records(args: argparse.Namespace) -> list[OperatorRecord]:
    paths = _resolve_input_paths(
        Path(args.db_root),
        args.hardware,
        args.framework,
        args.framework_version,
        "moe",
    )
    if not paths[0].exists():
        raise FileNotFoundError(f"missing input JSONL: {paths[0]}")
    records = load_records(paths[0], args.hardware)
    if args.case_id:
        records = [r for r in records if r.source.get("case_id") == args.case_id]
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report logical internal breakdown for MoE roofline gaps.",
    )
    parser.add_argument("--db-root", default="collector/data/operator_db")
    parser.add_argument("--hardware", default="RTX_4090")
    parser.add_argument("--framework", default="vllm")
    parser.add_argument("--framework-version", default="0.19.1")
    parser.add_argument("--case-id", help="only report one collector case_id")
    parser.add_argument("--csv", help="write per-record CSV")
    parser.add_argument("--jsonl", help="write per-record JSONL")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    try:
        records = _resolve_records(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not records:
        print("no MoE records loaded", file=sys.stderr)
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
        rows.append(estimate_moe_breakdown(record, backends[mode]))

    print_summary(rows, top_n=args.top_n)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote CSV: {args.csv}")
    if args.jsonl:
        write_jsonl(rows, args.jsonl)
        print(f"wrote JSONL: {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
