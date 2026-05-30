"""Profile measured vLLM fused_moe kernels for one standalone collector case.

Breaks down measured GPU kernel time: runs the same standalone vLLM
fused_experts path used by the MoE collector, exports a torch.profiler trace,
and aggregates CUDA kernels by name-derived categories.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CATEGORIES = [
    (r"moe_align|align_block|count_and_sort|moe_sort|sort|histogram|inclusive_scan|cumsum",
     "moe_align_sort"),
    (r"reduce_kernel|combine|weighted_sum", "moe_reduce_combine"),
    (r"permute|unpermute|moe_gather|moe_scatter|scatter|gather|index_select",
     "moe_permute_gather"),
    (r"fused_moe|grouped_gemm|moe_gemm|cutlass|cublas|gemm|matmul",
     "moe_grouped_gemm"),
    (r"silu|swiglu|swish|activation|mul", "moe_activation"),
    (r"topk|top_k|softmax|routing|gating", "routing"),
    (r"copy|memcpy|cast|transpose|contiguous|fill|zero", "memory_misc"),
]

FIELDNAMES = [
    "category",
    "kernel_count",
    "total_us",
    "percent",
    "avg_us",
    "max_us",
    "sample_kernels",
]


def categorize_kernel(name: str) -> str:
    lowered = name.lower()
    for pattern, category in CATEGORIES:
        if re.search(pattern, lowered):
            return category
    return "other"


def _open_trace(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open()


def parse_trace(trace_path: Path | str) -> dict[str, Any]:
    path = Path(trace_path)
    with _open_trace(path) as f:
        trace = json.load(f)

    by_cat: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"kernel_count": 0, "total_us": 0.0, "max_us": 0.0, "samples": []}
    )
    total_us = 0.0
    total_count = 0
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", "")).lower()
        # Keep GPU kernel/device events.  Exclude cuda_runtime/cuda_driver API
        # calls such as cudaLaunchKernel/cudaDeviceSynchronize; CUDA events used
        # by the collector measure device work, not CPU API duration.
        if not ("kernel" in cat or "gpu" in cat):
            continue
        name = str(event.get("name", ""))
        dur = float(event.get("dur", 0.0))
        if dur <= 0:
            continue
        bucket = categorize_kernel(name)
        stats = by_cat[bucket]
        stats["kernel_count"] += 1
        stats["total_us"] += dur
        stats["max_us"] = max(float(stats["max_us"]), dur)
        if len(stats["samples"]) < 4:
            stats["samples"].append(name)
        total_us += dur
        total_count += 1

    rows = []
    for category, stats in sorted(by_cat.items(), key=lambda item: -item[1]["total_us"]):
        count = int(stats["kernel_count"])
        total = float(stats["total_us"])
        rows.append({
            "category": category,
            "kernel_count": count,
            "total_us": total,
            "percent": total / total_us * 100.0 if total_us > 0 else 0.0,
            "avg_us": total / count if count > 0 else 0.0,
            "max_us": float(stats["max_us"]),
            "sample_kernels": " | ".join(stats["samples"]),
        })
    return {
        "trace_path": str(path),
        "total_kernel_count": total_count,
        "total_kernel_us": total_us,
        "rows": rows,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path | str) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(result: dict[str, Any]) -> None:
    print(f"trace: {result['trace_path']}")
    print(f"total kernels: {result['total_kernel_count']}")
    print(f"total kernel time: {result['total_kernel_us']:.3f} us")
    print()
    print(
        f"{'category':<20} {'count':>8} {'total_us':>12} "
        f"{'%':>7} {'avg_us':>10} {'max_us':>10}  sample kernels"
    )
    print("-" * 118)
    for row in result["rows"]:
        print(
            f"{row['category']:<20} {row['kernel_count']:>8} "
            f"{row['total_us']:>12.3f} {row['percent']:>6.1f}% "
            f"{row['avg_us']:>10.3f} {row['max_us']:>10.3f}  "
            f"{str(row['sample_kernels'])[:42]}"
        )


def _profile_standalone(args: argparse.Namespace) -> Path:
    import torch
    import vllm
    from torch.profiler import ProfilerActivity, profile
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.layer import determine_expert_map

    from collector.runners._vllm_dist import ensure_initialized
    from collector.runners.vllm_moe import _build_routing, _moe_dtype_to_torch

    device = f"cuda:{args.device}"
    torch.cuda.set_device(device)
    dtype = _moe_dtype_to_torch(args.moe_dtype)

    if args.moe_tp_size > 1 and args.moe_ep_size > 1:
        raise NotImplementedError("vLLM fused_experts collector path supports TP or EP, not both")
    if args.inter_size % args.moe_tp_size != 0:
        raise ValueError("inter_size must be divisible by moe_tp_size")
    if args.num_experts % args.moe_ep_size != 0:
        raise ValueError("num_experts must be divisible by moe_ep_size")

    local_inter = args.inter_size // args.moe_tp_size
    local_experts = args.num_experts // args.moe_ep_size
    trace_path = Path(args.trace or tempfile.mktemp(suffix=".json", prefix="moe_fused_"))

    with set_current_vllm_config(VllmConfig()):
        ensure_initialized(args.device)
        x = torch.randn((args.num_tokens, args.hidden_size), dtype=dtype, device=device)
        w1 = torch.randn(
            (local_experts, 2 * local_inter, args.hidden_size),
            dtype=dtype,
            device=device,
        )
        w2 = torch.randn(
            (local_experts, args.hidden_size, local_inter),
            dtype=dtype,
            device=device,
        )
        topk_weights, topk_ids = _build_routing(
            args.num_tokens, args.topk, args.num_experts, args.distribution, device,
        )
        if args.moe_ep_size > 1:
            _, expert_map, _ = determine_expert_map(
                ep_size=args.moe_ep_size,
                ep_rank=args.ep_rank,
                global_num_experts=args.num_experts,
            )
            expert_map = expert_map.to(device) if expert_map is not None else None
        else:
            expert_map = None

        def run_once() -> None:
            fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                global_num_experts=args.num_experts,
                expert_map=expert_map,
            )

        for _ in range(args.warmup):
            run_once()
        torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities, record_shapes=False) as prof:
            for _ in range(args.iters):
                run_once()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_path))

    print(
        f"profiled vLLM {vllm.__version__}: "
        f"tokens={args.num_tokens}, hidden={args.hidden_size}, inter={args.inter_size}, "
        f"topk={args.topk}, experts={args.num_experts}, "
        f"tp={args.moe_tp_size}, ep={args.moe_ep_size}, iters={args.iters}"
    )
    return trace_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile measured standalone vLLM fused_moe kernels.",
    )
    parser.add_argument("--parse", help="parse an existing torch profiler trace")
    parser.add_argument("--trace", help="write/read chrome trace path")
    parser.add_argument("--csv", help="write category CSV")
    parser.add_argument("--json", help="write full parsed JSON")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--inter-size", type=int, default=768)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--moe-tp-size", type=int, default=1)
    parser.add_argument("--moe-ep-size", type=int, default=4)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--distribution", default="balanced")
    parser.add_argument("--moe-dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    trace_path = Path(args.parse) if args.parse else _profile_standalone(args)
    result = parse_trace(trace_path)
    _print_report(result)

    if args.csv:
        _write_csv(result["rows"], args.csv)
        print(f"\nwrote CSV: {args.csv}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"wrote JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
