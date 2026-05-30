"""测真实 vLLM MoE decode 路径的 kernel 分布(用 torch.profiler).

替代 measure_moe_kernel_launch.py 的 naive synthetic 测试:那个用 Python loop 1200 个
独立 matmul 不代表 vLLM 实际行为 — vLLM 用 FusedMoE / grouped GEMM, 实际 kernel 数
远小于 1200.

本脚本直接 profile 真实 vLLM Qwen3-30B-A3B TP=4 decode, 从 trace 抓:
  - 每 decode step 实际 kernel count
  - 每 kernel category 总时间 (gemm / attn / fused_moe / nccl / activation / ...)
  - 跟 sim 给的 compute/memory/comm 分解对照

跑法 (两步):
  1) 启动带 profiler 的 vllm serve (后台)
  2) 跑 1 prompt + 几个 decode token 触发 /start_profile + /stop_profile
  3) parse trace JSON

Usage:
  bash scripts/measure_moe_decode_profile.py.sh    # 包装脚本,负责启动 server + bench + parse
  或直接调用 parse 部分:
  python scripts/measure_moe_decode_profile.py --parse /tmp/moe_prof/*.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---- kernel name → category ----
CATEGORIES = [
    # (regex, category, description)
    (r"nccl|all_?reduce|all_?gather|reduce_?scatter|all2all|alltoall", "comm",        "NCCL collectives"),
    (r"fused_moe|moe_align|moe_gather|grouped_gemm|expert", "moe_fused",   "fused MoE kernels"),
    (r"flash.?attn|paged.?attn|attention",                  "attention",   "attention kernels"),
    (r"gemm|matmul|cublas|cutlass|gemv",                    "gemm",        "dense GEMM"),
    (r"rms_?norm|layer_?norm|norm",                         "norm",        "normalization"),
    (r"silu|gelu|swiglu|swish|activation",                  "activation",  "activation"),
    (r"softmax|topk|top_k|gating|routing",                  "routing",     "MoE routing"),
    (r"sample|logits|categorical|multinomial",              "sampling",    "sampler"),
    (r"copy|memcpy|cast|to_dtype|gather|scatter|index_select", "memory",   "memory copies"),
    (r"embed",                                              "embedding",   "embedding lookup"),
]


def categorize(name: str) -> str:
    n = name.lower()
    for pattern, cat, _ in CATEGORIES:
        if re.search(pattern, n):
            return cat
    return "other"


def parse_trace(trace_path: Path) -> dict:
    """Parse Chrome trace JSON (gz or plain). Returns dict of category → stats."""
    if str(trace_path).endswith(".gz"):
        with gzip.open(trace_path, "rt") as f:
            trace = json.load(f)
    else:
        with trace_path.open() as f:
            trace = json.load(f)

    events = trace.get("traceEvents", [])
    # Filter to GPU kernel events only (cat = "kernel" or device type)
    by_cat: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0, "samples": []})
    total_kernel_us = 0.0
    total_kernel_count = 0
    for e in events:
        # Only count kernel events (not CPU launch overhead, not framework annotations)
        # Chrome trace standard: ph='X' (complete event), cat contains 'kernel' or 'gpu'
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        # Match GPU/kernel categories from torch.profiler
        if not (("kernel" in cat.lower()) or ("gpu" in cat.lower()) or ("cuda" in cat.lower())):
            continue
        name = e.get("name", "")
        dur = float(e.get("dur", 0))  # microseconds
        category = categorize(name)
        by_cat[category]["count"] += 1
        by_cat[category]["total_us"] += dur
        if len(by_cat[category]["samples"]) < 3:
            by_cat[category]["samples"].append(name)
        total_kernel_us += dur
        total_kernel_count += 1

    return {
        "trace": str(trace_path),
        "total_kernel_count": total_kernel_count,
        "total_kernel_us": total_kernel_us,
        "by_category": dict(by_cat),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", help="Path to trace JSON(.gz) to parse")
    ap.add_argument("--out", default="/tmp/moe_decode_profile.json")
    args = ap.parse_args()

    if not args.parse:
        print("Usage: --parse <trace.json[.gz]>")
        print("Run measure_moe_decode_profile.py.sh first to generate trace.")
        return 1

    trace_path = Path(args.parse)
    if not trace_path.exists():
        # try glob
        matches = list(Path("/tmp/moe_prof").glob("**/*.json*"))
        if matches:
            trace_path = sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
            print(f"using latest trace: {trace_path}")
        else:
            print(f"not found: {args.parse}")
            return 1

    result = parse_trace(trace_path)
    print(f"\ntrace: {result['trace']}")
    print(f"total kernels: {result['total_kernel_count']}")
    print(f"total kernel time: {result['total_kernel_us']/1000:.2f} ms")
    print(f"\n{'category':<14} {'count':>8} {'total_ms':>12} {'%':>6}  sample names")
    print("-" * 90)
    by_cat = result["by_category"]
    for cat, stats in sorted(by_cat.items(), key=lambda x: -x[1]["total_us"]):
        pct = stats["total_us"] / max(result["total_kernel_us"], 1) * 100
        samples = ", ".join(stats["samples"][:2])
        print(f"{cat:<14} {stats['count']:>8} {stats['total_us']/1000:>12.2f} {pct:>5.1f}%  {samples[:60]}")

    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
