"""测 in-context AllReduce sync floor (MoE decode 关键路径).

目的:验证 Stage M-A TPOT -53% gap 的真实根因 — sim 假设 AllReduce 之间 NCCL pipelining
能 hide sync overhead, 但 MoE decode 路径有 fused_moe + topk gating 等较重 kernel 分散
在 AllReduce 之间, NCCL pipeline 被打断, 每次 AllReduce 都付完整 latency.

v2 改动 (2026-05-18):
  - 之前 v1 用 tiny softmax 当 inter-AR kernel, 只测到 ~47 µs floor (太轻)
  - 真 vLLM profile 显示 in-context AR 平均 ~649 µs (大 ~10×)
  - v2 改用 MoE-like kernel chain (router + grouped expert GEMM + activation + down GEMM)
    模拟真实 MoE decode 在 AR 之间的 kernel 堆积

三档对比, 同样 size 同样 N 次:
  A) isolated:   纯 AllReduce × N
  B) tiny:       softmax + AR + softmax × N (v1, 轻 kernel disruption)
  C) moe_like:   router + grouped_GEMM + silu + down_GEMM + AR × N (重 kernel, 接近真实)

Δ(C - A) 应该 ≈ 500-650 µs 才能解释 Stage M-A 28ms/step 的 comm gap.

Sizes 锁定 MoE decode 单 token:
  - 4 KB (hidden=2048 × bf16, batch=1)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29612 \\
    /data1/home/ygw268/llm_sim/LLMInferSim/scripts/measure/measure_allreduce_in_context.py
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Allow importing _meas_common from /scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from _meas_common import make_record, env_metadata  # noqa


def init_dist() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def _bench_isolated(tensor: torch.Tensor, n_iters: int, warmup: int) -> list[float]:
    """连续 AllReduce, NCCL pipelining 可以 hide sync."""
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        t1.record()
        torch.cuda.synchronize()
        times_us.append(t0.elapsed_time(t1) * 1000)  # ms→us
    return times_us


def _tiny_kernel(scratch: torch.Tensor) -> torch.Tensor:
    """v1: 轻 softmax — 几乎 0 cost,仅触发 kernel launch + GPU stream sync."""
    return torch.softmax(scratch, dim=-1)


# v2 MoE-like kernel chain config — 接近 Qwen3-30B-A3B TP=4 per-rank decode 实际 shape
HIDDEN = 2048
N_EXPERTS = 128
TOP_K = 8
FFN_PER_RANK = 768 // 4   # = 192


def _moe_like_kernels(
    x: torch.Tensor,
    router_w: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """模拟 MoE decode 一层在 AR 之间夹的 kernel 堆 (router + grouped expert + activation + down).

    跟真实 vLLM fused_moe 不完全一致 (vLLM 用 grouped GEMM batch 8 expert),
    但保持 kernel 数量 + 总 FLOPs 量级 + GPU stream 占用近似:
      1. router GEMM: [1, 2048] × [2048, 128]            (gate logits)
      2. softmax + topk: [1, 128] → [1, 8] 选 top-8
      3. fused expert gate+up: [1, 2048] × [2048, 192*8]  ≈ vllm grouped GEMM
      4. silu * up half
      5. fused expert down: [1, 192*8] × [192*8, 2048]
    """
    # 1. router
    router_logits = x @ router_w               # [1, 128]
    # 2. softmax + topk
    probs = torch.softmax(router_logits, dim=-1)
    _, _ = torch.topk(probs, TOP_K, dim=-1)
    # 3. fused expert gate+up (假 grouped: top-8 expert 的 ffn shard 拼成一个 GEMM)
    gate_up = x @ gate_up_w                     # [1, 192*8 = 1536]
    # 4. silu * up (split half/half)
    half = gate_up.shape[-1] // 2
    gate = torch.nn.functional.silu(gate_up[..., :half])
    up = gate_up[..., half:]
    hidden_act = gate * up                       # [1, half=768]
    # 5. fused expert down
    out = hidden_act @ down_w                   # [1, hidden=2048]
    return out


def _bench_in_context(
    tensor: torch.Tensor, scratch: torch.Tensor, n_iters: int, warmup: int
) -> list[float]:
    """v1: tiny_softmax + AllReduce + tiny_softmax × N (轻 disruption baseline)."""
    for _ in range(warmup):
        _tiny_kernel(scratch)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        _tiny_kernel(scratch)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _tiny_kernel(scratch)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        _tiny_kernel(scratch)
        t1.record()
        torch.cuda.synchronize()
        times_us.append(t0.elapsed_time(t1) * 1000)
    return times_us


def _bench_moe_in_context(
    tensor: torch.Tensor,
    x: torch.Tensor,
    router_w: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
    n_iters: int,
    warmup: int,
) -> list[float]:
    """v2: moe_like kernel chain + AllReduce × N (重 disruption, 接近真实 MoE decode)."""
    for _ in range(warmup):
        _ = _moe_like_kernels(x, router_w, gate_up_w, down_w)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _ = _moe_like_kernels(x, router_w, gate_up_w, down_w)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        t1.record()
        torch.cuda.synchronize()
        times_us.append(t0.elapsed_time(t1) * 1000)
    return times_us


def _bench_moe_only(
    x: torch.Tensor,
    router_w: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
    n_iters: int,
    warmup: int,
) -> list[float]:
    """moe_like kernel chain WITHOUT AllReduce — 拿到纯 inter-AR kernel 时长."""
    for _ in range(warmup):
        _ = _moe_like_kernels(x, router_w, gate_up_w, down_w)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _ = _moe_like_kernels(x, router_w, gate_up_w, down_w)
        t1.record()
        torch.cuda.synchronize()
        times_us.append(t0.elapsed_time(t1) * 1000)
    return times_us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="/tmp/allreduce_in_context.jsonl")
    args = ap.parse_args()

    rank, world_size, local_rank = init_dist()
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"world_size={world_size}, n_iters={args.n_iters}, warmup={args.warmup}")

    # Scratch for tiny kernel (v1) — fixed small size (1024 elem fp32 = 4 KB)
    scratch = torch.randn(1024, device=device, dtype=torch.float32)
    # MoE-like weights (v2)
    dtype = torch.bfloat16
    x = torch.randn(1, HIDDEN, device=device, dtype=dtype)
    router_w = torch.randn(HIDDEN, N_EXPERTS, device=device, dtype=dtype)
    gate_up_w = torch.randn(HIDDEN, FFN_PER_RANK * 2 * TOP_K, device=device, dtype=dtype)
    half = (FFN_PER_RANK * 2 * TOP_K) // 2
    down_w = torch.randn(half, HIDDEN, device=device, dtype=dtype)

    # ---- baseline: 纯 inter-AR kernel chain (无 AR) ----
    moe_only_times = _bench_moe_only(x, router_w, gate_up_w, down_w, args.n_iters, args.warmup)
    moe_only_med = statistics.median(moe_only_times)
    if rank == 0:
        print(f"\n[baseline] moe_like kernel chain only (no AR): {moe_only_med:.1f} µs")

    # ---- 三档 in-context AR 对比 ----
    if rank == 0:
        print(f"\n{'size':>10} {'iso_us':>10} {'v1_us':>10} {'v2_us':>10} {'Δv1_us':>10} {'Δv2_us':>10}")
        print("-" * 75)

    # Key sizes (重点是 MoE decode 4KB)
    SIZES_BYTES = [
        1024,       # 1 KB
        4096,       # 4 KB  ← MoE decode hidden=2048 × bf16 = key size
        16 * 1024,  # 16 KB
        65536,      # 64 KB
    ]

    records = []
    for nbytes in SIZES_BYTES:
        nelem = max(1, nbytes // 4)
        tensor = torch.randn(nelem, device=device, dtype=torch.float32)

        iso_times = _bench_isolated(tensor, args.n_iters, args.warmup)
        v1_times = _bench_in_context(tensor, scratch, args.n_iters, args.warmup)
        v2_times = _bench_moe_in_context(
            tensor, x, router_w, gate_up_w, down_w, args.n_iters, args.warmup
        )

        iso_med = statistics.median(iso_times)
        v1_med = statistics.median(v1_times)
        v2_med = statistics.median(v2_times)
        # v2 减掉 baseline moe_only 才是 in-context AR 真实时间
        v2_ar_only = v2_med - moe_only_med
        delta_v1 = v1_med - iso_med
        delta_v2 = v2_ar_only - iso_med

        if rank == 0:
            print(f"{nbytes:>10} {iso_med:>10.1f} {v1_med:>10.1f} "
                  f"{v2_med:>10.1f} {delta_v1:>+10.1f} {delta_v2:>+10.1f}")
            rec = make_record(
                script_name="measure_allreduce_in_context",
                world_size=world_size,
                bytes=nbytes,
                isolated_us_median=iso_med,
                v1_tiny_us_median=v1_med,
                v1_floor_us=delta_v1,
                v2_moelike_us_median=v2_med,
                v2_moelike_minus_baseline_us=v2_ar_only,
                v2_moelike_floor_us=delta_v2,
                moe_only_baseline_us=moe_only_med,
            )
            rec["env"] = env_metadata()
            records.append(rec)

    if rank == 0:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(records)} records to {out_path}")
        print(f"\n关键观察 (4 KB MoE decode size):")
        for r in records:
            if r["bytes"] == 4096:
                print(f"  isolated AR              = {r['isolated_us_median']:.1f} µs")
                print(f"  v1 tiny-softmax floor    = {r['v1_floor_us']:+.1f} µs  ← 上轮误测")
                print(f"  v2 moe-like floor        = {r['v2_moelike_floor_us']:+.1f} µs  ← 接近真实")
                print(f"  vLLM profile 实测/AR     = ~650 µs  ← 目标对照")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
