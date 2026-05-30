"""测 MoE decode 路径上大量小 GEMM 的 eager-mode kernel launch overhead.

Stage M-A TPOT -53% 在排查 in-context AllReduce 之后, 嫌疑转到:
  MoE decode 每 token 触发 1000+ 个小 GEMM 启动 (8 active expert × 3 GEMM × 48 layer),
  eager mode 每 GEMM kernel launch ~20-30µs, 累计 20-30ms — 跟实测 gap 量级吻合.

测三档:
  A) eager (back-to-back): 真实 MoE decode 路径模拟, 每个 GEMM 独立 kernel launch
  B) cudagraph: 同样 GEMM, capture 进 graph 一次性 replay, 消除 launch overhead
  C) baseline: 只跑 dense (Qwen3-4B-like) decode 模式, 作对照

Δ(A - B) = eager kernel launch overhead per decode step
Δ(A - C) = MoE 比 dense 多的"启动税"

Qwen3-30B-A3B + TP=4 per-rank workload:
  - 48 layers
  - per layer: router (1 GEMM) + 8 active expert × 3 GEMM (gate, up, down) = 25 GEMMs
  - per rank shard: ffn/tp = 768/4 = 192
  - total per decode step: ~1200 small GEMMs

跑法 (单 GPU,无需 TP):
  CUDA_VISIBLE_DEVICES=0 python scripts/measure_moe_kernel_launch.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from _meas_common import make_record, env_metadata


# ---- model dims (Qwen3-30B-A3B, TP=4 per-rank) ----
HIDDEN = 2048
FFN_PER_RANK = 768 // 4   # = 192 (TP=4)
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8


def build_weights(device, dtype):
    """Pre-allocate weights for one MoE decode step (per-rank, TP=4).

    每层:
      - router: [hidden, num_experts] = [2048, 128]
      - 128 experts × (gate_up [2048, 2*192], down [192, 2048])
        但 decode 时只激活 top-8, 我们也只为 top-8 准备 weight (其余不读)
    """
    layers = []
    for _ in range(NUM_LAYERS):
        router = torch.randn(HIDDEN, NUM_EXPERTS, device=device, dtype=dtype)
        # 每层 8 个 active expert 用同样 buffer (只测启动开销, 不测正确性)
        experts = []
        for _ in range(TOP_K):
            gate = torch.randn(HIDDEN, FFN_PER_RANK, device=device, dtype=dtype)
            up   = torch.randn(HIDDEN, FFN_PER_RANK, device=device, dtype=dtype)
            down = torch.randn(FFN_PER_RANK, HIDDEN, device=device, dtype=dtype)
            experts.append((gate, up, down))
        layers.append({"router": router, "experts": experts})
    return layers


def run_one_decode_moe(x: torch.Tensor, layers: list) -> torch.Tensor:
    """模拟一次 MoE decode step. x shape: [1, hidden]."""
    for layer in layers:
        # router (1 GEMM)
        _ = x @ layer["router"]   # [1, num_experts]

        # 8 active experts × 3 GEMMs
        h = x
        for gate_w, up_w, down_w in layer["experts"]:
            g = h @ gate_w        # [1, ffn_per_rank]
            u = h @ up_w          # [1, ffn_per_rank]
            silu_gu = torch.nn.functional.silu(g) * u
            h = silu_gu @ down_w  # [1, hidden]
        x = h
    return x


def run_one_decode_dense(x: torch.Tensor, layers: list) -> torch.Tensor:
    """对照: dense decode (只 1 个 FFN, 不走 expert 路径). 用 layers[0] 的 expert[0] 当 dense FFN."""
    for layer in layers:
        gate_w, up_w, down_w = layer["experts"][0]
        g = x @ gate_w
        u = x @ up_w
        silu_gu = torch.nn.functional.silu(g) * u
        x = silu_gu @ down_w
    return x


def bench_eager(x: torch.Tensor, layers: list, n_iters: int, warmup: int, fn) -> list[float]:
    """Eager mode (每 GEMM 独立 launch)."""
    for _ in range(warmup):
        _ = fn(x.clone(), layers)
    torch.cuda.synchronize()
    times_us = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _ = fn(x.clone(), layers)
        t1.record()
        torch.cuda.synchronize()
        times_us.append(t0.elapsed_time(t1) * 1000)
    return times_us


def bench_cudagraph(x: torch.Tensor, layers: list, n_iters: int, warmup: int, fn) -> list[float]:
    """CUDA Graph (capture + replay,消除 launch overhead)."""
    # warmup with side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = fn(x.clone(), layers)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    x_static = x.clone()
    with torch.cuda.graph(g):
        out = fn(x_static, layers)

    # warmup replay
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(n_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        g.replay()
        t1.record()
        torch.cuda.synchronize()
        times_us.append(t0.elapsed_time(t1) * 1000)
    return times_us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="/tmp/moe_kernel_launch.jsonl")
    args = ap.parse_args()

    device = "cuda:0"
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    print(f"device={torch.cuda.get_device_name()}, dtype={dtype}")
    print(f"Qwen3-30B-A3B per-rank shape: hidden={HIDDEN}, ffn/tp={FFN_PER_RANK}, "
          f"layers={NUM_LAYERS}, top_k={TOP_K}")
    print(f"per decode step:")
    print(f"  MoE   = {NUM_LAYERS} layers × (1 router + {TOP_K} expert × 3 GEMM) = "
          f"{NUM_LAYERS * (1 + TOP_K * 3)} GEMMs")
    print(f"  dense = {NUM_LAYERS} layers × 3 GEMM = {NUM_LAYERS * 3} GEMMs")
    print()

    layers = build_weights(device, dtype)
    x = torch.randn(1, HIDDEN, device=device, dtype=dtype)

    results = {}
    for label, fn, mode in [
        ("MoE eager",       run_one_decode_moe,   "eager"),
        ("MoE cudagraph",   run_one_decode_moe,   "graph"),
        ("dense eager",     run_one_decode_dense, "eager"),
        ("dense cudagraph", run_one_decode_dense, "graph"),
    ]:
        if mode == "eager":
            times = bench_eager(x, layers, args.n_iters, args.warmup, fn)
        else:
            times = bench_cudagraph(x, layers, args.n_iters, args.warmup, fn)
        results[label] = times

    print(f"{'pattern':<20} {'median_us':>12} {'p10_us':>10} {'p90_us':>10}  {'ms':>8}")
    print("-" * 68)
    medians = {}
    for label, times in results.items():
        med = statistics.median(times)
        p10 = statistics.quantiles(times, n=10)[0]
        p90 = statistics.quantiles(times, n=10)[8]
        medians[label] = med
        print(f"{label:<20} {med:>12.1f} {p10:>10.1f} {p90:>10.1f}  {med/1000:>8.2f}")

    print()
    print("=== Δ analysis ===")
    moe_launch_oh = medians["MoE eager"] - medians["MoE cudagraph"]
    dense_launch_oh = medians["dense eager"] - medians["dense cudagraph"]
    moe_extra = medians["MoE eager"] - medians["dense eager"]
    print(f"MoE launch overhead   = MoE eager - MoE cudagraph     = {moe_launch_oh:7.1f} µs ({moe_launch_oh/1000:.2f} ms)")
    print(f"dense launch overhead = dense eager - dense cudagraph = {dense_launch_oh:7.1f} µs ({dense_launch_oh/1000:.2f} ms)")
    print(f"MoE 'startup tax'     = MoE eager - dense eager       = {moe_extra:7.1f} µs ({moe_extra/1000:.2f} ms)")
    print()
    print("Stage M-A 待解释: TPOT real-sim gap ≈ 25-33 ms / step")
    print(f"  本测 MoE launch overhead = {moe_launch_oh/1000:.2f} ms → 是否吻合?")

    # 落 JSONL
    rec = make_record(
        script_name="measure_moe_kernel_launch",
        hidden=HIDDEN, ffn_per_rank=FFN_PER_RANK,
        num_layers=NUM_LAYERS, top_k=TOP_K,
        n_iters=args.n_iters, warmup=args.warmup,
        median_us={k: statistics.median(v) for k, v in results.items()},
        moe_launch_overhead_us=moe_launch_oh,
        dense_launch_overhead_us=dense_launch_oh,
        moe_extra_vs_dense_us=moe_extra,
    )
    rec["env"] = env_metadata()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
