"""实测 NCCL AllReduce latency α + 有效带宽 β on TP=4 RTX 4090.

Ring AllReduce on n GPUs:
  time = 2 * (n-1) * (alpha + bytes / (n * beta))

扫描 byte size, 线性拟合 → α (intercept) + β (slope).

Usage:
  torchrun --nproc_per_node=4 /tmp/measure_allreduce.py
"""
import os, time, statistics
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"world_size={world_size}")

    # 测各种 size, 多次取 median
    SIZES_BYTES = [
        16, 64, 256, 1024, 4096, 16384,
        64*1024, 256*1024, 1024*1024,
        4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024,
    ]
    REPEATS = 30
    WARMUP = 10

    if rank == 0:
        print(f"\n{'bytes':>12} {'tensor':>10} {'median_us':>12} {'p10_us':>10} {'p90_us':>10}")
        print("-" * 60)

    measurements = []
    for nbytes in SIZES_BYTES:
        nelem = max(1, nbytes // 4)   # fp32 elements
        t = torch.zeros(nelem, device=f"cuda:{local_rank}", dtype=torch.float32)

        # warmup
        for _ in range(WARMUP):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # time
        times_us = []
        for _ in range(REPEATS):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) * 1000)  # ms→us

        times_us.sort()
        med = times_us[len(times_us)//2]
        p10 = times_us[len(times_us)//10]
        p90 = times_us[len(times_us)*9//10]

        if rank == 0:
            print(f"{nbytes:>12} {nelem:>10} {med:>12.2f} {p10:>10.2f} {p90:>10.2f}")
        measurements.append((nbytes, med))

    if rank == 0:
        # 拟合 ring allreduce 公式: time = 2(n-1) * (alpha + bytes / (n * beta))
        # 改写: time / (2(n-1)) = alpha + bytes / (n * beta)
        # 设 y = time/(2(n-1)), x = bytes/n
        #     y = alpha + x / beta
        # OLS: 取大于 1KB 的点拟合 (小 size 全是 latency, 不参与 BW 拟合)
        big = [(x, y) for (x, y) in measurements if x >= 1024]
        small = [(x, y) for (x, y) in measurements if x < 1024]

        coeff = 2 * (world_size - 1)
        # 大 size 拟合 β
        if len(big) >= 2:
            # y' = time / coeff, x' = bytes / world_size
            xs = [b / world_size for b, _ in big]
            ys = [t / coeff for _, t in big]
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            intercept = my - slope * mx
            beta_GBps = 1 / slope / 1000   # us per byte → GB/s = 1/slope * 1e6 / 1e9
            print(f"\n=== large-size fit (>=1KB) ===")
            print(f"  intercept α (per-segment overhead): {intercept:.2f} us")
            print(f"  slope 1/β: {slope:.6e} us/byte")
            print(f"  effective β: {beta_GBps:.2f} GB/s")

        # 小 size: 直接看 ring latency = coeff * α → α = time / coeff
        if small:
            small_alphas = [t / coeff for _, t in small]
            avg_alpha = statistics.median(small_alphas)
            print(f"\n=== small-size latency (<1KB) ===")
            print(f"  median α (per-segment): {avg_alpha:.2f} us")
            print(f"  inferred raw link_latency: {avg_alpha:.2f} us")

        # ring 全公式重新预测一组对照
        print(f"\n=== predictions vs actual (using fit) ===")
        if len(big) >= 2:
            print(f"{'bytes':>12} {'actual_us':>12} {'pred_us':>10} {'gap%':>8}")
            for b, actual in measurements:
                pred = coeff * (intercept + b / (world_size * (1/slope)))
                gap = (pred - actual) / actual * 100 if actual > 0 else 0
                print(f"{b:>12} {actual:>12.2f} {pred:>10.2f} {gap:>7.1f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
