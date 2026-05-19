"""多 GPU collector runners (NCCL collective + tp/ep MoE).

主 scheduler 不能跑这些 (entry.multi_gpu=True), 用 torchrun 启动:

    torchrun --nproc-per-node=4 -m collector.distributed.run_collective \\
        --shape-profiles qwen3_30b_a3b --topology concentrated

只 rank 0 写 JSONL / checkpoint, 其他 rank 参与 NCCL 但不落盘.
"""
