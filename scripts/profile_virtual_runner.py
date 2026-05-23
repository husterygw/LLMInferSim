#!/usr/bin/env python3
"""cProfile VirtualModelRunner.execute_step in isolation.

bench_compare.sh 端到端测出来 sim_TPOT - sim_cost_engine_latency ≈ 10ms/step 是
LLMInferSim Python glue 开销 (cost engine 之外的所有 wrapper). 要决定砍哪一段,
先量化各 sub-step 的 tottime / cumtime.

此脚本不起 vllm server, 不走 API. 直接在本进程内:
  1. EngineArgs → VllmConfig (借 vllm 解析 model config, 不分配 GPU)
  2. 实例化 VirtualModelRunner
  3. Seed N 个 fake "decode-ready" 请求到 _request_states
  4. 用 SimpleNamespace 模拟 SchedulerOutput (parallel arrays + decode 单 token)
  5. cProfile 包住 N=100 个 decode step, 出 top 30 by tottime / cumtime

LLM_INFER_SIM_TIME_MODE=instant 让 time_emulator 不真 sleep, 这样测的是纯 Python.

Usage:
  python scripts/profile_virtual_runner.py
  python scripts/profile_virtual_runner.py --batch 16 --steps 200 --ctx-len 1024
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from types import SimpleNamespace

# env 必须在 import vllm 之前
os.environ.setdefault("VLLM_VIRTUAL_BACKEND", "1")
os.environ.setdefault("LLM_INFER_SIM_HW", "RTX_4090")
os.environ.setdefault("LLM_INFER_SIM_TIME_MODE", "instant")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("VLLM_USE_V1", "1")


def make_decode_so(req_ids, ctx_lens, num_outputs):
    """Synthetic SchedulerOutput for pure-decode step (all reqs in cached path)."""
    cached = SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens=list(ctx_lens),
        num_output_tokens=list(num_outputs),
        new_block_ids=[],
        resumed_from_preemption=[False] * len(req_ids),
    )
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={rid: 1 for rid in req_ids},
        total_num_scheduled_tokens=len(req_ids),
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/ygw/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--batch", type=int, default=10,
                    help="并发 decode 请求数 (= 默认 stage A 配置)")
    ap.add_argument("--steps", type=int, default=100, help="profile 多少 decode step")
    ap.add_argument("--warmup", type=int, default=20, help="跳过前 N step (JIT / cache warmup)")
    ap.add_argument("--ctx-len", type=int, default=256,
                    help="每 req 的 starting context length")
    ap.add_argument("--out", default="/tmp/profile_virtual_runner",
                    help="输出 prefix (生成 .prof + .txt)")
    ap.add_argument("--top", type=int, default=30, help="top N 行打印")
    args = ap.parse_args()

    # ---- 1. VllmConfig (会调 hugginface tokenizer config 解析, 不分 GPU) ----
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=max(args.batch, 16),
        max_num_batched_tokens=8192,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        skip_tokenizer_init=True,
        max_logprobs=0,
    )
    vllm_config = engine_args.create_engine_config()

    # block allocator 需要 num_gpu_blocks; 真路径走 determine_available_memory,
    # 这里我们直接灌一个数, 让 KVBlockAllocator init 成功.
    if not getattr(vllm_config.cache_config, "num_gpu_blocks", None):
        vllm_config.cache_config.num_gpu_blocks = 4096
    if not getattr(vllm_config.cache_config, "block_size", None):
        vllm_config.cache_config.block_size = 16

    # ---- 2. VirtualModelRunner ----
    from llm_infer_sim.adapters.vllm.virtual_model_runner import VirtualModelRunner

    runner = VirtualModelRunner(vllm_config)

    # ---- 3. seed _request_states (跳过 prefill, 直接进入稳态 decode) ----
    req_ids = [f"r{i}" for i in range(args.batch)]
    prompt_len = args.ctx_len
    for rid in req_ids:
        runner._request_states[rid] = {
            "target_output_len": 128,
            "prompt_token_ids": [1] * prompt_len,
        }

    # ---- 4. warmup ----
    for step in range(args.warmup):
        so = make_decode_so(
            req_ids,
            ctx_lens=[prompt_len + step] * len(req_ids),
            num_outputs=[step] * len(req_ids),
        )
        runner.execute_step(so, step_id=step)

    # ---- 5. profile ----
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for step in range(args.warmup, args.warmup + args.steps):
        so = make_decode_so(
            req_ids,
            ctx_lens=[prompt_len + step] * len(req_ids),
            num_outputs=[step] * len(req_ids),
        )
        runner.execute_step(so, step_id=step)
    pr.disable()
    elapsed = time.perf_counter() - t0

    per_step_ms = elapsed / args.steps * 1000
    print()
    print(f"=== summary ===")
    print(f"  steps    = {args.steps}")
    print(f"  batch    = {args.batch}")
    print(f"  ctx_len  = {args.ctx_len}")
    print(f"  wall     = {elapsed*1000:.1f} ms")
    print(f"  per step = {per_step_ms:.2f} ms")
    print()

    # ---- 6. dump ----
    pr.dump_stats(args.out + ".prof")

    out_txt = open(args.out + ".txt", "w")

    def emit(header: str, sort_key: str):
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(args.top)
        body = buf.getvalue()
        out_txt.write(f"\n{header}\n")
        out_txt.write(body)
        print(f"\n{header}\n{body}")

    emit(f"=== top {args.top} by tottime (self time) ===", "tottime")
    emit(f"=== top {args.top} by cumulative ===", "cumulative")

    out_txt.close()
    print(f"saved: {args.out}.prof  +  {args.out}.txt")


if __name__ == "__main__":
    main()
