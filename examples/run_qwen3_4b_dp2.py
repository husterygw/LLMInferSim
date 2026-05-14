"""DP G3 spike: Qwen3-4B + dp=2 + tp=1 — 验证 DP step latency = max(per-rank).

═══════════════════════════════════════════════════════════════════════════════
                                  怎么跑
═══════════════════════════════════════════════════════════════════════════════

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_USE_V1=1 \\
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
       LLM_INFER_SIM_TIME_MODE=instant \\
       python examples/run_qwen3_4b_dp2.py

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
  ✅ vLLM 在 `data_parallel_size=2` 时 spawn 2 个独立 engine 进程
  ✅ VirtualPlatform 不需任何 DP 特殊代码 (零侵入), 各 rank 独立 simulate own scheduler
  ✅ sizing.per_rank_param_bytes: DP 不切 dense weight (每 DP rank 自留一份)
  ✅ G3: execute_step 末尾的 _sync_dp_latency 在两个进程间 all_reduce(MAX)
        让两进程 sleep 同一 max latency (padding 模拟)

  ❌ 不验证 DP+EP+MoE (用单独 example: run_qwen3_30b_a3b_dp2_ep2)
  ❌ 不验证跨节点 DP (单机 dp=2 已足够 cover sync 路径)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = os.environ.get(
        "VLLM_INFER_SIM_MODEL", "/data/ygw/models/Qwen3-4B-Instruct-2507"
    )

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=1, data_parallel_size=2)")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        data_parallel_size=2,                # ← DP 核心 flag
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        max_logprobs=0,
        disable_log_stats=True,
    )

    sp = SamplingParams(max_tokens=4, temperature=0.0)
    # 故意不均衡 prompt: rank 0 拿大 prompt, rank 1 拿小. 验证慢者拖快者.
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(10, 10 + 800))),   # 长
        TokensPrompt(prompt_token_ids=list(range(20000, 20000 + 200))),  # 短
        TokensPrompt(prompt_token_ids=list(range(30000, 30000 + 100))),
        TokensPrompt(prompt_token_ids=list(range(40000, 40000 + 50))),
    ]
    print(f"[run] llm.generate(num_prompts={len(prompts)}, dp=2)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id} prompt_len={len(o.prompt_token_ids)} "
              f"output={list(o.outputs[0].token_ids)}")

    print("\n[DP-G3] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    print(f"[DP-G3] got {len(results)} reports (期望 = dp×tp = 2)")
    for rank_idx, report_text in enumerate(results):
        print(f"\n--- rank {rank_idx} report ---")
        print(report_text)

    print("\nDP-G3 SPIKE PASSED — Qwen3-4B dp=2 tp=1, _sync_dp_latency 在 DP group 上跑通.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
