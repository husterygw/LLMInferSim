"""阶段 4-α spike: Qwen3-4B + tensor_parallel_size=2 真实多进程跑通。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

默认走本地路径 /data1/home/ygw268/models/Qwen3-4B-Instruct-2507 (不需要联网):

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 python examples/run_qwen3_4b_tp2.py

预期 ~5 秒内见到 "4-α SPIKE PASSED — Qwen3-4B tp=2 multi-worker end-to-end."

可选环境变量:
   VLLM_INFER_SIM_MODEL=...      覆盖默认本地路径
   LLM_INFER_SIM_HW=B200         覆盖默认 H100 硬件 profile
   LLM_INFER_SIM_TIME_MODE=instant  秒级 sleep 改瞬时, 加速 smoke

(如果用 HF identifier 形式 `Qwen/Qwen3-4B` 而非本地路径, 需要先
 `huggingface-cli download Qwen/Qwen3-4B config.json tokenizer_config.json
 vocab.json merges.txt` 并设 `HF_HUB_OFFLINE=1` —— 见
 examples/run_qwen3_32b_tp2.py 顶部说明。)

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
按 mvp_scoping 哲学 de-risk 多进程未知路径:
  ✅ MultiprocExecutor 起 2 个 VirtualWorker 子进程
  ✅ gloo PG 跨进程 init / barrier 跑通 (VirtualWorker.init_device 已用 gloo)
  ✅ VirtualPlatform 注入 worker_cls 在多进程下生效
  ✅ collective_rpc 返回所有 rank 的 reporter 报告 (rank=0 + rank=1)
  ✅ phase=prefill/mixed/decode 都跑过

显式不做 (推到 4-β / γ / δ):
  - per-rank cost 真实差异 (symmetric ranks 假设, cost 仍是阶段 3.5 单 rank)
  - Qwen3-32B (4-ζ 才上, 见 run_qwen3_32b_tp2.py)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    model = os.environ.get(
        "VLLM_INFER_SIM_MODEL", "/data1/home/ygw268/models/Qwen3-4B-Instruct-2507"
    )

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=1, chunked_prefill 强制)")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,                     # ← 4-α spike 核心
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=128,                  # 触发 chunked prefill + mixed step
        max_logprobs=0,
        disable_log_stats=False,
    )

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(10, 10 + 600))),
        TokensPrompt(prompt_token_ids=list(range(100, 100 + 200))),
        TokensPrompt(prompt_token_ids=list(range(200, 200 + 80))),
        TokensPrompt(prompt_token_ids=list(range(300, 300 + 40))),
    ]
    print(f"[run] llm.generate(num_prompts={len(prompts)}, "
          f"max_tokens={sp.max_tokens}, tp=2)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    # ---- 抓每 rank 的报告 ----
    print("\n[4-α spike] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 2, f"expected 2 reports (tp=2), got {len(results)}"
    for rank_idx, report_text in enumerate(results):
        print(f"\n--- rank {rank_idx} report ---")
        print(report_text)

    print("\n4-α SPIKE PASSED — Qwen3-4B tp=2 multi-worker end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
