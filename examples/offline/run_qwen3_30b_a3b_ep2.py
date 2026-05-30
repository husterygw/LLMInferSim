"""阶段 6-α spike: Qwen3-30B-A3B + tp=2 + enable_expert_parallel=True。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置 (跟阶段 5 同, 已 cache 过 HF 元数据就跳过):

   hf download Qwen/Qwen3-30B-A3B \\
       config.json tokenizer_config.json vocab.json merges.txt

跑 spike:

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 python examples/run_qwen3_30b_a3b_ep2.py

   预期 ~10 秒内见到 "6-α SPIKE PASSED — Qwen3-30B-A3B tp=2 ep=2 multi-worker."

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
按 mvp_scoping 哲学 de-risk EP 未知路径:
  ✅ vLLM `enable_expert_parallel=True` flag 在 EngineArgs 接住
     (单节点下 EP group = TP × DP, 这里 tp=2 dp=1 → ep_size=2)
  ✅ VirtualWorker 多进程下 EP group init 不爆 (gloo backend 跟 TP allreduce 共用)
  ✅ profile_extractor 把 enable_expert_parallel 透传到 ParallelConfig.enable_ep
  ✅ layer_builder._build_moe_ffn_block 走 ep>1 分支,
     注入 ep_alltoall_dispatch + ep_alltoall_combine ops
  ✅ alltoall_time 计算并累加到 t_comm
  ✅ TP allreduce 在 ep>1 时不再插 (routed_expert 已经做了 reduce)

═══════════════════════════════════════════════════════════════════════════════
                         阶段 6 显式不做 (对齐 §10)
═══════════════════════════════════════════════════════════════════════════════
  ❌ 跨节点 EP → 阶段 7
  ❌ Expert load imbalance microbench → 阶段 X
  ❌ per-rank cost asymmetric reporting → 6-γ 阶段判断 (uniform skew=0 下严格 symmetric)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "Qwen/Qwen3-30B-A3B")

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=2, enable_expert_parallel=True)")
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,        # ← 6-α spike 核心
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=128,
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=2, ep=2)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[6-α spike] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 2, f"expected 2 reports (tp=2), got {len(results)}"
    for rank_idx, report_text in enumerate(results):
        print(f"\n--- rank {rank_idx} report ---")
        print(report_text)

    print("\n6-α SPIKE PASSED — Qwen3-30B-A3B tp=2 ep=2 multi-worker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
