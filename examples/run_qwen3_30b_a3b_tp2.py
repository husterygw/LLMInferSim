"""阶段 5: Qwen3-30B-A3B MoE 入门 (TP=2, 不开 EP)。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

1. 拉 HF 元数据 (只要 config + tokenizer, 不拉 ~60GB MoE 权重):

   hf download Qwen/Qwen3-30B-A3B \\
       config.json tokenizer_config.json vocab.json merges.txt

2. 跑 spike:

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 python examples/run_qwen3_30b_a3b_tp2.py

   预期 ~10 秒内见到 "阶段 5 PASSED — Qwen3-30B-A3B tp=2 MoE end-to-end."

3. 可选环境变量:

   VLLM_INFER_SIM_MODEL=...    覆盖默认 "Qwen/Qwen3-30B-A3B"
   LLM_INFER_SIM_HW=B200       覆盖默认 H100 硬件 profile

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
  ✅ Qwen3MoeForCausalLM hf_config 解析正确 (num_experts=128, top_k=8,
     moe_intermediate_size=768)
  ✅ profile_extractor 兼容 Qwen `num_experts` 字段 (不只是 DeepSeek
     `n_routed_experts`); is_moe=True 触发 _build_moe_ffn_block 路径
  ✅ determine_available_memory 真实公式: 30B fp16 × top_k expert read
     ≈ small fraction (因为 FusedMoE 只读 active experts)
  ✅ routed_experts op 真实激活, comm_time 体现 TP allreduce after expert
  ✅ MoE 模型 decode 远快于 dense 32B (3B 激活 vs 32B 全部)

═══════════════════════════════════════════════════════════════════════════════
                         阶段 5 显式不做 (对齐 §10)
═══════════════════════════════════════════════════════════════════════════════
  ❌ EP (enable_ep + AllToAll dispatch/combine) → 阶段 6
  ❌ 负载不均衡 / expert load imbalance → 阶段 6
  ❌ 跨节点通信修正 → 阶段 7

HF_HUB_OFFLINE=1 自动设上 (见 examples/run_qwen3_32b_tp2.py 同样原因)。
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "Qwen/Qwen3-30B-A3B")

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=2, MoE)")
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=2, MoE 128 experts top-8)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 5] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 2, f"expected 2 reports (tp=2), got {len(results)}"
    for rank_idx, report_text in enumerate(results):
        print(f"\n--- rank {rank_idx} report ---")
        print(report_text)

    print("\n阶段 5 PASSED — Qwen3-30B-A3B tp=2 MoE end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
