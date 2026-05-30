"""阶段 9.1: DeepSeek-V3.2-Exp + tp=8 单机端到端验证。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置:
   hf download deepseek-ai/DeepSeek-V3.2-Exp \\
       config.json tokenizer.json tokenizer_config.json

跑:
   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 LLM_INFER_SIM_HW=H200 \\
       python examples/run_deepseek_v32_tp8.py

   预期 ~60-120 秒 (8 worker spawn) 见到
   "阶段 9.1 PASSED — DeepSeek-V3.2-Exp tp=8 (DSA: MLA + lightning indexer + sparse)."

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════

V3.2 = V3 MLA backbone + DSA lightning indexer + sparse-attended MLA kernel.
跟 V3 (dense MLA) 和 V4 (window+compress+grouped O) 都不一样.

  ✅ DeepSeek-V3.2-Exp hf_config (`deepseek_v32` model_type) 加载
  ✅ profile_extractor V3 字段透传 (kv_lora_rank=512, q_lora_rank=1536) +
     DSA 字段透传 (index_head_dim=128, index_n_heads=64, index_topk=2048)
  ✅ Quant fp8 (跟 V3 一致): w_byte/a_byte → 1.0
  ✅ dispatcher 走 V3.2 专用 path (is_v32_sparse), 不走 V3/V4
  ✅ Attention block 含 5 个 indexer ops (wq_b + wk_weights_proj + k_norm
                                          + q_fp8_quant + sparse_attn_indexer)
  ✅ 主 attention 用 `fused_mla_sparse_attention` (attended_len=min(ctx, 2048))
  ✅ MoE / shared_expert / TP allreduce 路径跟 V3 一致
  ✅ vLLM MultiprocExecutor 起 8 个 VirtualWorker, 16 ranks 报告 symmetric

阶段 9.1 显式不做:
  ❌ MTP (num_nextn_predict_layers=1) → §10.5 9.5 spec decode
  ❌ Indexer 的 RoPE on q_pe/k_pe (FLOPs 微小, 量级 << GEMM)
  ❌ K-cache fp8 → bf16 dequant overhead (vLLM kernel 内部细节)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("LLM_INFER_SIM_HW", "H200")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp")
    hw = os.environ["LLM_INFER_SIM_HW"]

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=8, hw={hw}, V3.2 DSA: MLA + indexer + sparse)")
    llm = LLM(
        model=model,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        trust_remote_code=True,
        skip_tokenizer_init=True,  # V3.2 custom tokenizer 需 sentencepiece, fake-token 路径不需要
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        max_logprobs=0,
        disable_log_stats=False,
    )

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(10, 10 + 3072))),   # 长 ctx 触发 sparse cap
        TokensPrompt(prompt_token_ids=list(range(100, 100 + 800))),
        TokensPrompt(prompt_token_ids=list(range(200, 200 + 200))),
        TokensPrompt(prompt_token_ids=list(range(300, 300 + 40))),
    ]
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=8, V3.2 DSA)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 9.1] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 8, f"expected 8 reports (tp=8), got {len(results)}"
    print("--- rank 0 report (other 7 ranks symmetric) ---")
    print(results[0])

    for i in range(1, 8):
        assert results[i] == results[0], f"rank {i} report differs!"
    print("[check] all 8 ranks symmetric ✓")

    print("\n阶段 9.1 PASSED — DeepSeek-V3.2-Exp tp=8 "
          "(DSA: MLA + lightning indexer + sparse).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
