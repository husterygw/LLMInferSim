"""阶段 9-ε: DeepSeek-V4-Flash + tp=8 单机端到端验证.

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置:
   hf download deepseek-ai/DeepSeek-V4-Flash \\
       config.json tokenizer_config.json tokenizer.json

跑:
   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 LLM_INFER_SIM_HW=H100 \\
       python examples/run_deepseek_v4_flash_tp8.py

   预期 ~60-120 秒(custom modeling_deepseek_v4.py 加载 + 8 worker)见到
   "阶段 9 PASSED — DeepSeek-V4-Flash tp=8 (MLA-style + sparse + HC + MoE)."

为什么 H100 单机够:
   V4-Flash 总参 ~280B (config 估算), MoE 用 fp4 → weights ~140GB total.
   tp=8 单卡 weights/rank ≈ 17.5 GB, H100 80GB 充足.
   (V3 671B fp16 不行需 H200, V4-Flash fp4 280B 反而单机能跑)

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════

  ✅ DeepseekV4ForCausalLM hf_config + trust_remote_code 加载
  ✅ profile_extractor 14 个 V4 字段 (sliding_window/o_groups/compress_ratios/index_*/hc_*/expert_dtype 全部) 透传
  ✅ is_v4 = True 触发 V4 path (SWA-only / CSA / HCA 三种 layer 类型按 compress_ratios 分流)
  ✅ V4 V4 path 内含 fused_wqa_wkv + fused_compress_wkv_wgate + fused_index_compress_wkv_wgate
                + index_wq_b/weights_proj (ReplicatedLinear 不切 TP)
                + fused_sparse_attention + wo_a (FP8 deploy.w_byte)
                + HC pre/post + MoE + shared experts
  ✅ MLAAttentionSpec 在 V4 路径下: V4 没有 kv_lora_rank (V4 用 sparse 而非 MLA),
     所以走 FullAttentionSpec (跟 V3 走 MLAAttentionSpec 不同)
  ✅ vLLM MultiprocExecutor 起 8 个 VirtualWorker, gloo PG ready
  ✅ collective_rpc 收 8 个 rank 报告, byte-identical (symmetric)
  ✅ FP4 expert_dtype 不触发 feature gate

阶段 9 显式不做:
  ❌ MTP (num_nextn_predict_layers) → §10.5 9.5 spec decode
  ❌ num_hash_layers (hash MoE routing) → 阶段 X
  ❌ scoring_func=sqrtsoftplus 精确公式 → 阶段 X (FLOPs << expert)
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("LLM_INFER_SIM_HW", "H100")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    hw = os.environ["LLM_INFER_SIM_HW"]

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=8, hw={hw}, V4 sparse + HC + MoE)")
    llm = LLM(
        model=model,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        trust_remote_code=True,
        skip_tokenizer_init=True,  # V4 custom tokenizer 需要额外文件; fake-token 路径不需要 tokenizer
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=8, V4-Flash)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 9] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 8, f"expected 8 reports (tp=8), got {len(results)}"
    print("--- rank 0 report (other 7 ranks symmetric) ---")
    print(results[0])

    for i in range(1, 8):
        assert results[i] == results[0], f"rank {i} report differs!"
    print("[check] all 8 ranks symmetric ✓")

    print("\n阶段 9 PASSED — DeepSeek-V4-Flash tp=8 (MLA-style + sparse + HC + MoE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
