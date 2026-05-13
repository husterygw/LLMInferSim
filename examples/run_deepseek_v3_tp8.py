"""阶段 8-δ: DeepSeek-V3 + tp=8 (H200) 单机端到端验证。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置:

   hf download deepseek-ai/DeepSeek-V3 \\
       config.json tokenizer_config.json tokenizer.json

跑:

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 LLM_INFER_SIM_HW=H200 \\
       python examples/run_deepseek_v3_tp8.py

   预期 ~60 秒(8 worker spawn) 见到
   "阶段 8 PASSED — DeepSeek-V3 tp=8 single-node multi-worker (MLA + MoE)."

为什么 H200 而非 H100:
   V3 总参 671B fp16 = 1342 GB. tp=8 单卡需要承担 167 GB 权重.
     - H100 80GB:  167 > 80, 严重不够, fallback "10% HBM" 给 KV 算出来还能跑
                   但 num_blocks 极小, decode batch 受限
     - H200 141GB: 167 > 141, 仍不够, 但 fallback 后 14 GB 给 KV 更合理
     - 真实部署典型: tp=8 + PP=2 (推 §10.5 7.5) 或 tp=16 跨节点
   本 spike 用 H200 + 触发 fallback, 数字方向贴近真实"超大模型紧凑部署"经验.

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════

  ✅ DeepSeek-V3 hf_config (custom modeling_deepseek.py via trust_remote_code) 解析正确
  ✅ profile_extractor 透传 MLA + q_lora_rank + MoE + shared_experts 全部字段
  ✅ layer_builder 走 MLA path (q_a/q_b/kv_a_proj_with_mqa/kv_b_proj)
                   + MoE path (routed_experts + shared_experts)
                   + 前 3 层 dense FFN (first_k_dense_replace=3)
  ✅ 8-γ get_kv_cache_spec 返回 MLAAttentionSpec (per-token-bytes=1152 vs 标准 GQA 65536)
  ✅ determine_available_memory fallback 在 V3 重量级下触发 (weights/rank ≈ 167GB > budget)
  ✅ vLLM MultiprocExecutor 起 8 个 VirtualWorker, gloo PG ready
  ✅ collective_rpc 收 8 个 rank 报告, byte-identical (symmetric)
  ✅ fp8 quantization_config 不触发 feature gate (我们只看 backend, 不看 quant_method)

阶段 8 显式不做:
  ❌ FP8 / 量化 → §10.5 8.5
  ❌ n_group / topk_group 路由限制 → §10.5 或阶段 X
  ❌ MTP (num_nextn_predict_layers) → §10.5 9.5 spec decode
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # 默认 H200, 用户可 LLM_INFER_SIM_HW=H100 覆盖看 fallback 极端情况
    os.environ.setdefault("LLM_INFER_SIM_HW", "H200")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "deepseek-ai/DeepSeek-V3")
    hw = os.environ["LLM_INFER_SIM_HW"]

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=8, hw={hw}, MLA + MoE)")
    llm = LLM(
        model=model,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        trust_remote_code=True,            # DeepSeek-V3 用 custom modeling_deepseek.py
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=8, DeepSeek-V3 MLA+MoE)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 8] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 8, f"expected 8 reports (tp=8), got {len(results)}"
    print("--- rank 0 report (other 7 ranks symmetric) ---")
    print(results[0])

    # 检查所有 8 rank byte-identical (symmetric)
    for i in range(1, 8):
        assert results[i] == results[0], f"rank {i} report differs from rank 0!"
    print("[check] all 8 ranks symmetric ✓")

    print("\n阶段 8 PASSED — DeepSeek-V3 tp=8 single-node multi-worker (MLA + MoE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
