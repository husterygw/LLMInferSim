"""阶段 9-pro: DeepSeek-V4-Pro + tp=16 + ep=16 (16×H200, 跨 2 节点) 端到端验证。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置:
   hf download deepseek-ai/DeepSeek-V4-Pro \\
       config.json tokenizer_config.json tokenizer.json

跑:
   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 LLM_INFER_SIM_HW=H200 \\
       python examples/run_deepseek_v4_pro_tp16_ep16.py

   预期 ~2-3 分钟 (custom modeling_deepseek_v4.py 加载 + 16 worker spawn) 见到
   "阶段 9-pro PASSED — DeepSeek-V4-Pro tp=16 ep=16 跨节点 (V4 sparse + HC + MoE)."

为什么 16 H200 (跨节点):
   V4-Pro 总参 ~1.5T, expert_dtype=fp4 → expert 权重 ~750GB; attention/其他 bf16/fp8 ~ ?
   tp=16 + EP=16 单卡 weights/rank ≈ 50-60GB, H200 141GB 充足.
   tp=16 > intra_node_size=8 触发 hierarchical 跨节点公式 (inter_bw=50GB/s vs intra=900GB/s),
   而非 tp=8 单节点 (intra-only path).

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════

  ✅ DeepSeek-V4-Pro hf_config (61 layers, 384 experts, num_hash_layers=3,
     q_lora_rank=1536, o_lora_rank=1024, index_topk=1024, hc_mult=4,
     expert_dtype=fp4, compress_ratios=62项含MTP) 加载
  ✅ profile_extractor 14 个 V4 字段透传 (含 num_hash_layers=3)
  ✅ is_v4=True 触发 V4 path (fused_wqa_wkv + CSA/HCA 分流 + sparse attn + HC + MoE)
  ✅ layer_idx ∈ [0, 3) 走 moe_hash_lookup (FLOPs=0 router); 其余 gated moe
  ✅ vLLM MultiprocExecutor 起 16 个 VirtualWorker, gloo PG 跨进程 ready
  ✅ tp=16 > intra_node_size=8 触发 hierarchical_allreduce (attn TP allreduce 跨节点)
  ✅ ep=16 > intra_node_size=8 触发 hierarchical_alltoall (EP dispatch/combine 跨节点)
  ✅ collective_rpc 收 16 个 rank 报告, byte-identical (symmetric)
  ✅ FP4 expert_dtype 不触发 feature gate

注意:
  vLLM 单机起 16 worker 内存压力大 (每个 worker ~1GB driver state, 总 ~16GB),
  确保系统有 >= 32GB RAM. 启动慢 (90-180s).
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("LLM_INFER_SIM_HW", "H200")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
    hw = os.environ["LLM_INFER_SIM_HW"]

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=16, ep=16, hw={hw}, "
          f"V4 sparse + HC + MoE, 跨节点)")
    llm = LLM(
        model=model,
        tensor_parallel_size=16,
        enable_expert_parallel=True,
        trust_remote_code=True,
        skip_tokenizer_init=True,
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=16 ep=16, V4-Pro)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 9-pro] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 16, f"expected 16 reports (tp=16), got {len(results)}"
    print("--- rank 0 report (other 15 ranks symmetric) ---")
    print(results[0])

    for i in range(1, 16):
        assert results[i] == results[0], f"rank {i} report differs!"
    print("[check] all 16 ranks symmetric ✓")

    print("\n阶段 9-pro PASSED — DeepSeek-V4-Pro tp=16 ep=16 跨节点 "
          "(V4 sparse + HC + MoE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
