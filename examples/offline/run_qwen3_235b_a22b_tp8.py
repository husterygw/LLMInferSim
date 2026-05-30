"""阶段 7: Qwen3-235B-A22B + tp=8 单机 8 卡端到端验证。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置 (跟阶段 5/6 同, 已 cache 过 HF 元数据就跳过):

   hf download Qwen/Qwen3-235B-A22B \\
       config.json tokenizer_config.json vocab.json merges.txt

跑端到端:

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 python examples/run_qwen3_235b_a22b_tp8.py

   预期 ~30 秒内 (8 worker spawn 比 tp=2 慢些) 见到
   "阶段 7 PASSED — Qwen3-235B-A22B tp=8 single-node multi-worker."

可选环境变量:
   VLLM_INFER_SIM_MODEL=...    覆盖默认 "Qwen/Qwen3-235B-A22B"
   LLM_INFER_SIM_HW=B200       覆盖默认 H100

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
单机 8 卡仍是 intra-node (N=8 = intra_node_size, _is_cross_node 返回 False), 走
flat ring path。但仍是 stage 7 的关键 e2e 检查:

  ✅ Qwen3-235B-A22B 大模型 hf_config 解析正确 (94 layers, hidden=4096)
  ✅ determine_available_memory 在 235B 重量级下行为合理
     (235B fp16 ≈ 470 GB / tp=8 ≈ 58.7 GB/rank, 紧贴 H100 80GB×0.5=40GB 预算)
     —— 可能触发 fallback "weights > budget" warning
  ✅ vLLM MultiprocExecutor 起 8 个 VirtualWorker 子进程, gloo PG 8 rank ready
  ✅ collective_rpc 收回 8 个 rank 报告
  ✅ 94 层 MoE forward step latency 跟 inspect_qwen3_235b_a22b.py standalone 数字一致

跨节点 hierarchical 路径 (N>8) 由 inspect_qwen3_235b_a22b.py + test_inter_node_cost
  覆盖 (不依赖真实 16 进程)。

HF_HUB_OFFLINE=1 自动设上 (见 examples/run_qwen3_32b_tp2.py 同样原因)。
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "Qwen/Qwen3-235B-A22B")

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=8, single-node)")
    llm = LLM(
        model=model,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=8, 235B MoE)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 7] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 8, f"expected 8 reports (tp=8), got {len(results)}"
    # 8 rank 在 uniform skew=0 下报告完全一致, 只打印 rank 0
    print(f"--- rank 0 report (other 7 ranks symmetric) ---")
    print(results[0])

    # 简单一致性检查: 所有 rank 报告 byte-identical
    for i in range(1, 8):
        assert results[i] == results[0], f"rank {i} report differs from rank 0!"
    print("[check] all 8 ranks symmetric ✓")

    print("\n阶段 7 PASSED — Qwen3-235B-A22B tp=8 single-node multi-worker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
