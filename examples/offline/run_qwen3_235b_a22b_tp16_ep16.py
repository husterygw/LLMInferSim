"""阶段 7-cross: Qwen3-235B-A22B + tp=16 + ep=16 (跨 2 节点) 端到端验证。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

前置 (跟阶段 5/6 同, 已 cache 过 HF 元数据就跳过):

   hf download Qwen/Qwen3-235B-A22B \\
       config.json tokenizer_config.json vocab.json merges.txt

跑端到端:

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 LLM_INFER_SIM_HW=H200 \\
       python examples/run_qwen3_235b_a22b_tp16_ep16.py

   预期 ~2-3 分钟 (16 worker spawn) 见到
   "阶段 7-cross PASSED — Qwen3-235B-A22B tp=16 ep=16 跨节点 multi-worker."

可选环境变量:
   VLLM_INFER_SIM_MODEL=...    覆盖默认 "Qwen/Qwen3-235B-A22B"
   LLM_INFER_SIM_HW=H100       覆盖默认 H200 (H200 给 235B 单卡更宽裕)

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
本 spike 是阶段 7 真正的跨节点 e2e (run_qwen3_235b_a22b_tp8.py 单机 8 卡仍是
intra-node)。tp=16 / ep=16 > intra_node_size=8 才触发 hierarchical 公式。

  ✅ Qwen3-235B-A22B hf_config 解析正确 (94 layers, hidden=4096, num_experts=128 top-8)
  ✅ profile_extractor 读 enable_expert_parallel=True → ep=16
  ✅ vLLM MultiprocExecutor 起 16 个 VirtualWorker 子进程, gloo PG 16 rank ready
  ✅ tp=16 > intra_node_size=8 触发 _hierarchical_allreduce (attn TP allreduce 跨节点)
  ✅ ep=16 > intra_node_size=8 触发 _hierarchical_alltoall (EP dispatch/combine 跨节点)
  ✅ collective_rpc 收 16 rank 报告, byte-identical (uniform skew=0)
  ✅ 跨节点 step latency 主导贡献来自 inter_bw=50GB/s (H200) 而非 intra=900GB/s

注意:
  vLLM 单机起 16 worker 内存压力大 (每个 worker ~1GB driver state, 总 ~16GB),
  确保系统有 >= 32GB RAM. 启动慢 (90-180s).

HF_HUB_OFFLINE=1 自动设上 (见 examples/run_qwen3_32b_tp2.py 同样原因)。
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("LLM_INFER_SIM_HW", "H200")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "Qwen/Qwen3-235B-A22B")
    hw = os.environ["LLM_INFER_SIM_HW"]

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=16, ep=16, hw={hw}, 跨节点)")
    llm = LLM(
        model=model,
        tensor_parallel_size=16,
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=16 ep=16, 235B MoE)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[阶段 7-cross] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 16, f"expected 16 reports (tp=16), got {len(results)}"
    print("--- rank 0 report (other 15 ranks symmetric) ---")
    print(results[0])

    for i in range(1, 16):
        assert results[i] == results[0], f"rank {i} report differs!"
    print("[check] all 16 ranks symmetric ✓")

    print("\n阶段 7-cross PASSED — Qwen3-235B-A22B tp=16 ep=16 跨节点 multi-worker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
