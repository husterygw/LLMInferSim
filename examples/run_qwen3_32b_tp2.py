"""阶段 4-ζ: Qwen3-32B + tensor_parallel_size=2 大模型端到端跑通。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

1. 拉 HF 元数据文件 (只要 config + tokenizer, 不拉 ~64GB 的 safetensors):

   huggingface-cli download Qwen/Qwen3-32B \\
       config.json tokenizer_config.json vocab.json merges.txt

   (若 huggingface-cli 命令找不到: `pip install huggingface_hub[cli]`)

2. 跑 spike:

   conda activate llm_sim
   VLLM_VIRTUAL_BACKEND=1 python examples/run_qwen3_32b_tp2.py

   预期 ~10 秒内见到 "4-ζ PASSED — Qwen3-32B tp=2 multi-worker end-to-end."

3. 可选环境变量:

   VLLM_INFER_SIM_MODEL=...    覆盖默认 "Qwen/Qwen3-32B" (HF id 或本地路径)
   LLM_INFER_SIM_HW=B200       覆盖默认 H100 (见 core/profiles/hardware.py)
   LLM_INFER_SIM_TIME_MODE=instant   秒级 sleep 改瞬时, 加速 smoke

═══════════════════════════════════════════════════════════════════════════════
                              这个 spike 验证什么
═══════════════════════════════════════════════════════════════════════════════
  ✅ 32B 模型 hf_config 解析正确 (GQA 64q/8kv, 64 层, hidden=5120)
  ✅ determine_available_memory 真实公式: 80GB×0.5 − 32.8GB(weight/rank)
     − 0.01GB(activation) ≈ 10.2GB 给 KV
  ✅ KV cache spec / num_blocks 不爆 (~4855 blocks)
  ✅ mixed step + TP allreduce comm_time 真实出现 (~372 µs/step prefill)
  ✅ MultiprocExecutor + gloo PG 多进程, 2 rank 报告通过 collective_rpc 收回

═══════════════════════════════════════════════════════════════════════════════
                          为什么需要 HF_HUB_OFFLINE=1
═══════════════════════════════════════════════════════════════════════════════
vLLM 在 platform plugin / "Enabled custom fusions" 之后仍会尝试 resolve 模型
safetensors index, 即使 VirtualWorker 是 config-only load_model 不下载权重。
网络受限或缓存不全时会**无限挂起** (sleeping on futex, 没有 timeout / 没有报错)。

HF_HUB_OFFLINE=1 强制只用 cache, 跳过 weight resolve —— script 已经
os.environ.setdefault 自动设上, 你不用手动 export。

详见记忆 feedback_hf_hub_offline.md。
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    # 阶段 4-ζ 排查发现: HF identifier + VirtualPlatform 路径下 vLLM 仍会尝试
    # 在 platform plugin 之后 resolve weight 文件, 必须显式 offline 否则挂起。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model = os.environ.get("VLLM_INFER_SIM_MODEL", "Qwen/Qwen3-32B")

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=2)")
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
    print(f"[run] llm.generate(num_prompts={len(prompts)}, tp=2, model=32B)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    print("\n[4-ζ] collective_rpc 抓每 rank reporter:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    assert len(results) == 2, f"expected 2 reports (tp=2), got {len(results)}"
    for rank_idx, report_text in enumerate(results):
        print(f"\n--- rank {rank_idx} report ---")
        print(report_text)

    print("\n4-ζ PASSED — Qwen3-32B tp=2 multi-worker end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
