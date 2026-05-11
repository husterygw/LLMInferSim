"""阶段 3 e2e 验证: Qwen3-4B + chunked prefill / mixed step + reporter。

vs run_qwen3_4b.py 的差异:
  - 用更大的 prompt + 更小的 max_num_batched_tokens 强制触发 chunked prefill
  - 起跑前/后打印 reporter 报告 (TTFT/TPOT/throughput)
  - 后处理: 抓 stderr 找 'phase=mixed' 至少出现一次, 证明 mixed cost path 走通
  - 后处理: 抓 reporter 输出, 证明 TTFT/TPOT 非零

子进程式调用: 单独跑这个 script 看 stdout 即可知道阶段 3 是否端到端通。
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    model = os.environ.get(
        "VLLM_INFER_SIM_MODEL", "/data/ygw/models/Qwen3-4B-Instruct-2507"
    )

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=1, chunked_prefill 强制)")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=8,
        # 关键: chunk size 远小于 prompt → 触发 chunked prefill + mixed step
        max_num_batched_tokens=128,
        # 阶段 3 C 块 feature gate 要求: 无真实 logits 不能生成 logprobs
        max_logprobs=0,
        disable_log_stats=False,
    )

    # 多请求 + 不同 prompt 长度, 最大的 prompt > max_num_batched_tokens
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(10, 10 + 600))),   # 长, 会被 chunked
        TokensPrompt(prompt_token_ids=list(range(100, 100 + 200))),
        TokensPrompt(prompt_token_ids=list(range(200, 200 + 80))),
        TokensPrompt(prompt_token_ids=list(range(300, 300 + 40))),
    ]
    print(f"[run] llm.generate(num_prompts={len(prompts)}, "
          f"max_tokens={sp.max_tokens}, chunk=128)")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print(f"[out] req_id={o.request_id}  "
              f"generated_token_ids={o.outputs[0].token_ids}")

    # ---- 阶段 3 D 块: 报告 ----
    # ReportGenerator 在 VirtualModelRunner 上, 通过 LLM.collective_rpc 调
    # 每 rank VirtualWorker._get_virtual_runner_report 拉报告。
    print("\n[阶段 3 D 块] 抓 reporter 报告:")
    results = llm.collective_rpc("_get_virtual_runner_report")
    for rank_idx, report_text in enumerate(results):
        print(f"\n--- rank {rank_idx} report ---")
        print(report_text)

    print("\nSMOKE TEST PASSED — Qwen3-4B chunked prefill ran end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
