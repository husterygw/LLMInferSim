"""阶段 2 e2e helper: 启动 vLLM 跑 Qwen3-4B, 输出 op dump 供测试抓取。

子进程隔离: pytest 的 test_run_qwen3_4b.py 通过 subprocess 调用这个 script,
检查 stdout/stderr 中:
  - 'SMOKE TEST PASSED'
  - 'init model=Qwen3-4B'
  - 'gate_proj' / 'up_proj' / 'down_proj' (SwiGLU 三段)
  - 'q_proj' / 'k_proj' / 'v_proj' (按 GQA)
  - phase=prefill / phase=decode 至少各出现一次
"""
from __future__ import annotations


def main() -> int:
    model = "/data1/home/ygw268/models/Qwen3-4B-Instruct-2507"

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, tp=1)")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        # 阶段 3 C 块 feature gate: 无真实 logits, 不能生成 logprobs
        max_logprobs=0,
        # 默认开 vLLM 自己的 stat logger (吞吐/running/waiting 数字), 调试时方便看
        # 调度行为; 不需要时改 True 关掉减少噪音
        disable_log_stats=False,
    )

    sp = SamplingParams(max_tokens=3, temperature=0.0)
    prompts = [
        TokensPrompt(prompt_token_ids=[10, 11, 12, 13, 14, 15, 16]),
        TokensPrompt(prompt_token_ids=[20, 21]),
        TokensPrompt(prompt_token_ids=[30, 31, 32, 33]),
    ]
    print(f"[run] llm.generate(num_prompts={len(prompts)}, max_tokens={sp.max_tokens})")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print("[out] req_id=%s  generated_token_ids=%s" % (
            o.request_id, o.outputs[0].token_ids,
        ))

    print("\nSMOKE TEST PASSED — Qwen3-4B ran end-to-end on VirtualPlatform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
