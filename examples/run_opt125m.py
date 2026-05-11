"""Smoke test: 真正启动 vLLM LLM 实例, 跑一条 prompt, 验证空 step 链路。

run:
    VLLM_VIRTUAL_BACKEND=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    VLLM_USE_V1=1 \
    python tests/smoke_run_one_step.py

预期成功标准:
  - 进程不 crash
  - VirtualPlatform / VirtualWorker 路径被走到 (看日志)
  - LLM.generate() 正常返回 (即便 token 是 fake 的)
"""
from __future__ import annotations

import os


def main() -> int:
    # 选择 opt-125m: 小, 纯文本, hf_config 已在本地缓存
    model = os.environ.get("VLLM_INFER_SIM_MODEL", "facebook/opt-125m")

    from vllm import LLM, SamplingParams

    print(f"[init] LLM(model={model!r}, tp=1)")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,           # 没真实模型, 也别走 cudagraph
        max_model_len=512,
        max_num_seqs=4,
        disable_log_stats=True,
    )

    sp = SamplingParams(max_tokens=3, temperature=0.0)
    # 多个请求 + 不等长 prompt → 同时观察 prefill 和 decode 的 phase 分类
    from vllm import TokensPrompt
    prompts = [
        TokensPrompt(prompt_token_ids=[10, 11, 12, 13, 14, 15, 16]),  # 7 token prompt
        TokensPrompt(prompt_token_ids=[20, 21]),                      # 2 token prompt
        TokensPrompt(prompt_token_ids=[30, 31, 32, 33]),              # 4 token prompt
    ]
    print(f"[run] llm.generate(num_prompts={len(prompts)}, max_tokens={sp.max_tokens})")
    outs = llm.generate(prompts, sampling_params=sp)
    for o in outs:
        print("[out] req_id=%s  generated_token_ids=%s" % (
            o.request_id, o.outputs[0].token_ids,
        ))

    print("\nSMOKE TEST PASSED — LLM ran end-to-end on VirtualPlatform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
