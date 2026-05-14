"""Prefix Caching e2e 验证: enable_prefix_caching=True 同前缀第二次跑应 cache 命中。

设计:
  - vLLM enable_prefix_caching 是 block 级 (默认 16 tok/block)。
  - Batch 1 用一个 800 token 的 long prompt P。
  - Batch 2 用 [P + 100 个新 tail token], 前 800 tok 命中 cache, 只算 100 tail 的 prefill。
  - 通过新加的 `_get_per_request_metrics` rpc 拉每个请求的 sim-time TTFT。
  - 断言:
      * Batch 2 TTFT 显著小于 Batch 1 TTFT (期望 < 0.5×, 因为 cache 命中 ~7/8 长度)。
      * Batch 1 + Batch 2 request_id 不重叠 (vLLM 给每个 generate 调用新 req_id)。

验证我们的 cost model 是否透明支持 prefix caching:
  step_extractor 直接读 vLLM 的 num_computed_tokens / num_scheduled_tokens,
  num_computed_tokens 是 vLLM 命中 cache 后告诉我们的"已计算 token 数",
  我们只对 num_scheduled_tokens 部分计 cost — 自动得到加速。

⚠️ 内存上限假设 (P2):
  我们 `determine_available_memory` 按 HBM × util - weights - activations 算 KV budget,
  不感知 vLLM PrefixCache block allocator 的去重收益。真机 prefix caching ON 下
  共享 prefix 的并发请求会复用同一组 block, 实际能装的 max_num_seqs 比我们的
  budget 算出来的更大 — 这是**保守偏差**, 不会让 cost 偏快; 但 throughput simulation
  会比真机偏低。准确建模需扩展 KV block allocator (阶段 ~7.6 PD 分离也涉及)。

跑:
  conda activate llm_sim
  VLLM_VIRTUAL_BACKEND=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_USE_V1=1 \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    LLM_INFER_SIM_TIME_MODE=instant \\
    python examples/run_prefix_caching.py

退出码: 0 = PASSED, 非 0 = 验证失败 (TTFT 比例不对 / cache 没命中)。
"""
from __future__ import annotations

import os
import sys


def _print_batch_metrics(label: str, metrics: list[dict]) -> None:
    print(f"\n--- {label} per-request metrics ---")
    print(f"  {'req_id':<12} {'ttft(ms)':>10} {'tpot(ms)':>10} {'steps':>6} {'out_tok':>8}")
    for m in metrics:
        print(
            f"  {m['request_id']:<12} "
            f"{m['ttft']*1e3:>10.3f} {m['tpot']*1e3:>10.3f} "
            f"{m['num_steps_scheduled']:>6} {m['output_tokens']:>8}"
        )


def main() -> int:
    model = os.environ.get(
        "VLLM_INFER_SIM_MODEL", "/data/ygw/models/Qwen3-4B-Instruct-2507"
    )

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"[init] LLM(model={model!r}, enable_prefix_caching=True)")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=8192,
        max_num_seqs=4,
        max_logprobs=0,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )

    # 共享前缀 (3500 tok); tail 只 50 tok。
    # 长 prompt 让 prefill 远大于 decode, TTFT 直接反映 prefill 节省。
    SHARED_PREFIX_LEN = 3500
    TAIL_LEN_BATCH2 = 50
    shared_prefix = list(range(1000, 1000 + SHARED_PREFIX_LEN))
    tail_b2 = list(range(20000, 20000 + TAIL_LEN_BATCH2))

    sp = SamplingParams(max_tokens=4, temperature=0.0)

    # ---- Batch 1: cold cache, 全 prompt 都要 prefill ----
    print(f"\n[batch 1] cold cache, prompt_len={SHARED_PREFIX_LEN}")
    batch1_prompts = [
        TokensPrompt(prompt_token_ids=shared_prefix),
    ]
    llm.generate(batch1_prompts, sampling_params=sp)
    batch1_metrics = llm.collective_rpc("_get_per_request_metrics")[0]
    _print_batch_metrics("Batch 1 (cold)", batch1_metrics)

    seen_req_ids = {m["request_id"] for m in batch1_metrics}

    # ---- Batch 2: warm cache, 前 800 应命中, 只算 100 tail ----
    print(f"\n[batch 2] warm cache, prompt_len={SHARED_PREFIX_LEN + TAIL_LEN_BATCH2} "
          f"(前 {SHARED_PREFIX_LEN} 应命中 cache, 只 prefill 末尾 {TAIL_LEN_BATCH2})")
    batch2_prompts = [
        TokensPrompt(prompt_token_ids=shared_prefix + tail_b2),
    ]
    llm.generate(batch2_prompts, sampling_params=sp)
    all_metrics = llm.collective_rpc("_get_per_request_metrics")[0]
    batch2_metrics = [m for m in all_metrics if m["request_id"] not in seen_req_ids]
    _print_batch_metrics("Batch 2 (warm)", batch2_metrics)

    # ---- 验证 ----
    if not batch1_metrics or not batch2_metrics:
        print(f"[FAIL] batch1={len(batch1_metrics)} batch2={len(batch2_metrics)} 应各 1 个")
        return 1

    ttft1 = batch1_metrics[0]["ttft"]
    ttft2 = batch2_metrics[0]["ttft"]
    if ttft1 <= 0 or ttft2 <= 0:
        print(f"[FAIL] TTFT 为 0: batch1={ttft1} batch2={ttft2}")
        return 2

    ratio = ttft2 / ttft1
    # Batch 2 实际 prefill tokens = 50 / 3500 ≈ 1.4%, prefill 主导 TTFT 时
    # 比例应远小于 1; TTFT 还含 1 个 decode step ~3ms 公共项, 故阈值 0.3
    print(f"\n[verify] Batch 1 TTFT = {ttft1*1e3:.3f} ms")
    print(f"[verify] Batch 2 TTFT = {ttft2*1e3:.3f} ms")
    print(f"[verify] Batch 2 / Batch 1 ratio = {ratio:.3f} (期望 < 0.3)")

    if ratio >= 0.3:
        print(f"\n[FAIL] Batch 2 TTFT 没有显著下降 — prefix caching 可能未生效")
        print(f"       可能原因: vLLM 没真启用 prefix caching, 或 step_extractor "
              f"没正确透传 num_computed_tokens")
        return 3

    print(f"\nPREFIX CACHING e2e PASSED — Batch 2 TTFT 降到 Batch 1 的 {ratio*100:.1f}%, "
          f"证明 step_extractor 正确透传 num_computed_tokens, cost model 只对未缓存 "
          f"token 计费。")

    # 打印聚合报告供肉眼检查
    print("\n--- final aggregate report ---")
    reports = llm.collective_rpc("_get_virtual_runner_report")
    print(reports[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
