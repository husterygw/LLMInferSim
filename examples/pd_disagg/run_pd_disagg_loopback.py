"""PD 分离 KV transfer cost e2e 验证 — sim-only env path (详设 §7.6).

设计:
  - 真 PD 分离要起 2+ 个 vllm serve (producer + consumer + router), 真 connector,
    多进程 / 多机协调; 是 deployment-level scenario, 不在 example 范围。
  - 我们走 simulator-only 路径: 设 `LLM_INFER_SIM_PD_ROLE=kv_both` 让 cost path
    认为本进程兼任 producer + consumer, 每个 prefill 完成都加 send cost; 但**不动
    vLLM 的 kv_transfer_config**, vLLM 内部完全不知道 PD 已"启用"。
  - 验证: 第二次同 LLM 实例下, PD on 比 PD off TTFT 多出 ~transfer_time。

跑:
  conda activate llm_sim
  bash examples/run_pd_disagg_loopback.sh
  # 或手动两次启 python, 第二次加 env

退出码: 0 = PASSED, 非 0 = 验证失败。
"""
from __future__ import annotations

import os
import sys


def _run_one(label: str, model: str) -> tuple[float, dict]:
    """跑一次 generate, 返回 (TTFT_seconds, pd_stats).

    PD 行为由当前 process env 决定 (LLM_INFER_SIM_PD_ROLE), 调用方负责设置。
    """
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="float16",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=4,
        max_logprobs=0,
        disable_log_stats=True,
    )

    sp = SamplingParams(max_tokens=4, temperature=0.0)
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(1000, 1000 + 2000))),
    ]
    llm.generate(prompts, sampling_params=sp)
    metrics = llm.collective_rpc("_get_per_request_metrics")[0]
    pd_stats = llm.collective_rpc("_get_pd_stats")[0]
    print(f"\n[{label}] TTFT = {metrics[0]['ttft']*1e3:.3f} ms")
    print(f"[{label}] PD stats: {pd_stats}")
    return metrics[0]["ttft"], pd_stats


def main() -> int:
    model = os.environ.get(
        "VLLM_INFER_SIM_MODEL", "/data1/home/ygw268/models/Qwen3-4B-Instruct-2507"
    )

    # 两个模式由调用方在 shell 里设 env 分别跑; 这里默认按当前 env 跑一次。
    role = os.environ.get("LLM_INFER_SIM_PD_ROLE", "").strip()
    if role == "":
        label = "baseline_pd_off"
    else:
        label = f"pd_on_role={role}"

    print(f"[mode] {label} (LLM_INFER_SIM_PD_ROLE={role!r})")
    ttft, pd_stats = _run_one(label, model)

    # 单次跑只验证: PD on 时确有 transfer; 完整 baseline 对比由 shell 脚本做
    if role:
        if pd_stats["pd_num_transfers"] < 1:
            print(f"\n[FAIL] PD role={role} 但未捕获 transfer 事件")
            return 1
        if pd_stats["pd_total_transfer_time_s"] <= 0:
            print(f"\n[FAIL] PD transfer_time=0, cost 未计入")
            return 2
        print(f"\nPD DISAGG single-run PASSED ({label})")
    else:
        print(f"\nBASELINE single-run completed ({label})")

    print(f"\n  ttft_ms              = {ttft*1e3:.3f}")
    print(f"  pd_total_xfer_ms     = {pd_stats['pd_total_transfer_time_s']*1e3:.3f}")
    print(f"  pd_total_xfer_MB     = {pd_stats['pd_total_transfer_bytes']/1e6:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
