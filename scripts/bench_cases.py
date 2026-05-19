"""Case-driven benchmark definitions for sim vs real comparison.

ISL/OSL 组合参考 docs/CALIBRATION_METHODOLOGY.md 用户提供的
`compare_sim_real_ttft_tpot.py` 校准用例设计:
  - prefill 系列: ISL 变 / OSL 固定 128 → TPOT 接近恒定, TTFT 差异纯归因 prefill
  - decode 系列:  ISL 固定 128 / OSL 变 → TTFT 接近恒定, TPOT 差异纯归因 decode
  - mix 1 个:    prefill + decode 共变, 端到端 sanity

多 TP / MoE / 并发 group 在这套 ISL/OSL 之上扩维度.

跑法:
    python scripts/bench_cases.py --out /tmp/llm_infer_sim_bench/cases.jsonl

详 docs/CALIBRATION_METHODOLOGY.md §5(stage 名 ↔ group 名 映射).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Model aliases — 把符号名映到 /data/ygw/models 实际路径
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "qwen3_4b":      "/data/ygw/models/Qwen3-4B-Instruct-2507",
    "qwen2_5_3b":    "/data/ygw/models/Qwen2.5-3B-Instruct",
    "qwen3_32b":     "/data/ygw/models/Qwen3-32B",
    "qwen3_30b_a3b": "/data/ygw/models/Qwen3-30B-A3B-Instruct-2507",
}


# ---------------------------------------------------------------------------
# Workload patterns(混合负载用,目前 fixed 占主)
# ---------------------------------------------------------------------------

CHAT_MIX = [
    {"ratio": 0.50, "input_len": 512,  "output_len": 512},
    {"ratio": 0.30, "input_len": 1024, "output_len": 512},
    {"ratio": 0.15, "input_len": 2048, "output_len": 1024},
    {"ratio": 0.05, "input_len": 4096, "output_len": 1024},
]
RAG_MIX = [
    {"ratio": 0.20, "input_len": 1024,  "output_len": 512},
    {"ratio": 0.50, "input_len": 4096,  "output_len": 256},
    {"ratio": 0.25, "input_len": 8192,  "output_len": 256},
    {"ratio": 0.05, "input_len": 16384, "output_len": 128},
]


# ---------------------------------------------------------------------------
# ISL/OSL 校准用 shape 集
# ---------------------------------------------------------------------------

# 单请求校准 shape 集(对齐参考脚本 build_single_request_calibration_cases):
#   - "baseline":      i128_o128, 用来定 sim/real 的 fixed overhead 差异
#   - "prefill_*":     OSL 固定 128, ISL 变化, 隔离 prefill 路径
#   - "decode_*":      ISL 固定 128, OSL 变化, 隔离 decode 路径
#   - "mix":           prefill+decode 共变, 端到端 sanity
SINGLE_REQUEST_SHAPES: list[tuple[str, int, int, str]] = [
    ("baseline_i128_o128",   128,   128, "baseline_fixed_overhead"),
    ("prefill_i512_o128",     512,  128, "small_prefill"),
    ("prefill_i1024_o128",   1024,  128, "medium_prefill"),
    ("prefill_i2048_o128",   2048,  128, "prefill_scaling"),
    ("prefill_i4096_o128",   4096,  128, "long_prefill"),
    ("prefill_i8192_o128",   8192,  128, "long_context_prefill"),
    ("decode_i128_o512",      128,  512, "decode_scaling_small"),
    ("decode_i128_o1024",     128, 1024, "decode_scaling_med"),
    ("decode_i128_o2048",     128, 2048, "long_decode"),
    ("mix_i4096_o1024",      4096, 1024, "prefill_decode_mix"),
]

# multi_tp 选 3 个代表 shape(prefill_only / decode_only / mix), 控制 case 数
MULTI_TP_SHAPES = [
    ("prefill_i2048_o128",  2048, 128, "prefill_scaling"),
    ("decode_i128_o2048",    128, 2048, "long_decode"),
    ("mix_i4096_o1024",     4096, 1024, "prefill_decode_mix"),
]

# concurrent groups 用 reference homogeneous_concurrency 的 4 workload
CONCURRENT_WORKLOADS = [
    ("chat",                  512,  512),
    ("rag_prefill_heavy",    4096,  128),
    ("decode_heavy",          128, 2048),
    ("long_context",         8192,  512),
]


# ---------------------------------------------------------------------------
# Case dataclass
# ---------------------------------------------------------------------------

@dataclass
class Case:
    case_id: str
    group: str
    tags: list[str]
    model_alias: str
    tp: int = 1
    ep: int = 1
    topology_hint: str = "concentrated"     # "concentrated" | "balanced"
    execution_mode: str = "eager"            # "eager" | "cudagraph"
    # 语义峰值并发度 (informational, 真正决定 bench 行为的是 num_prompts + request_rate)
    concurrency: int = 1
    # bench --num-prompts: 总共发多少 prompt. concurrent 模式 = concurrency. serial 模式 > 1 用来平均
    num_prompts: int = 1
    # bench --request-rate: inf = 全部 t=0 到达 → 峰值 = num_prompts; 0.5 = 2s 间隔 → serial
    request_rate: float = float("inf")
    # bench --num-warmups: 跑前 N 个 prompt 不计入测量 (排除冷启动). 默认 1 个 prompt 足以激活 lazy init
    num_warmups: int = 1
    workload: dict = field(default_factory=dict)   # {"type":"fixed","input_len","output_len"}
                                                    # 或 {"type":"mixed","pattern":[...]}
    description: str = ""
    gpu_mem_util: float = 0.5
    enable_expert_parallel: bool = False     # only relevant for MoE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_workload(input_len: int, output_len: int) -> dict:
    return {"type": "fixed", "input_len": input_len, "output_len": output_len}


def _mixed_workload(name: str, pattern: list[dict]) -> dict:
    return {"type": "mixed", "name": name, "pattern": pattern}


# 单请求 serial: 3 个 prompt 间隔 2s, 取平均, 跟 reference 1 prompt 比更稳
SINGLE_REQUEST_NUM_PROMPTS = 3
SINGLE_REQUEST_RATE = 0.5
SINGLE_REQUEST_CONCURRENCY = 1


# ---------------------------------------------------------------------------
# Group builders
# ---------------------------------------------------------------------------

def build_single_request_tp1() -> list[Case]:
    """Stage A: dense Qwen3-4B TP=1, 10 个校准 shape.

    校准目标:基础 cost model(纯 GPU 计算, 无通信, 无并发).
    prefill 系列 (OSL=128 固定) 用来隔离 prefill 路径回归.
    """
    cases = []
    for shape_name, i, o, workload_tag in SINGLE_REQUEST_SHAPES:
        cases.append(Case(
            case_id=f"single_request_tp1__{shape_name}__tp1",
            group="single_request_tp1",
            tags=["dense", "calibration", "tp1", workload_tag],
            model_alias="qwen3_4b",
            tp=1, ep=1,
            topology_hint="concentrated",
            concurrency=SINGLE_REQUEST_CONCURRENCY,
            num_prompts=SINGLE_REQUEST_NUM_PROMPTS,
            request_rate=SINGLE_REQUEST_RATE,
            workload=_fixed_workload(i, o),
            description=f"dense Qwen3-4B TP=1 single_request, i={i} o={o} ({workload_tag})",
        ))
    return cases


def build_single_request_multi_tp() -> list[Case]:
    """Stage B: dense Qwen3-4B TP>1 串行, 多拓扑.

    校准目标:通信参数(comm_step_latency / protocol_eff / framework_oh).
    选 3 代表 shape (prefill_only / decode_only / mix) × 5 TP 配置.
    """
    cases = []
    tp_configs = [
        (2, "concentrated", "tp2_same_numa"),
        (2, "balanced",     "tp2_cross_numa"),
        (4, "concentrated", "tp4_same_numa"),
        (4, "balanced",     "tp4_cross_numa"),
        (8, "balanced",     "tp8"),
    ]
    for shape_name, i, o, workload_tag in MULTI_TP_SHAPES:
        for tp, hint, tp_label in tp_configs:
            cases.append(Case(
                case_id=f"single_request_multi_tp__{shape_name}__{tp_label}",
                group="single_request_multi_tp",
                tags=["dense", "communication", f"tp{tp}", hint, workload_tag],
                model_alias="qwen3_4b",
                tp=tp, ep=1,
                topology_hint=hint,
                concurrency=SINGLE_REQUEST_CONCURRENCY,
                num_prompts=SINGLE_REQUEST_NUM_PROMPTS,
                request_rate=SINGLE_REQUEST_RATE,
                workload=_fixed_workload(i, o),
                description=f"dense Qwen3-4B TP={tp} {hint} single_request, i={i} o={o}",
            ))
    return cases


def build_concurrent_tp1() -> list[Case]:
    """Stage C: dense Qwen3-4B TP=1 多请求并发. 测 chunked prefill / scheduler.

    Reference homogeneous_concurrency: 4 workload × {4,16,32} concurrency.
    (c=1 单请求 baseline 已经被 single_request_tp1 覆盖, 这里去掉.)
    """
    cases = []
    concurrencies = [4, 16, 32]
    for workload_name, i, o in CONCURRENT_WORKLOADS:
        for c in concurrencies:
            cases.append(Case(
                case_id=f"concurrent_tp1__{workload_name}__c{c}__tp1",
                group="concurrent_tp1",
                tags=["dense", "concurrency", "tp1", workload_name],
                model_alias="qwen3_4b",
                tp=1, ep=1,
                topology_hint="concentrated",
                concurrency=c,
                num_prompts=c,
                request_rate=float("inf"),
                workload=_fixed_workload(i, o),
                description=f"dense Qwen3-4B TP=1 c={c} {workload_name} (i={i} o={o})",
            ))
    return cases


def build_concurrent_multi_tp() -> list[Case]:
    """Stage D: dense Qwen3-4B TP>1 多请求并发. production-grade.

    2 workload (chat 平衡 + rag_prefill_heavy 重 prefill) × {16, 32} × 3 TP.
    """
    cases = []
    workloads = [("chat", 512, 512), ("rag_prefill_heavy", 4096, 128)]
    concurrencies = [16, 32]
    tp_configs = [
        (2, "concentrated", "tp2"),
        (4, "concentrated", "tp4"),
        (8, "balanced",     "tp8"),
    ]
    for workload_name, i, o in workloads:
        for c in concurrencies:
            for tp, hint, tp_label in tp_configs:
                cases.append(Case(
                    case_id=f"concurrent_multi_tp__{workload_name}__c{c}__{tp_label}",
                    group="concurrent_multi_tp",
                    tags=["dense", "production", f"tp{tp}", workload_name],
                    model_alias="qwen3_4b",
                    tp=tp, ep=1,
                    topology_hint=hint,
                    concurrency=c,
                    num_prompts=c,
                    request_rate=float("inf"),
                    workload=_fixed_workload(i, o),
                    description=f"dense Qwen3-4B TP={tp} c={c} {workload_name}",
                ))
    return cases


def build_moe_single_request_tp_only() -> list[Case]:
    """Stage M-A (TP-only): MoE Qwen3-30B-A3B TP>1 EP=1 串行.

    路径:`routed_expert_allreduce` (AllReduce 出口, 各 rank 持所有 expert).
    用单请求 5 代表 shape (baseline + 2 prefill + 1 decode + 1 mix).
    """
    cases = []
    # 5 代表 shape: baseline / small_prefill / long_prefill / decode / mix
    moe_shapes = [
        ("baseline_i128_o128",   128,  128, "baseline_fixed_overhead"),
        ("prefill_i1024_o128",  1024,  128, "medium_prefill"),
        ("prefill_i4096_o128",  4096,  128, "long_prefill"),
        ("decode_i128_o2048",    128, 2048, "long_decode"),
        ("mix_i4096_o1024",     4096, 1024, "prefill_decode_mix"),
    ]
    for shape_name, i, o, workload_tag in moe_shapes:
        cases.append(Case(
            case_id=f"moe_single_request_tp_only__{shape_name}__tp4",
            group="moe_single_request_tp_only",
            tags=["moe", "tp_only", "tp4", workload_tag],
            model_alias="qwen3_30b_a3b",
            tp=4, ep=1,
            topology_hint="concentrated",
            concurrency=SINGLE_REQUEST_CONCURRENCY,
            num_prompts=SINGLE_REQUEST_NUM_PROMPTS,
            request_rate=SINGLE_REQUEST_RATE,
            workload=_fixed_workload(i, o),
            description=f"MoE Qwen3-30B-A3B TP=4 EP=1 (AllReduce path), i={i} o={o}",
            gpu_mem_util=0.85,
            enable_expert_parallel=False,
        ))
    return cases


def build_moe_single_request_ep() -> list[Case]:
    """Stage M-A (EP): MoE EP>1 串行, 走 AllToAll dispatch/combine."""
    cases = []
    ep_shapes = [
        ("baseline_i128_o128",   128,  128, "baseline_fixed_overhead"),
        ("prefill_i2048_o128",  2048,  128, "prefill_scaling"),
        ("decode_i128_o2048",    128, 2048, "long_decode"),
    ]
    for shape_name, i, o, workload_tag in ep_shapes:
        cases.append(Case(
            case_id=f"moe_single_request_ep__{shape_name}__tp4_ep4",
            group="moe_single_request_ep",
            tags=["moe", "ep", "tp4_ep4", workload_tag],
            model_alias="qwen3_30b_a3b",
            tp=4, ep=4,
            topology_hint="concentrated",
            concurrency=SINGLE_REQUEST_CONCURRENCY,
            num_prompts=SINGLE_REQUEST_NUM_PROMPTS,
            request_rate=SINGLE_REQUEST_RATE,
            workload=_fixed_workload(i, o),
            description=f"MoE Qwen3-30B-A3B TP=4 EP=4 (AllToAll path), i={i} o={o}",
            gpu_mem_util=0.85,
            enable_expert_parallel=True,
        ))
    return cases


def build_moe_concurrent_tp_only() -> list[Case]:
    """Stage M-B (TP-only): MoE 多请求并发. chat workload × {4, 16}."""
    cases = []
    for c in (4, 16):
        cases.append(Case(
            case_id=f"moe_concurrent_tp_only__chat__c{c}__tp4",
            group="moe_concurrent_tp_only",
            tags=["moe", "tp_only", "concurrent", "tp4"],
            model_alias="qwen3_30b_a3b",
            tp=4, ep=1,
            topology_hint="concentrated",
            concurrency=c, num_prompts=c, request_rate=float("inf"),
            workload=_fixed_workload(512, 512),
            description=f"MoE Qwen3-30B-A3B TP=4 EP=1 c={c} concurrent chat",
            gpu_mem_util=0.85,
            enable_expert_parallel=False,
        ))
    return cases


def build_moe_concurrent_ep() -> list[Case]:
    """Stage M-B (EP): MoE EP 路径 + 并发."""
    cases = []
    for c in (4, 16):
        cases.append(Case(
            case_id=f"moe_concurrent_ep__chat__c{c}__tp4_ep4",
            group="moe_concurrent_ep",
            tags=["moe", "ep", "concurrent", "tp4_ep4"],
            model_alias="qwen3_30b_a3b",
            tp=4, ep=4,
            topology_hint="concentrated",
            concurrency=c, num_prompts=c, request_rate=float("inf"),
            workload=_fixed_workload(512, 512),
            description=f"MoE Qwen3-30B-A3B TP=4 EP=4 c={c} concurrent chat",
            gpu_mem_util=0.85,
            enable_expert_parallel=True,
        ))
    return cases


def build_multi_model_regression() -> list[Case]:
    """Stage E: 跨模型回归. 用 chat 代表 shape (512/512) 抽样."""
    cases = []
    # Qwen2.5-3B TP=1 baseline
    cases.append(Case(
        case_id="multi_model_regression__qwen2_5_3b__chat__tp1",
        group="multi_model_regression",
        tags=["dense", "regression", "qwen2_5"],
        model_alias="qwen2_5_3b",
        tp=1, ep=1,
        concurrency=SINGLE_REQUEST_CONCURRENCY,
        num_prompts=SINGLE_REQUEST_NUM_PROMPTS,
        request_rate=SINGLE_REQUEST_RATE,
        workload=_fixed_workload(512, 512),
        description="Qwen2.5-3B TP=1 chat (512/512) regression sample",
    ))
    # Qwen3-32B TP=4 (≥4 才 fit 24GB 卡)
    cases.append(Case(
        case_id="multi_model_regression__qwen3_32b__chat__tp4",
        group="multi_model_regression",
        tags=["dense", "regression", "qwen3_32b"],
        model_alias="qwen3_32b",
        tp=4, ep=1,
        topology_hint="concentrated",
        concurrency=SINGLE_REQUEST_CONCURRENCY,
        num_prompts=SINGLE_REQUEST_NUM_PROMPTS,
        request_rate=SINGLE_REQUEST_RATE,
        workload=_fixed_workload(512, 512),
        description="Qwen3-32B TP=4 chat (512/512) regression sample",
        gpu_mem_util=0.85,
    ))
    return cases


def build_mixed_workload_validation() -> list[Case]:
    """泛化验证: chat_mix / rag_mix 真实 production 混合 ISL/OSL.

    NOTE: vllm bench serve 的 random dataset 默认所有 prompt 同长度.
    mixed workload 需要 bench client 改造 (sharegpt-like 或 case-aware client).
    现在 emit case 但 run_bench_group 检测 type=mixed 会 SKIP, TODO 项.
    """
    cases = []
    cases.append(Case(
        case_id="mixed_workload_validation__chat_mix__c32__tp1",
        group="mixed_workload_validation",
        tags=["dense", "mixed", "validation", "tp1"],
        model_alias="qwen3_4b",
        tp=1, ep=1,
        topology_hint="concentrated",
        concurrency=32, num_prompts=32, request_rate=float("inf"),
        workload=_mixed_workload("chat_mix", CHAT_MIX),
        description="chat_mix (ratio-based ISL/OSL) c=32, TP=1 — TODO: bench client 待支持",
    ))
    cases.append(Case(
        case_id="mixed_workload_validation__rag_mix__c32__tp1",
        group="mixed_workload_validation",
        tags=["dense", "mixed", "validation", "tp1"],
        model_alias="qwen3_4b",
        tp=1, ep=1,
        topology_hint="concentrated",
        concurrency=32, num_prompts=32, request_rate=float("inf"),
        workload=_mixed_workload("rag_mix", RAG_MIX),
        description="rag_mix (ratio-based ISL/OSL) c=32, TP=1 — TODO: bench client 待支持",
    ))
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_GROUP_BUILDERS = [
    build_single_request_tp1,
    build_single_request_multi_tp,
    build_concurrent_tp1,
    build_concurrent_multi_tp,
    build_moe_single_request_tp_only,
    build_moe_single_request_ep,
    build_moe_concurrent_tp_only,
    build_moe_concurrent_ep,
    build_multi_model_regression,
    build_mixed_workload_validation,
]


def build_all_cases() -> list[Case]:
    cases = []
    for fn in ALL_GROUP_BUILDERS:
        cases.extend(fn())
    return cases


def save_cases_jsonl(path: str | Path, cases: list[Case]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for c in cases:
            d = asdict(c)
            # math.inf 在 JSON 里要存成 string, 读侧统一转回 float
            if d["request_rate"] == float("inf"):
                d["request_rate"] = "inf"
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def print_summary(cases: list[Case]) -> None:
    from collections import Counter
    groups = Counter(c.group for c in cases)
    print(f"Total cases: {len(cases)}")
    print()
    print(f"{'group':<35} count")
    print("-" * 50)
    for g, n in sorted(groups.items()):
        print(f"{g:<35} {n:>5}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/llm_infer_sim_bench/cases.jsonl")
    args = p.parse_args()
    cases = build_all_cases()
    save_cases_jsonl(args.out, cases)
    print(f"Wrote {len(cases)} cases → {args.out}")
    print()
    print_summary(cases)


if __name__ == "__main__":
    main()
