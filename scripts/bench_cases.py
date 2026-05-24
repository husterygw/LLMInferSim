#!/usr/bin/env python3
"""Generate case-driven benchmark suites for real-vs-sim comparison."""
from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL_ALIASES: dict[str, str] = {
    "qwen3_4b": "/data/ygw/models/Qwen3-4B-Instruct-2507",
    "qwen2_5_3b": "/data/ygw/models/Qwen2.5-3B-Instruct",
    "qwen3_32b": "/data/ygw/models/Qwen3-32B",
    "qwen3_30b_a3b": "/data/ygw/models/Qwen3-30B-A3B-Instruct-2507",
}


STAGE_ALIASES = {
    "A": "single_tp1_roofline",
    "B": "tp_comm_sweep",
    "C": "batch_tp1_sweep",
    "D": "tp_batch_sweep",
    "E": "multi_model_regression",
}


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    suite: str
    model: str
    tp: int
    ep: int
    topology_hint: str
    input_len: int
    output_len: int
    concurrency: int
    num_prompts: int
    request_rate: str
    execution_mode: str
    prefix_cache: bool = False
    chunked_prefill: bool = False
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int = 8192
    num_gpu_blocks_override: int | None = None
    num_warmups: int = 1
    gpu_mem_util: float = 0.5
    enable_expert_parallel: bool = False
    tags: tuple[str, ...] = ()
    description: str = ""

    @property
    def group(self) -> str:
        return self.suite

    @property
    def model_alias(self) -> str:
        return self.model

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["group"] = self.suite
        data["model_alias"] = self.model
        data["tags"] = list(self.tags)
        data["workload"] = {
            "type": "fixed",
            "input_len": self.input_len,
            "output_len": self.output_len,
        }
        return data


def _case(
    suite: str,
    shape_name: str,
    model: str,
    tp: int,
    input_len: int,
    output_len: int,
    *,
    ep: int = 1,
    topology_hint: str = "concentrated",
    concurrency: int = 1,
    num_prompts: int | None = None,
    request_rate: str = "inf",
    execution_mode: str = "cudagraph",
    prefix_cache: bool = False,
    chunked_prefill: bool = False,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int = 8192,
    num_gpu_blocks_override: int | None = None,
    num_warmups: int = 1,
    gpu_mem_util: float = 0.5,
    enable_expert_parallel: bool = False,
    tags: tuple[str, ...] = (),
    description: str = "",
) -> BenchCase:
    prompts = concurrency if num_prompts is None else num_prompts
    hint_part = "" if topology_hint == "concentrated" else f"__{topology_hint}"
    ep_part = "" if ep == 1 else f"__ep{ep}"
    c_part = "" if concurrency == 1 else f"__c{concurrency}"
    case_id = f"{suite}__{model}__{shape_name}__tp{tp}{ep_part}{hint_part}{c_part}"
    return BenchCase(
        case_id=case_id,
        suite=suite,
        model=model,
        tp=tp,
        ep=ep,
        topology_hint=topology_hint,
        input_len=input_len,
        output_len=output_len,
        concurrency=concurrency,
        num_prompts=prompts,
        request_rate=request_rate,
        execution_mode=execution_mode,
        prefix_cache=prefix_cache,
        chunked_prefill=chunked_prefill,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_gpu_blocks_override=num_gpu_blocks_override,
        num_warmups=num_warmups,
        gpu_mem_util=gpu_mem_util,
        enable_expert_parallel=enable_expert_parallel,
        tags=tags,
        description=description,
    )


def build_single_tp1_roofline() -> list[BenchCase]:
    suite = "single_tp1_roofline"
    shapes = [
        ("i128_o128", 128, 128),
        ("i512_o128", 512, 128),
        ("i2048_o128", 2048, 128),
        ("i4096_o128", 4096, 128),
        ("i128_o512", 128, 512),
        ("i128_o2048", 128, 2048),
        ("i512_o512", 512, 512),
        ("i2048_o512", 2048, 512),
        ("i4096_o1024", 4096, 1024),
    ]
    return [
        _case(
            suite,
            name,
            "qwen3_4b",
            1,
            i,
            o,
            concurrency=1,
            num_prompts=1,
            request_rate="inf",
            execution_mode="cudagraph",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            tags=("dense", "roofline", "tp1", "single"),
            description=f"Qwen3-4B TP=1 single request roofline baseline, ISL={i}, OSL={o}",
        )
        for name, i, o in shapes
    ]


def build_batch_tp1_sweep() -> list[BenchCase]:
    suite = "batch_tp1_sweep"
    shapes = [
        ("chat_i512_o512", 512, 512),
        ("prefill_i2048_o128", 2048, 128),
        ("decode_i128_o2048", 128, 2048),
        ("long_i4096_o512", 4096, 512),
    ]
    cases: list[BenchCase] = []
    for c in (1, 4, 8, 16, 32):
        for name, i, o in shapes:
            cases.append(
                _case(
                    suite,
                    name,
                    "qwen3_4b",
                    1,
                    i,
                    o,
                    concurrency=c,
                    request_rate="inf",
                    execution_mode="cudagraph",
                    max_model_len=8192,
                    # cap 32768 防 RTX 4090 (24GB) real 启动失败 (No available
                    # memory for cache blocks). 65536+ 让 activation > 1.34GB.
                    max_num_batched_tokens=min(max(8192, c * i), 32768),
                    tags=("dense", "batch", "tp1", f"c{c}"),
                    description=f"Qwen3-4B TP=1 batch sweep c={c}, ISL={i}, OSL={o}",
                )
            )
    return cases


def build_tp_comm_sweep() -> list[BenchCase]:
    suite = "tp_comm_sweep"
    shapes = [
        ("prefill_i2048_o128", 2048, 128),
        ("decode_i128_o2048", 128, 2048),
        ("mix_i4096_o1024", 4096, 1024),
    ]
    tp_configs = [
        (2, "concentrated"),
        (2, "balanced"),
        (4, "concentrated"),
        (4, "balanced"),
        (8, "balanced"),
    ]
    cases: list[BenchCase] = []
    for tp, hint in tp_configs:
        for name, i, o in shapes:
            cases.append(
                _case(
                    suite,
                    name,
                    "qwen3_4b",
                    tp,
                    i,
                    o,
                    topology_hint=hint,
                    concurrency=1,
                    num_prompts=3,
                    request_rate="0.5",
                    execution_mode="cudagraph",
                    max_model_len=8192,
                    max_num_batched_tokens=8192,
                    tags=("dense", "communication", f"tp{tp}", hint),
                    description=f"Qwen3-4B TP={tp} {hint} communication sweep, ISL={i}, OSL={o}",
                )
            )
    return cases


def build_tp_batch_sweep() -> list[BenchCase]:
    suite = "tp_batch_sweep"
    shapes = [
        ("chat_i512_o512", 512, 512),
        ("rag_i4096_o128", 4096, 128),
    ]
    cases: list[BenchCase] = []
    for tp in (2, 4, 8):
        hint = "balanced" if tp == 8 else "concentrated"
        for c in (8, 16, 32):
            for name, i, o in shapes:
                cases.append(
                    _case(
                        suite,
                        name,
                        "qwen3_4b",
                        tp,
                        i,
                        o,
                        topology_hint=hint,
                        concurrency=c,
                        request_rate="inf",
                        execution_mode="cudagraph",
                        max_model_len=8192,
                        max_num_batched_tokens=max(8192, c * i),
                        tags=("dense", "production", f"tp{tp}", f"c{c}"),
                        description=f"Qwen3-4B TP={tp} batch sweep c={c}, ISL={i}, OSL={o}",
                    )
                )
    return cases


def build_long_context_sweep() -> list[BenchCase]:
    suite = "long_context_sweep"
    shapes = [
        ("i4096_o128", 4096, 128),
        ("i8192_o128", 8192, 128),
        ("i8192_o512", 8192, 512),
        ("i16384_o128", 16384, 128),
    ]
    cases: list[BenchCase] = []
    for c in (1, 4):
        for name, i, o in shapes:
            cases.append(
                _case(
                    suite,
                    name,
                    "qwen3_4b",
                    1,
                    i,
                    o,
                    concurrency=c,
                    request_rate="inf",
                    execution_mode="cudagraph",
                    max_model_len=32768,
                    max_num_batched_tokens=max(32768, c * i),
                    tags=("dense", "long_context", f"c{c}"),
                    description=f"Qwen3-4B long context sweep c={c}, ISL={i}, OSL={o}",
                )
            )
    return cases


def build_moe_tp_sweep() -> list[BenchCase]:
    suite = "moe_tp_sweep"
    shapes = [
        ("i128_o128", 128, 128),
        ("i1024_o128", 1024, 128),
        ("i4096_o128", 4096, 128),
        ("i128_o2048", 128, 2048),
        ("i4096_o1024", 4096, 1024),
    ]
    return [
        _case(
            suite,
            name,
            "qwen3_30b_a3b",
            4,
            i,
            o,
            ep=1,
            topology_hint="concentrated",
            concurrency=1,
            num_prompts=3,
            request_rate="0.5",
            execution_mode="cudagraph",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            gpu_mem_util=0.85,
            enable_expert_parallel=False,
            tags=("moe", "tp_only", "tp4"),
            description=f"Qwen3-30B-A3B TP=4 EP=1 MoE TP-only sweep, ISL={i}, OSL={o}",
        )
        for name, i, o in shapes
    ]


def build_moe_ep_sweep() -> list[BenchCase]:
    suite = "moe_ep_sweep"
    shapes = [
        ("i128_o128", 128, 128),
        ("i2048_o128", 2048, 128),
        ("i128_o2048", 128, 2048),
    ]
    return [
        _case(
            suite,
            name,
            "qwen3_30b_a3b",
            4,
            i,
            o,
            ep=4,
            topology_hint="concentrated",
            concurrency=1,
            num_prompts=3,
            request_rate="0.5",
            execution_mode="cudagraph",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            gpu_mem_util=0.85,
            enable_expert_parallel=True,
            tags=("moe", "ep", "tp4", "ep4"),
            description=f"Qwen3-30B-A3B TP=4 EP=4 MoE EP sweep, ISL={i}, OSL={o}",
        )
        for name, i, o in shapes
    ]


def build_multi_model_regression() -> list[BenchCase]:
    suite = "multi_model_regression"
    configs = [
        ("qwen2_5_3b", 1, 0.5),
        ("qwen3_4b", 1, 0.5),
        ("qwen3_32b", 4, 0.85),
    ]
    return [
        _case(
            suite,
            "chat_i512_o512",
            model,
            tp,
            512,
            512,
            concurrency=1,
            num_prompts=3,
            request_rate="0.5",
            execution_mode="cudagraph",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            gpu_mem_util=gpu_mem,
            tags=("regression", "dense", model),
            description=f"{model} regression sample, TP={tp}, ISL=512, OSL=512",
        )
        for model, tp, gpu_mem in configs
    ]


SUITE_BUILDERS = {
    "single_tp1_roofline": build_single_tp1_roofline,
    "batch_tp1_sweep": build_batch_tp1_sweep,
    "tp_comm_sweep": build_tp_comm_sweep,
    "tp_batch_sweep": build_tp_batch_sweep,
    "long_context_sweep": build_long_context_sweep,
    "moe_tp_sweep": build_moe_tp_sweep,
    "moe_ep_sweep": build_moe_ep_sweep,
    "multi_model_regression": build_multi_model_regression,
}


def canonical_suite(name: str) -> str:
    return STAGE_ALIASES.get(name, name)


def build_cases(suite: str | None) -> list[BenchCase]:
    if suite is None or suite == "all":
        cases: list[BenchCase] = []
        for builder in SUITE_BUILDERS.values():
            cases.extend(builder())
        return cases
    suite = canonical_suite(suite)
    if suite not in SUITE_BUILDERS:
        known = ", ".join(SUITE_BUILDERS)
        raise SystemExit(f"unknown suite '{suite}'. Known suites: {known}")
    return SUITE_BUILDERS[suite]()


def save_cases_jsonl(path: str | Path, cases: list[BenchCase]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_json(), ensure_ascii=False) + "\n")


def print_summary(cases: list[BenchCase]) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.suite] = counts.get(case.suite, 0) + 1
    print(f"Total cases: {len(cases)}")
    print(f"{'suite':<28} count")
    print("-" * 40)
    for suite, count in sorted(counts.items()):
        print(f"{suite:<28} {count:>5}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", help="suite name, all, or legacy A/B/C/D/E")
    parser.add_argument("--out", default="/tmp/llm_infer_sim_bench/cases.jsonl")
    parser.add_argument("--filter-case", help="fnmatch pattern applied to case_id")
    parser.add_argument("--list", action="store_true", help="list suites and exit")
    args = parser.parse_args()

    if args.list:
        print("Suites:")
        for suite in SUITE_BUILDERS:
            print(f"  {suite}")
        print("\nLegacy aliases:")
        for old, new in STAGE_ALIASES.items():
            print(f"  {old} -> {new}")
        return 0

    cases = build_cases(args.suite)
    if args.filter_case:
        cases = [case for case in cases if fnmatch.fnmatch(case.case_id, args.filter_case)]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("duplicate case_id generated")

    save_cases_jsonl(args.out, cases)
    print(f"Wrote {len(cases)} cases -> {args.out}")
    print_summary(cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
