#!/usr/bin/env python3
"""Profile vLLM worker runtime buckets for one short request.

This script intentionally monkeypatches vLLM in-process instead of editing the
installed package. It measures the non-operator buckets that explain short TTFT:
input preparation, attention metadata, model forward, CUDA graph replay, sampler,
and total offline generate wall time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class BucketStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        self.samples_ms.append(ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "avg_ms": self.avg_ms,
            "max_ms": self.max_ms,
            "samples_ms": self.samples_ms,
        }


class RuntimeProfiler:
    def __init__(self, event_path: Path | None = None) -> None:
        self.buckets: dict[str, BucketStats] = defaultdict(BucketStats)
        self.events: list[dict[str, Any]] = []
        self.event_path = event_path

    def record(self, name: str, ms: float, **metadata: Any) -> None:
        self.buckets[name].add(ms)
        event = {"name": name, "ms": ms}
        event.update(metadata)
        self.events.append(event)
        if self.event_path is not None:
            payload = dict(event)
            payload["pid"] = os.getpid()
            with self.event_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def wrap(
        self,
        owner: Any,
        attr: str,
        name: str,
        *,
        sync: bool = True,
        metadata_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        original = getattr(owner, attr)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if sync:
                _cuda_sync()
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if sync:
                    _cuda_sync()
                ms = (time.perf_counter() - t0) * 1e3
                md = metadata_fn(*args, **kwargs) if metadata_fn else {}
                self.record(name, ms, **md)

        setattr(owner, attr, wrapped)


def _safe_name(obj: Any) -> str:
    name = getattr(obj, "name", None)
    if name is not None:
        return str(name)
    return str(obj)


def install_patches(prof: RuntimeProfiler) -> None:
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    def exec_md(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        return {
            "num_reqs": len(scheduled),
            "num_tokens": total_tokens,
            "max_query_len": max(scheduled.values()) if scheduled else None,
        }

    def prep_md(self: Any, scheduler_output: Any, batch_desc: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        return {
            "num_reqs": len(scheduled),
            "num_tokens": getattr(scheduler_output, "total_num_scheduled_tokens", None),
            "cg_mode": _safe_name(getattr(batch_desc, "cg_mode", None)),
            "batch_num_tokens": getattr(batch_desc, "num_tokens", None),
            "batch_num_reqs": getattr(batch_desc, "num_reqs", None),
        }

    def input_batch_md(self: Any, input_batch: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "num_reqs": getattr(input_batch, "num_reqs", None),
            "num_tokens": getattr(input_batch, "num_tokens", None),
            "num_tokens_after_padding": getattr(input_batch, "num_tokens_after_padding", None),
        }

    def state_attn_md(self: Any, input_batch: Any, cudagraph_mode: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "num_reqs": getattr(input_batch, "num_reqs", None),
            "num_tokens": getattr(input_batch, "num_tokens", None),
            "num_tokens_after_padding": getattr(input_batch, "num_tokens_after_padding", None),
            "cg_mode": _safe_name(cudagraph_mode),
        }

    prof.wrap(GPUModelRunner, "execute_model", "execute_model_total", metadata_fn=exec_md)
    if hasattr(GPUModelRunner, "prepare_inputs"):
        prof.wrap(GPUModelRunner, "prepare_inputs", "prepare_inputs", metadata_fn=prep_md)
    if hasattr(GPUModelRunner, "_prepare_inputs"):
        prof.wrap(GPUModelRunner, "_prepare_inputs", "prepare_inputs", sync=True)
    if hasattr(GPUModelRunner, "prepare_attn"):
        prof.wrap(GPUModelRunner, "prepare_attn", "prepare_attn_buffers", metadata_fn=input_batch_md)
    if hasattr(GPUModelRunner, "_build_attention_metadata"):
        prof.wrap(GPUModelRunner, "_build_attention_metadata", "build_attn_metadata", sync=True)
    prof.wrap(DefaultModelState, "prepare_attn", "build_attn_metadata", metadata_fn=state_attn_md)
    prof.wrap(DefaultModelState, "prepare_inputs", "model_state_prepare_inputs", metadata_fn=input_batch_md)
    if hasattr(GPUModelRunner, "sample"):
        prof.wrap(GPUModelRunner, "sample", "sample_compute_logits_and_sampler", metadata_fn=input_batch_md)
    if hasattr(GPUModelRunner, "_model_forward"):
        prof.wrap(GPUModelRunner, "_model_forward", "model_forward", sync=True)
    if hasattr(GPUModelRunner, "_preprocess"):
        prof.wrap(GPUModelRunner, "_preprocess", "preprocess", sync=True)
    if hasattr(GPUModelRunner, "_sample"):
        prof.wrap(GPUModelRunner, "_sample", "sample_kernel", sync=True)
    prof.wrap(GPUModelRunner, "sample_tokens", "sample_tokens_total")

    original_call = CUDAGraphWrapper.__call__

    def cg_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        mode = "NO_CONTEXT"
        descriptor = None
        kind = "none"
        if is_forward_context_available():
            ctx = get_forward_context()
            mode_obj = getattr(ctx, "cudagraph_runtime_mode", None)
            mode = _safe_name(mode_obj)
            descriptor = getattr(ctx, "batch_descriptor", None)
            if mode_obj == CUDAGraphMode.NONE or mode_obj != getattr(self, "runtime_mode", None):
                kind = "bypass"
            elif descriptor is not None:
                entry = self.concrete_cudagraph_entries.get(descriptor)
                kind = "capture" if entry is None or entry.cudagraph is None else "replay"
        t0 = time.perf_counter()
        try:
            return original_call(self, *args, **kwargs)
        finally:
            # Do not synchronize here: this wrapper is sometimes invoked while
            # an outer CUDA graph capture is active. Synchronizing during
            # capture invalidates the graph. Treat this bucket as a lightweight
            # call-count / Python-wall probe; use execute_model for synced time.
            prof.record(
                "cudagraph_wrapper_call",
                (time.perf_counter() - t0) * 1e3,
                cg_mode=mode,
                runtime_mode=_safe_name(getattr(self, "runtime_mode", None)),
                kind=kind,
                descriptor=str(descriptor),
            )

    CUDAGraphWrapper.__call__ = cg_call


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    event_path = args.events_jsonl
    if event_path is not None:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text("", encoding="utf-8")
    prof = RuntimeProfiler(event_path)
    install_patches(prof)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
    )

    prompt = list(range(args.input_len))
    sampling = SamplingParams(
        max_tokens=args.output_len,
        ignore_eos=True,
        temperature=0.0,
    )

    # Warm one request so compilation/cache state is closer to benchmark steady state.
    if args.warmup:
        llm.generate([prompt], sampling, use_tqdm=False)
        _cuda_sync()
        prof.buckets.clear()
        prof.events.clear()
        if event_path is not None:
            event_path.write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    _cuda_sync()
    wall_ms = (time.perf_counter() - t0) * 1e3

    token_count = 0
    if outputs:
        token_count = len(outputs[0].outputs[0].token_ids)

    all_events = list(prof.events)
    if event_path is not None and event_path.exists():
        with event_path.open(encoding="utf-8") as f:
            all_events = [json.loads(line) for line in f if line.strip()]

    buckets: dict[str, BucketStats] = defaultdict(BucketStats)
    for event in all_events:
        buckets[event["name"]].add(float(event["ms"]))

    result = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "tp": args.tp,
        "enforce_eager": args.enforce_eager,
        "offline_generate_wall_ms": wall_ms,
        "generated_tokens": token_count,
        "buckets": {k: v.to_dict() for k, v in sorted(buckets.items())},
        "event_count": len(all_events),
        "events": all_events if args.include_events else [],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/ygw/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--events-jsonl", type=Path, default=Path("/tmp/vllm_runtime_events.jsonl"))
    parser.add_argument("--include-events", action="store_true")
    args = parser.parse_args()

    result = run_once(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
