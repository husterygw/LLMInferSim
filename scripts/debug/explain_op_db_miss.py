#!/usr/bin/env python3
"""Explain why runtime ops miss the measured operator DB, field by field.

For each runtime op in a model step, compute its query-side OperatorSignature,
look it up in the measured DB partition; on miss, find the best-overlapping
stored record (same op_kind) and print exactly which signature fields differ
(op_subtype / dtype / shape.* / parallel.* / runtime.*). Then summarize, per
op_kind, which fields are the recurring culprits.

This turns "DB miss" from a black box into a classified diagnosis BEFORE the
interface refactor — so Phase 3/4 fixes a known target instead of guessing.

Usage:
    PYTHONPATH=. python scripts/explain_op_db_miss.py
    PYTHONPATH=. python scripts/explain_op_db_miss.py --hw RTX_4090 --fw 0.19.1
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from llm_infer_sim.core.cost.engine import build_qwen_roofline_engine
from llm_infer_sim.core.step.step_shape import StepShape
from llm_infer_sim.core.operator_db.importers.collector_v2 import raw_record_to_signature
from llm_infer_sim.core.operator_db.version_check import (
    OperatorDBVersionMismatch, verify_db_framework_version,
)
from llm_infer_sim.core.operator_schema import operator_to_signature
from llm_infer_sim.core.deployment.profile import DeploymentProfile
from llm_infer_sim.core.runtime.profile import RuntimeProfile
from llm_infer_sim.core.hardware import get_hardware_config as get_hardware_profile
from llm_infer_sim.core.models.config import ModelConfig
from llm_infer_sim.core.workload.workload import (
    GlobalStepWorkload, RequestWorkload, StepPhase,
)

DB_KINDS = ("gemm", "attention", "moe", "collective")


# --------------------------------------------------------------------------- #
# model configs
# --------------------------------------------------------------------------- #
def qwen3_4b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-4B", hidden_dim=2560, num_heads=32, num_kv_heads=8,
        head_dim=128, ffn_dim=9728, num_layers=36, vocab_size=151936,
    )


def qwen3_30b_a3b() -> ModelConfig:
    return ModelConfig(
        name="Qwen3-30B-A3B", hidden_dim=2048, num_heads=32, num_kv_heads=4,
        head_dim=128, ffn_dim=6144, num_layers=48, vocab_size=151936,
        is_moe=True, num_experts=128, num_activated_experts=8, expert_dim=768,
        moe_layer_freq=1, first_moe_layer=0,
    )


def deploy(tp=1, ep=1, moe_tp=1, moe_ep=1):
    deployment = DeploymentProfile.flat(tp=tp, ep=ep, moe_tp=moe_tp, moe_ep=moe_ep)
    runtime = RuntimeProfile.flat(
        execution_mode="cudagraph", backend="vllm", backend_version="0.19.0",
    )
    return deployment, runtime


def prefill_wl(isl: int) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=0, phase=StepPhase.PREFILL,
        requests=[RequestWorkload(request_id="p0", phase=StepPhase.PREFILL,
                                  num_tokens=isl, context_len=0)],
        num_prefill_tokens=isl, total_scheduled_tokens=isl, num_prefill_requests=1,
    )


def decode_wl(n: int, ctx: int = 512) -> GlobalStepWorkload:
    return GlobalStepWorkload(
        step_id=1, phase=StepPhase.DECODE,
        requests=[RequestWorkload(request_id=f"d{i}", phase=StepPhase.DECODE,
                                  num_tokens=1, context_len=ctx, generated_tokens=8)
                  for i in range(n)],
        num_decode_tokens=n, total_scheduled_tokens=n, num_decode_requests=n,
    )


def mixed_wl(isl: int, n_decode: int, ctx: int = 512) -> GlobalStepWorkload:
    reqs = [RequestWorkload(request_id="p0", phase=StepPhase.PREFILL,
                            num_tokens=isl, context_len=0)]
    reqs += [RequestWorkload(request_id=f"d{i}", phase=StepPhase.DECODE,
                             num_tokens=1, context_len=ctx, generated_tokens=8)
             for i in range(n_decode)]
    return GlobalStepWorkload(
        step_id=2, phase=StepPhase.MIXED, requests=reqs,
        num_prefill_tokens=isl, num_decode_tokens=n_decode,
        total_scheduled_tokens=isl + n_decode,
        num_prefill_requests=1, num_decode_requests=n_decode,
    )


# --------------------------------------------------------------------------- #
# DB index
# --------------------------------------------------------------------------- #
def load_db(db_dir: Path) -> dict[str, list[tuple[dict, dict]]]:
    """op_kind -> [(flat_signature_dict, raw_params), ...] for all stored records."""
    index: dict[str, list[tuple[dict, dict]]] = collections.defaultdict(list)
    for kind in DB_KINDS:
        f = db_dir / f"{kind}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                sig = raw_record_to_signature(rec)
            except Exception as e:  # noqa: BLE001
                print(f"  (skip {kind} record: {e})")
                continue
            index[kind].append((flatten(sig.to_json_dict()), rec.get("params", {})))
    return index


def flatten(sig_json: dict) -> dict[str, Any]:
    """OperatorSignature.to_json_dict -> flat {field: value} for diffing."""
    flat: dict[str, Any] = {
        "op_subtype": sig_json["op_subtype"],
        "dtype": sig_json["dtype"],
    }
    for part in ("shape", "parallel", "runtime"):
        for k, v in sig_json[part]:
            flat[f"{part}.{k}"] = v
    return flat


def best_candidate(qflat: dict, candidates: list[tuple[dict, dict]]):
    """Pick the stored record that most plausibly is the SAME kernel.

    Prefer a record whose shape.* fields all match the query (true shape
    match) — otherwise the residual diff is polluted by nearest-neighbour
    noise (wrong-shape / wrong-model records). Returns (candidate, shape_exact).
    """
    shape_keys = [k for k in qflat if k.startswith("shape.")]
    shape_exact = [(c, p) for c, p in candidates
                   if all(c.get(k) == qflat[k] for k in shape_keys)]
    pool = shape_exact or candidates
    best, best_score = None, -1
    for cflat, params in pool:
        score = sum(1 for k in qflat if cflat.get(k) == qflat[k])
        if score > best_score:
            best, best_score = (cflat, params), score
    return best, bool(shape_exact)


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def scenarios():
    return [
        ("qwen3_4b dense tp1 prefill2048", qwen3_4b(), deploy(tp=1), prefill_wl(2048)),
        ("qwen3_4b dense tp1 decode8",     qwen3_4b(), deploy(tp=1), decode_wl(8)),
        ("qwen3_4b dense tp2 prefill2048", qwen3_4b(), deploy(tp=2), prefill_wl(2048)),
        ("qwen3_30b moe tp4ep4 decode8",   qwen3_30b_a3b(), deploy(tp=4, ep=4, moe_ep=4), decode_wl(8)),
        ("qwen3_30b moe tp4ep4 prefill2048", qwen3_30b_a3b(), deploy(tp=4, ep=4, moe_ep=4), prefill_wl(2048)),
        ("qwen3_30b moe tp4ep4 mixed",     qwen3_30b_a3b(), deploy(tp=4, ep=4, moe_ep=4), mixed_wl(2048, 4)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="RTX_4090")
    ap.add_argument("--db", default="collector/data/operator_db/RTX_4090/vllm-0.19.1")
    ap.add_argument("--db-version", default="0.19.1",
                    help="framework_version the DB partition was captured under")
    ap.add_argument("--ignore-framework-version", action="store_true",
                    help="downgrade runtime-vs-DB version mismatch from fail to warning (debug only)")
    args = ap.parse_args()

    # #3 precheck: a version skew makes every lookup silently miss → roofline.
    try:
        verify_db_framework_version(
            args.db_version, ignore_mismatch=args.ignore_framework_version,
            source="explain_op_db_miss",
        )
    except OperatorDBVersionMismatch as e:
        print(f"\nVERSION MISMATCH (fail-closed): {e}\n"
              f"Re-run with --ignore-framework-version to diagnose canonical口径 "
              f"fields anyway (framework_version will show as a diff on every op).")
        raise SystemExit(2)

    db = load_db(Path(args.db))
    print(f"loaded DB records: " + ", ".join(f"{k}={len(db[k])}" for k in DB_KINDS))
    hw = get_hardware_profile(args.hw)

    # culprit field -> count of misses where it differs
    culprit = collections.Counter()
    hits = misses = 0

    for label, model, dep, wl in scenarios():
        deployment, runtime = dep
        eng = build_qwen_roofline_engine(model, deployment, runtime, hw)
        step = StepShape.from_workload(wl, runtime.execution.execution_mode)
        plan = eng.model.build_grouped_step(step)
        ops = [g.op for g in plan.groups
               if getattr(g.op, "op_kind", "") in DB_KINDS]
        print(f"\n{'='*78}\n# {label}  ({len(ops)} DB-eligible ops)\n{'='*78}")
        seen = set()
        for op in ops:
            key = (op.op_kind, op.op_subtype)
            if key in seen:
                continue
            seen.add(key)
            try:
                qsig = operator_to_signature(op)
            except Exception as e:  # noqa: BLE001
                print(f"  {op.op_kind}/{op.op_subtype}: signature error: {e}")
                continue
            qflat = flatten(qsig.to_json_dict())
            cands = db.get(op.op_kind, [])
            # exact hit iff some stored record has an identical flat signature
            is_hit = any(cflat == qflat for cflat, _ in cands)
            if is_hit:
                hits += 1
                print(f"  HIT  {op.op_kind}/{op.op_subtype}")
                continue
            misses += 1
            bc, shape_exact = best_candidate(qflat, cands)
            if bc is None:
                print(f"  MISS {op.op_kind}/{op.op_subtype}  (no record of this kind)")
                culprit[f"{op.op_kind}: no-record"] += 1
                continue
            tag = "shape-exact cand" if shape_exact else "NO shape-match cand (coverage/shape口径)"
            print(f"  MISS {op.op_kind}/{op.op_subtype}  [{tag}]")
            cflat, params = bc
            for k in sorted(set(qflat) | set(cflat)):
                qv, cv = qflat.get(k, "<MISSING>"), cflat.get(k, "<MISSING>")
                if qv != cv:
                    print(f"        XX {k:24} query={qv!r:26} stored={cv!r}")
                    # Only count field diffs as canonical口径 culprits when the
                    # candidate is a true shape match. Against a no-shape-match
                    # nearest neighbour, ALL field diffs (subtype/parallel/...) are
                    # noise — the real conclusion there is just "no record".
                    if shape_exact:
                        culprit[f"{op.op_kind}: {k}"] += 1
            if not shape_exact:
                culprit[f"{op.op_kind}: <no shape-exact record>"] += 1

    print(f"\n{'='*78}\n# SUMMARY: {hits} hit, {misses} miss\n{'='*78}")
    print("culprit fields (how many distinct (op_kind,subtype) misses each breaks):")
    for fld, n in culprit.most_common():
        print(f"  {n:3}  {fld}")


if __name__ == "__main__":
    main()
