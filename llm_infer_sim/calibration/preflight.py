"""Preflight check — Layer 1 校准前的快速健康检查 (~1-2 min).

在跑长 profile (2-4 小时) 前调一次, 验证:
  1. VLLM_VIRTUAL_BACKEND 未设 (calibration 必须真 GPU)
  2. vLLM / torch 可 import + GPU 可见
  3. catalog YAML 加载成功
  4. 起 vLLM 引擎 + worker_extension_cls 字符串解析成功
  5. 跑 1 个最小 dense shot (tokens=8) 真出 layerwise_profile timing
  6. timing 数字合理 (µs 量级, layer 数 > 0)

用法:
  python -m llm_infer_sim.calibration preflight \\
      --model Qwen/Qwen3-4B \\
      --model-type qwen3 \\
      --hardware RTX_4090

或直接 (test 默认 Qwen3-4B):
  python -m llm_infer_sim.calibration.preflight

退出码: 0 = 全过, 非 0 = 哪步挂了 (stderr 有错误细节).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def preflight(
    model: str = "Qwen/Qwen3-4B",
    model_type: str = "qwen3",
    hardware: str = "RTX_4090",
    dtype: str = "bfloat16",
    test_tokens: int = 8,
) -> int:
    """返 0 = 全过; 1-9 = 各阶段失败码."""
    print("=" * 70)
    print(f"  Preflight check (calibration) — {model} on {hardware} ({dtype})")
    print("=" * 70)

    # ---- 1. env ----
    print("\n[1/6] 检查 env 变量...")
    if os.environ.get("VLLM_VIRTUAL_BACKEND") == "1":
        print("  ❌ VLLM_VIRTUAL_BACKEND=1 已设, calibration 必须真 GPU 路径.")
        print("     解决: `unset VLLM_VIRTUAL_BACKEND` 后重试.")
        return 1
    print("  ✓ VLLM_VIRTUAL_BACKEND 未设")

    # ---- 2. dep import ----
    print("\n[2/6] import vllm / torch / GPU 可见性...")
    try:
        import torch
        import vllm  # noqa: F401
    except ImportError as e:
        print(f"  ❌ import 失败: {e}")
        return 2
    if not torch.cuda.is_available():
        print("  ❌ torch.cuda.is_available()=False; 无 GPU.")
        return 2
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  ✓ torch {torch.__version__}, vllm {vllm.__version__}")
    print(f"  ✓ {n_gpu} GPU 可见, 主卡: {gpu_name}")
    if "4090" not in gpu_name and hardware == "RTX_4090":
        print(f"  ⚠ hardware={hardware} 但探测到 {gpu_name}, 可能 hw_profile 偏差")

    # ---- 3. catalog 加载 ----
    print(f"\n[3/6] 加载 catalog (model_type={model_type})...")
    try:
        from llm_infer_sim.calibration.catalog import Catalog
        catalog = Catalog.load(model_type)
        print(f"  ✓ catalog 加载, {len(catalog)} 个 canonical entry")
        # 分类校验
        dense_slice = catalog.slice_for_category("dense")
        attn_slice = catalog.slice_for_category("attention")
        per_seq_slice = catalog.slice_for_category("per_sequence")
        print(f"    dense={len(dense_slice)} attention={len(attn_slice)} "
              f"per_seq={len(per_seq_slice)}")
        if len(dense_slice) == 0:
            print("  ⚠ dense slice 为空, catalog 可能错")
    except Exception as e:    # noqa: BLE001
        print(f"  ❌ catalog 加载失败: {type(e).__name__}: {e}")
        return 3

    # ---- 4. spin_up engine + worker_extension ----
    print(f"\n[4/6] 起 vLLM 引擎 ({model}, dtype={dtype}, tp=1)...")
    print("     (首次启动可能 30-60s, 含 model load)")
    try:
        from llm_infer_sim.calibration.engine import spin_up, spin_down
        engine = spin_up(
            model=model, dtype=dtype, tp=1,
            max_model_len=2048, max_num_seqs=8,
            gpu_memory_utilization=0.5,
        )
        print("  ✓ vLLM 引擎起来了")
    except Exception as e:    # noqa: BLE001
        print(f"  ❌ spin_up 失败: {type(e).__name__}: {e}")
        return 4

    try:
        # ---- 5. 跑最小 dense shot ----
        print(f"\n[5/6] fire 最小 dense shot (tokens={test_tokens})...")
        from llm_infer_sim.calibration.engine import fire_shot
        from llm_infer_sim.calibration.shots import Shot
        shot = Shot(kind="dense", num_new_tokens=test_tokens)
        rank_results = fire_shot(
            engine, shot.to_dict(),
            catalog_slice=catalog.slice_for_category("dense"),
            kind="dense", iterations=1,
        )
        print(f"  ✓ fire 返回 (ranks={len(rank_results)})")

        # ---- 6. 验证 timing 出来 ----
        print("\n[6/6] 验证 timing 样本合理性...")
        if not rank_results or not rank_results[0]:
            print("  ❌ rank 0 返回空 list — extract_samples 没匹中任何 canonical")
            print("     可能原因: catalog 类名跟 vLLM 0.20.1 实际模型类名不对齐")
            return 6
        samples = rank_results[0]
        print(f"  ✓ 拿到 {len(samples)} 个 timing sample")
        # 展示前 5 个
        print("\n  样本前 5 个 (μs):")
        print(f"  {'layer':<20} {'op_kind':<15} {'microseconds':>12}")
        for s in samples[:5]:
            print(f"  {s['layer']:<20} {s['op_kind']:<15} {s['microseconds']:>12.2f}")
        # sanity: time 在 µs-ms 量级 (0.1 us < t < 100 ms)
        invalid = [s for s in samples if s["microseconds"] <= 0 or s["microseconds"] > 1e5]
        if invalid:
            print(f"  ⚠ {len(invalid)} 个 sample 时间不合理 ({invalid[:3]})")
        # 看 catalog 里 dense canonical 是否都至少出 1 个 sample
        canonicals_seen = {s["layer"] for s in samples}
        dense_canonicals = set(catalog.slice_for_category("dense").keys())
        missing = dense_canonicals - canonicals_seen
        if missing:
            print(f"  ⚠ catalog 里这些 canonical 没出 sample: {sorted(missing)}")
            print("     (可能 vLLM 类名变了, 或没那条 module 跑)")
    finally:
        print("\n清理...")
        spin_down(engine)

    print("\n" + "=" * 70)
    print("  PREFLIGHT PASSED — 可以跑主 profile 了:")
    print(
        f"  python -m llm_infer_sim.calibration profile "
        f"--model {model} --model-type {model_type} --hardware {hardware}"
    )
    print("=" * 70)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="llm_infer_sim.calibration.preflight",
        description="Layer 1 校准前快速健康检查 (~1-2 min)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-type", default="qwen3")
    parser.add_argument("--hardware", default="RTX_4090")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--test-tokens", type=int, default=8,
                        help="最小 dense shot 的 token 数 (默认 8, 跑得最快)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    return preflight(
        model=args.model, model_type=args.model_type,
        hardware=args.hardware, dtype=args.dtype,
        test_tokens=args.test_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
