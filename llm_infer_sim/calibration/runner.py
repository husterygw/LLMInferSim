"""顶层 calibration orchestration (详设 §9.4.2 B.3).

流程:
  1. spin_up vLLM (真 GPU, worker_extension_cls=LayerwiseProfileExtension)
  2. 加载 catalog (按 model_type 找 models/<type>.yaml)
  3. 三类 category 依次扫:
       for kind in [dense, attention, per_sequence]:
         slice_ = catalog.slice_for_category(kind)
         for shot in shots[kind]:
            samples = fire_shot(...)
            rows = samples_to_*_rows(shot, samples)
            sink.write_rows(rows)
  4. 写 meta.yaml (vllm/torch/cuda 版本 + engine kwargs + shot grid)
  5. spin_down

resume 行为: --resume 时读已有 CSV 提取 visited key, 跳过已 visited 的 shot.
"""
from __future__ import annotations

import datetime
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_infer_sim.calibration import csv_io
from llm_infer_sim.calibration.catalog import Catalog
from llm_infer_sim.calibration.shots import (
    ATTENTION_SHOTS,
    DENSE_SHOTS,
    PER_SEQUENCE_SHOTS,
    Shot,
)
from llm_infer_sim.calibration.timings import TimingSample

logger = logging.getLogger(__name__)


# ---------- category 配置 ----------

@dataclass(frozen=True)
class CategorySpec:
    """一类校准的元信息: 文件名 / 列名 / shot grid / sample→row 转换 / visited 提取."""
    kind: str
    filename: str
    columns: tuple[str, ...]
    shots: tuple[Shot, ...]
    samples_to_rows: Any        # callable (shot, samples) → list[dict]
    visited_keys: Any           # callable (path) → set[tuple]


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        kind="dense",
        filename="dense.csv",
        columns=csv_io.DENSE_COLS,
        shots=DENSE_SHOTS,
        samples_to_rows=csv_io.samples_to_dense_rows,
        visited_keys=csv_io.visited_keys_dense,
    ),
    CategorySpec(
        kind="attention",
        filename="attention.csv",
        columns=csv_io.ATTN_COLS,
        shots=ATTENTION_SHOTS,
        samples_to_rows=csv_io.samples_to_attn_rows,
        visited_keys=csv_io.visited_keys_attention,
    ),
    CategorySpec(
        kind="per_sequence",
        filename="per_sequence.csv",
        columns=csv_io.PER_SEQ_COLS,
        shots=PER_SEQUENCE_SHOTS,
        samples_to_rows=csv_io.samples_to_per_seq_rows,
        visited_keys=csv_io.visited_keys_per_sequence,
    ),
)


# ---------- run_calibration ----------

def run_calibration(
    model: str,
    model_type: str,
    hardware: str,
    dtype: str = "bfloat16",
    output_root: str | Path = "configs/efficiency/raw",
    tp: int = 1,
    iterations: int = 3,
    kinds: tuple[str, ...] | None = None,
    resume: bool = True,
    max_model_len: int = 20480,
    max_num_seqs: int = 16,
    engine_factory: Any = None,
    fire_fn: Any = None,
) -> Path:
    """跑完整 calibration, 写 CSV + meta.yaml. 返回输出目录.

    output 路径: `<output_root>/<hardware>/<model_subpath>/<dtype>/tp<N>/{dense,attention,per_sequence}.csv`.

    Args:
        model: HF id / 本地路径.
        model_type: HF model_type, 用来找 catalog YAML (例 "qwen3").
        hardware: 自由命名, 决定输出目录 (例 "RTX_4090").
        dtype: bf16 / fp16 / fp8 ...
        output_root: 输出 root (默认 configs/efficiency/raw).
        tp: tensor_parallel_size.
        iterations: 每 shot 内部 forward 次数 (默认 3).
        kinds: 只跑这些 kind ("dense"/"attention"/"per_sequence"), None=全跑.
        resume: True 时读已有 CSV 跳过 visited shot.
        engine_factory: 给 mock 测试用; 默认 calibration.engine.spin_up.
        fire_fn: 给 mock 测试用; 默认 calibration.engine.fire_shot.

    Returns:
        输出目录 Path.
    """
    if engine_factory is None:
        from llm_infer_sim.calibration.engine import spin_up as engine_factory  # noqa: E501
    if fire_fn is None:
        from llm_infer_sim.calibration.engine import fire_shot as fire_fn  # noqa: E501

    catalog = Catalog.load(model_type)
    out_dir = _build_output_dir(output_root, hardware, model, dtype, tp)
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds_to_run = set(kinds) if kinds is not None else {c.kind for c in CATEGORIES}
    selected = [c for c in CATEGORIES if c.kind in kinds_to_run]
    if not selected:
        raise ValueError(f"kinds={kinds!r} 没匹中任何 category")

    logger.info("calibration: model=%s hardware=%s dtype=%s tp=%d iterations=%d kinds=%s",
                model, hardware, dtype, tp, iterations, [c.kind for c in selected])

    # 起 engine — 唯一一次, 三 category 共享
    engine = engine_factory(
        model=model, dtype=dtype, tp=tp,
        max_model_len=max_model_len, max_num_seqs=max_num_seqs,
    )

    try:
        # 落 bundle.yaml (fit 阶段读它构造 ProfileBundle, 算 predicted_us)
        _maybe_write_bundle(engine, out_dir, hardware)
        for cat in selected:
            _run_category(catalog, cat, engine, fire_fn, out_dir, iterations, resume)
    finally:
        from llm_infer_sim.calibration.engine import spin_down
        spin_down(engine)

    # 落 meta.yaml
    _write_meta(out_dir, model, model_type, hardware, dtype, tp, iterations)
    logger.info("calibration done. Output: %s", out_dir)
    return out_dir


def _run_category(
    catalog: Catalog,
    cat: CategorySpec,
    engine: Any,
    fire_fn: Any,
    out_dir: Path,
    iterations: int,
    resume: bool,
) -> None:
    csv_path = out_dir / cat.filename
    slice_ = catalog.slice_for_category(cat.kind)
    if not slice_:
        logger.warning("category %r: catalog 切片为空, 跳过", cat.kind)
        return

    visited: set[tuple] = cat.visited_keys(csv_path) if resume else set()
    if visited:
        logger.info("category %r: %d shot 已 visited, resume 跳过", cat.kind, len(visited))

    fired = 0
    with csv_io.CsvSink(csv_path, cat.columns) as sink:
        for shot in cat.shots:
            if shot.csv_key() in visited:
                continue
            try:
                rank_results = fire_fn(
                    engine, shot.to_dict(), slice_, cat.kind, iterations,
                )
            except Exception as e:    # noqa: BLE001
                logger.error("shot %s 失败, skip: %s: %s",
                             shot.csv_key(), type(e).__name__, e)
                continue
            # 多 rank: 取 rank 0 (tp=1 时只一个); rank 间应近似相同 (我们 cost
            # model 假设 symmetric workers).
            samples_dicts = rank_results[0] if rank_results else []
            samples = [TimingSample(**d) for d in samples_dicts]
            rows = cat.samples_to_rows(shot, samples)
            n = sink.write_rows(rows)
            fired += 1
            logger.debug("shot %s → %d rows", shot.csv_key(), n)
    logger.info("category %r done: fired %d shots", cat.kind, fired)


def _build_output_dir(
    output_root: str | Path,
    hardware: str,
    model: str,
    dtype: str,
    tp: int,
) -> Path:
    """输出路径: <output_root>/<hardware>/<model_subpath>/<dtype>/tp<N>/.

    本地路径模型时取最后目录名 (e.g. /data/.../Qwen3-4B → Qwen3-4B).
    HF id 保留 (e.g. Qwen/Qwen3-4B → Qwen/Qwen3-4B).
    """
    p = Path(model)
    if p.is_absolute() and p.exists():
        model_sub = p.name
    else:
        model_sub = model
    return Path(output_root) / hardware / model_sub / dtype / f"tp{tp}"


def _maybe_write_bundle(engine: Any, out_dir: Path, hardware: str) -> None:
    """从 vllm engine 抽 ProfileBundle, 落 bundle.yaml 给 fit 阶段用.

    含 model_config 全部字段 + deploy 量化字节 + tp_size + hardware 名。
    fit 阶段读它 → 构造 ProfileBundle → 算 predicted_us.

    失败不挂 (例 mock engine 没 llm_engine 属性) — fit 阶段会用 hf_config 兜底。
    """
    import yaml
    try:
        from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
        vllm_config = engine.llm_engine.vllm_config
        bundle = extract_profile_bundle(vllm_config)
    except Exception as e:    # noqa: BLE001
        logger.warning("跳过 bundle.yaml (mock 或 vllm 路径异常): %s", e)
        return

    m = bundle.model
    d = bundle.deploy
    data = {
        "hardware": hardware,
        "model": {
            "name": m.name,
            "hidden_dim": m.hidden_dim,
            "num_heads": m.num_heads,
            "num_kv_heads": m.num_kv_heads,
            "head_dim": m.head_dim,
            "ffn_dim": m.ffn_dim,
            "num_layers": m.num_layers,
            "vocab_size": m.vocab_size,
            "is_moe": m.is_moe,
            "num_experts": m.num_experts,
            "num_activated_experts": m.num_activated_experts,
            "expert_dim": m.expert_dim,
            "num_shared_experts": m.num_shared_experts,
            "kv_lora_rank": m.kv_lora_rank,
            "qk_nope_head_dim": m.qk_nope_head_dim,
            "rope_head_dim": m.rope_head_dim,
            "v_head_dim": m.v_head_dim,
        },
        "deploy": {
            "tp": d.tp,
            "dp": d.dp,
            "ep": d.ep,
            "w_byte": d.w_byte,
            "a_byte": d.a_byte,
            "kv_byte": d.kv_byte,
            "base_w_byte": d.base_w_byte,
            "base_a_byte": d.base_a_byte,
            "use_flash_attention": d.use_flash_attention,
        },
    }
    (out_dir / "bundle.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )


def _write_meta(
    out_dir: Path,
    model: str,
    model_type: str,
    hardware: str,
    dtype: str,
    tp: int,
    iterations: int,
) -> None:
    """写 meta.yaml 含版本 / 时戳 / engine 参数."""
    import yaml
    # 强转 str: torch.__version__ 是 TorchVersion 类 (str subclass),
    # yaml.safe_dump 不认; vllm.__version__ 同理. 显式 str() 转 plain.
    try:
        import torch
        torch_version = str(torch.__version__)
        cuda_version = str(getattr(torch.version, "cuda", "unknown") or "unknown")
    except ImportError:
        torch_version = "unknown"
        cuda_version = "unknown"
    try:
        import vllm
        vllm_version = str(vllm.__version__)
    except ImportError:
        vllm_version = "unknown"

    meta = {
        "model": model,
        "model_type": model_type,
        "hardware": hardware,
        "dtype": dtype,
        "tp": tp,
        "iterations": iterations,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "vllm_version": vllm_version,
    }
    (out_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
    )
