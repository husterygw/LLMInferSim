"""CSV → EfficiencyProfile YAML 拟合 (详设 §9.4.2 B.5).

输入:
    <raw_dir>/{dense,attention,per_sequence}.csv   B.3 产物
    <raw_dir>/bundle.yaml                          B.3 产物 (model + deploy)
    <raw_dir>/meta.yaml                            B.3 产物 (hw / dtype / 版本)

流程:
    for csv in {dense, per_sequence, attention}:
      for row in csv:
        op_kind, predicted_us = predicted_for(row.layer or "attention", row, bundle)
        efficiency = predicted_us / row.time_us           # ∈ (0, 1] 期望
        bucket    = bucket_for(op_kind, row.tokens / sequences / kv_lens)
        groups[(op_kind, dtype, bucket)].append(efficiency)
    entries = [median(g) for g in groups]
    write EfficiencyProfile YAML

输出:
    <out_yaml>: EfficiencyProfile.to_yaml 格式

简化版 (B.5 v1):
  - 桶单维, 跟 tokens / sequences 对数对齐 (`tokens<=16/128/1024`, `>1024`)
  - attention 桶 = (n_decode_bucket, kv_decode_bucket) 二维, 但用 string key 拼接
  - 不做 piecewise regression, 不算 confidence interval (median + n_samples)
  - 不映射 dtype 之外的精度 (fp8 fp16 视作同 dtype 桶, 由 cost model 的 op.dtype 区分)
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_infer_sim.calibration import csv_io
from llm_infer_sim.calibration.catalog import Catalog
from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.ops.attention import (
    attention_decode_flash,
    attention_prefill_flash,
)
from llm_infer_sim.core.ops.embedding import embedding as embedding_op
from llm_infer_sim.core.ops.embedding import lm_head as lm_head_op
from llm_infer_sim.core.ops.linear import (
    fused_gate_up_gemm,
    fused_qkv_gemm,
    linear_layer,
)
from llm_infer_sim.core.ops.normalization import mlp_activation, norm_layer
from llm_infer_sim.core.ops.attention import rope_kernel
from llm_infer_sim.core.cost_model.roofline import RooflineAnalyzer
from llm_infer_sim.core.profiles.efficiency_profile import (
    EfficiencyEntry,
    EfficiencyProfile,
)
from llm_infer_sim.core.profiles.hardware import get_hardware_profile
from llm_infer_sim.core.profiles.shape_buckets import (
    OP_KIND_ATTN,
    attention_bucket,
    kv_bucket as _shared_kv_bucket,
    sequence_bucket as _shared_sequence_bucket,
    token_bucket as _shared_token_bucket,
)

logger = logging.getLogger(__name__)


# ---------- Bundle 反序列化 ----------

@dataclass
class BundleSpec:
    """构造 OperatorProfile 所需的最小 bundle 信息 (从 bundle.yaml 读)."""
    # model
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_dim: int
    num_layers: int
    vocab_size: int
    is_moe: bool
    kv_lora_rank: int
    rope_head_dim: int
    # deploy
    tp: int
    w_byte: float
    a_byte: float
    kv_byte: float
    base_w_byte: float
    base_a_byte: float
    # hw
    hardware: str

    @classmethod
    def from_yaml(cls, path: Path) -> "BundleSpec":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        m = data["model"]
        d = data["deploy"]
        return cls(
            hidden_dim=int(m["hidden_dim"]),
            num_heads=int(m["num_heads"]),
            num_kv_heads=int(m["num_kv_heads"]),
            head_dim=int(m["head_dim"]),
            ffn_dim=int(m["ffn_dim"]),
            num_layers=int(m["num_layers"]),
            vocab_size=int(m["vocab_size"]),
            is_moe=bool(m.get("is_moe", False)),
            kv_lora_rank=int(m.get("kv_lora_rank", 0)),
            rope_head_dim=int(m.get("rope_head_dim", 0)),
            tp=int(d.get("tp", 1)),
            w_byte=float(d["w_byte"]),
            a_byte=float(d["a_byte"]),
            kv_byte=float(d["kv_byte"]),
            base_w_byte=float(d.get("base_w_byte", d["w_byte"])),
            base_a_byte=float(d.get("base_a_byte", d["a_byte"])),
            hardware=str(data["hardware"]),
        )


# ---------- 桶 ----------

# 桶函数从 core.profiles.shape_buckets 统一导入 (跟 op 构造函数同源, 不漂移).
token_bucket = _shared_token_bucket
sequence_bucket = _shared_sequence_bucket
kv_bucket = _shared_kv_bucket


# ---------- canonical → OperatorProfile 构造 ----------

def predicted_op_dense(canonical: str, tokens: int, b: BundleSpec) -> OperatorProfile | None:
    """构造 dense category 下 canonical 的 OperatorProfile.

    覆盖 Qwen3 dense catalog: embedding / layernorm / qkv_proj / qk_norm / rotary_emb /
    o_proj / gate_up_proj / act_fn / down_proj / final_layernorm.
    """
    h = b.hidden_dim
    tp = max(1, b.tp)
    if canonical == "embedding":
        return embedding_op(tokens, b.vocab_size, h, b.base_w_byte, b.base_a_byte)
    if canonical in ("layernorm", "qk_norm", "final_layernorm"):
        # qk_norm 的 hidden 不是 model hidden 而是 head_dim; 第一版简化为 hidden.
        # 偏差: qk_norm bytes ≈ tokens × head_dim × a_byte, hidden=2560 vs head_dim=128
        # 差 20×; 算 efficiency 时 predicted 偏大 → efficiency 偏小. 后续 v2 区分.
        return norm_layer(canonical, tokens, h, b.base_a_byte)
    if canonical == "qkv_proj":
        q_per_tp = b.num_heads // tp
        kv_per_tp = max(1, b.num_kv_heads // tp)
        return fused_qkv_gemm(
            "qkv_proj", hidden=h,
            num_q_heads_per_tp=q_per_tp, num_kv_heads_per_tp=kv_per_tp,
            head_dim=b.head_dim, tokens=tokens,
            w_byte=b.w_byte, a_byte=b.a_byte, kv_byte=b.kv_byte,
        )
    if canonical == "o_proj":
        q_dim = (b.num_heads // tp) * b.head_dim
        return linear_layer(
            "o_proj", ic=q_dim, oc=h, tokens=tokens,
            w_byte=b.w_byte, a_byte=b.a_byte, kv_byte=b.kv_byte,
        )
    if canonical == "rotary_emb":
        q_per_tp = b.num_heads // tp
        kv_per_tp = max(1, b.num_kv_heads // tp)
        return rope_kernel(
            "rotary_emb", tokens,
            num_q_heads_per_tp=q_per_tp,
            num_kv_heads_per_tp=kv_per_tp,
            head_dim=b.head_dim, a_byte=b.a_byte,
        )
    if canonical == "gate_up_proj":
        ffn_per_tp = b.ffn_dim // tp
        return fused_gate_up_gemm(
            "gate_up_proj", hidden=h, intermediate_per_tp=ffn_per_tp,
            tokens=tokens, w_byte=b.w_byte, a_byte=b.a_byte,
        )
    if canonical == "act_fn":
        ffn_per_tp = b.ffn_dim // tp
        return mlp_activation("act_fn", tokens, ffn_per_tp, b.a_byte)
    if canonical == "down_proj":
        ffn_per_tp = b.ffn_dim // tp
        return linear_layer(
            "down_proj", ic=ffn_per_tp, oc=h, tokens=tokens,
            w_byte=b.w_byte, a_byte=b.a_byte, kv_byte=b.kv_byte,
        )
    return None


def predicted_op_per_seq(canonical: str, sequences: int, b: BundleSpec) -> OperatorProfile | None:
    if canonical == "lm_head":
        return lm_head_op(sequences, b.vocab_size, b.hidden_dim,
                          max(1, b.tp), b.base_w_byte, b.base_a_byte)
    return None


def predicted_op_attention(
    prefill_chunk: int, kv_prefill: int, n_decode: int, kv_decode: int,
    b: BundleSpec,
    onchip_buffer: float = 72 * 1024 * 1024,    # RTX 4090 L2 72 MB
) -> OperatorProfile | None:
    """attention shot 的 op. 这里用 flash 路径 (4090 vLLM 默认 FA2).

    Mixed shot (prefill_chunk > 0 AND n_decode > 0) 推到 v2.
    onchip_buffer 默认 4090 L2 大小, 调用方可显式覆盖给其他 HW.
    """
    if b.kv_lora_rank > 0:
        # MLA 模型走单独 op (FlashMLA), 当前不在 B.5 v1 范围 (Qwen3 不是 MLA)
        logger.debug("MLA 模型 attention efficiency 跳过 (B.5 v1 暂不支持)")
        return None
    n_q_per_tp = b.num_heads // max(1, b.tp)
    n_kv_per_tp = max(1, b.num_kv_heads // max(1, b.tp))
    if prefill_chunk > 0 and n_decode == 0:
        # 纯 prefill (chunked 或 initial), 用 prefill_flash
        seqlen = prefill_chunk + kv_prefill
        ops = attention_prefill_flash(
            seqlen=seqlen, batchsize=1,
            num_attention_heads=n_q_per_tp,
            num_key_value_heads=n_kv_per_tp,
            head_size=b.head_dim,
            a_byte=b.a_byte, kv_byte=b.kv_byte,
            onchip_buffer=onchip_buffer,
        )
        return _sum_ops(ops, name="attention")
    if prefill_chunk == 0 and n_decode > 0:
        # 纯 decode
        ops = attention_decode_flash(
            seqlen=kv_decode, batchsize=n_decode,
            num_attention_heads=n_q_per_tp,
            num_key_value_heads=n_kv_per_tp,
            head_size=b.head_dim,
            a_byte=b.a_byte, kv_byte=b.kv_byte,
            onchip_buffer=onchip_buffer,
        )
        return _sum_ops(ops, name="attention")
    # mixed 简化: 推到 v2
    return None


def _sum_ops(ops: list[OperatorProfile], name: str) -> OperatorProfile:
    """把 prefill_flash / decode_flash 多 op 合并为单 OperatorProfile (sum flops + bytes).

    layerwise_profile 里 attention 是单 `Attention` module 调用, 时间是融合后的总和.
    """
    return OperatorProfile(
        name=name,
        op_category="attention",
        flops=sum(o.flops for o in ops),
        load_weight=sum(o.load_weight for o in ops),
        load_act=sum(o.load_act for o in ops),
        store_act=sum(o.store_act for o in ops),
        load_kv_cache=sum(o.load_kv_cache for o in ops),
        store_kv_cache=sum(o.store_kv_cache for o in ops),
    )


# ---------- 拟合主流程 ----------

@dataclass
class FitGroup:
    """同 (op_kind, dtype, shape_key) 桶内累计的 efficiency 样本."""
    op_kind: str
    dtype: str
    shape_key: str
    efficiencies: list[float]

    def to_entry(self, source: str) -> EfficiencyEntry:
        med = statistics.median(self.efficiencies) if self.efficiencies else 1.0
        return EfficiencyEntry(
            op_kind=self.op_kind, dtype=self.dtype, shape_key=self.shape_key,
            efficiency=float(med),
            confidence=_confidence_from_spread(self.efficiencies),
            n_samples=len(self.efficiencies),
            source=source,
        )


def _confidence_from_spread(samples: list[float]) -> float:
    """简单 confidence 估算: 1 - (max-min)/median, clamp [0, 1].

    完全一致 → 1.0; 差异越大 confidence 越低.
    """
    if len(samples) < 2:
        return 1.0
    s = sorted(samples)
    med = s[len(s) // 2]
    if med <= 0:
        return 0.0
    spread = (s[-1] - s[0]) / med
    return max(0.0, min(1.0, 1.0 - spread))


def fit_efficiency(
    raw_dir: str | Path,
    out_yaml: str | Path,
    captured_at: str = "",
    vllm_version: str = "",
) -> EfficiencyProfile:
    """读 raw_dir CSVs + bundle.yaml + meta.yaml, 拟合并写 out_yaml.

    Returns:
        组装好的 EfficiencyProfile (已写 out_yaml).

    Raises:
        FileNotFoundError: bundle.yaml 不存在 (profile 阶段未跑 / 用了 mock engine).
    """
    raw_dir = Path(raw_dir)
    bundle_path = raw_dir / "bundle.yaml"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"bundle.yaml 不存在: {bundle_path}. profile 阶段没跑 真 vLLM 或被 mock?"
        )
    spec = BundleSpec.from_yaml(bundle_path)

    # meta (取 captured_at / vllm_version 等元信息)
    meta = _read_meta(raw_dir / "meta.yaml")
    captured_at = captured_at or meta.get("captured_at", "")
    vllm_version = vllm_version or meta.get("vllm_version", "")
    dtype_str = meta.get("dtype", _dtype_from_a_byte(spec.a_byte))

    # hardware 拿 effective hardware (apply roofline efficiency 之前 placeholder)
    hw = get_hardware_profile(spec.hardware)
    analyzer = RooflineAnalyzer(
        hw,
        w_bit=int(spec.w_byte * 8),
        a_bit=int(spec.a_byte * 8),
        kv_bit=int(spec.kv_byte * 8),
    )

    groups: dict[tuple[str, str, str], FitGroup] = {}
    source = f"{spec.hardware}/{Path(raw_dir).name}/{dtype_str}"

    # ---- dense ----
    dense_rows = csv_io.read_dense(raw_dir / "dense.csv")
    catalog = _load_catalog(meta)
    op_kind_by_canonical = {e.canonical: e.op_kind for e in catalog} if catalog else {}
    skipped = 0
    for r in dense_rows:
        op = predicted_op_dense(r.layer, r.tokens, spec)
        if op is None:
            skipped += 1
            continue
        predicted_s = analyzer.analyze(op).total_time
        if predicted_s <= 0 or r.time_us <= 0:
            skipped += 1
            continue
        eff = (predicted_s * 1e6) / r.time_us
        op_kind = op_kind_by_canonical.get(r.layer, r.layer)
        bucket = token_bucket(r.tokens)
        _add(groups, op_kind, dtype_str, bucket, eff)

    # ---- per_sequence ----
    ps_rows = csv_io.read_per_sequence(raw_dir / "per_sequence.csv")
    for r in ps_rows:
        op = predicted_op_per_seq(r.layer, r.sequences, spec)
        if op is None:
            skipped += 1
            continue
        predicted_s = analyzer.analyze(op).total_time
        if predicted_s <= 0 or r.time_us <= 0:
            skipped += 1
            continue
        eff = (predicted_s * 1e6) / r.time_us
        op_kind = op_kind_by_canonical.get(r.layer, r.layer)
        bucket = sequence_bucket(r.sequences)
        _add(groups, op_kind, dtype_str, bucket, eff)

    # ---- attention ----
    attn_rows = csv_io.read_attention(raw_dir / "attention.csv")
    for r in attn_rows:
        op = predicted_op_attention(
            r.prefill_chunk, r.kv_prefill, r.n_decode, r.kv_decode, spec,
            onchip_buffer=float(hw.onchip_buffer),
        )
        if op is None:
            skipped += 1
            continue
        predicted_s = analyzer.analyze(op).total_time
        if predicted_s <= 0 or r.time_us <= 0:
            skipped += 1
            continue
        eff = (predicted_s * 1e6) / r.time_us
        bucket = attention_bucket(r.prefill_chunk, r.kv_prefill, r.n_decode, r.kv_decode)
        _add(groups, OP_KIND_ATTN, dtype_str, bucket, eff)

    logger.info(
        "fit: dense=%d per_seq=%d attn=%d rows, %d skipped, %d groups",
        len(dense_rows), len(ps_rows), len(attn_rows), skipped, len(groups),
    )

    # ---- 组装 EfficiencyProfile ----
    profile = EfficiencyProfile(
        hardware=spec.hardware,
        captured_at=captured_at,
        vllm_version=vllm_version,
        w_byte=spec.w_byte, a_byte=spec.a_byte, kv_byte=spec.kv_byte,
        # default 取整体几何平均 / 简单平均 fallback. 这里用 median over all entries.
        default_compute=_default_efficiency(groups),
        default_mem=_default_efficiency(groups),
        default_comm=1.0,
    )
    for g in groups.values():
        profile.add_entry(g.to_entry(source))
    profile.to_yaml(out_yaml)
    return profile


def _add(
    groups: dict[tuple[str, str, str], FitGroup],
    op_kind: str, dtype: str, shape_key: str, efficiency: float,
) -> None:
    key = (op_kind, dtype, shape_key)
    g = groups.get(key)
    if g is None:
        g = FitGroup(op_kind=op_kind, dtype=dtype, shape_key=shape_key, efficiencies=[])
        groups[key] = g
    g.efficiencies.append(efficiency)


def _default_efficiency(groups: dict) -> float:
    """所有桶 efficiency 的中位数, 当 lookup miss 时 fallback."""
    all_meds = [
        statistics.median(g.efficiencies) for g in groups.values() if g.efficiencies
    ]
    if not all_meds:
        return 1.0
    return float(statistics.median(all_meds))


def _read_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def _load_catalog(meta: dict) -> Catalog | None:
    model_type = meta.get("model_type")
    if not model_type:
        return None
    try:
        return Catalog.load(str(model_type))
    except FileNotFoundError:
        return None


def _dtype_from_a_byte(a_byte: float) -> str:
    if a_byte <= 0.5:
        return "fp4"
    if a_byte <= 1.0:
        return "fp8"
    if a_byte <= 2.0:
        return "bf16"
    return "fp32"
