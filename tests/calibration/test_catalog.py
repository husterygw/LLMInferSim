"""catalog.py — model_type catalog YAML 加载 + 匹配规则 (B.1)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_infer_sim.calibration.catalog import Catalog, CatalogEntry


# ---- 加载 ----

def test_load_qwen3_catalog_from_default_dir():
    """calibration/models/qwen3.yaml 应该开箱可用."""
    cat = Catalog.load("qwen3")
    # 至少应有: embedding / qkv_proj / o_proj / gate_up_proj / down_proj / lm_head / attention
    canonicals = {e.canonical for e in cat}
    assert canonicals.issuperset({
        "embedding", "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
        "lm_head", "attention", "layernorm", "qk_norm",
    })


def test_load_unknown_model_type_raises():
    with pytest.raises(FileNotFoundError, match="未找到"):
        Catalog.load("nonexistent_model_type")


def test_load_qwen3_op_kind_mapping():
    """op_kind 字段应正确解析 (rmsnorm / dense_gemm / rope / swiglu / attn / embedding)."""
    cat = Catalog.load("qwen3")
    by_canonical = {e.canonical: e for e in cat}
    assert by_canonical["qkv_proj"].op_kind == "dense_gemm"
    assert by_canonical["layernorm"].op_kind == "rmsnorm"
    assert by_canonical["qk_norm"].op_kind == "rmsnorm"
    assert by_canonical["rotary_emb"].op_kind == "rope"
    assert by_canonical["act_fn"].op_kind == "swiglu"
    assert by_canonical["attention"].op_kind == "attn"
    assert by_canonical["embedding"].op_kind == "embedding"


def test_load_qwen3_within_constraints():
    """qk_norm 应受 within: Qwen3Attention 约束, layernorm 受 within: Qwen3DecoderLayer."""
    cat = Catalog.load("qwen3")
    by_canonical = {e.canonical: e for e in cat}
    assert by_canonical["qk_norm"].within == "Qwen3Attention"
    assert by_canonical["layernorm"].within == "Qwen3DecoderLayer"
    assert by_canonical["embedding"].within is None  # 任意 ancestor 都行


# ---- 匹配规则 ----

def _make_cat(*entries: CatalogEntry) -> Catalog:
    return Catalog(model_type="test", entries=list(entries))


def test_match_no_within_matches_anywhere():
    """within=None 在任意 ancestor 链都该匹中."""
    cat = _make_cat(CatalogEntry("emb", "Embedding", None, "embedding"))
    m = cat.match("Embedding", ancestors=["Model", "OtherWrapper"])
    assert m is not None and m.canonical == "emb"


def test_match_within_in_ancestor_chain():
    cat = _make_cat(
        CatalogEntry("qk_norm", "RMSNorm", "Qwen3Attention", "rmsnorm"),
    )
    m = cat.match("RMSNorm", ancestors=["Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention"])
    assert m is not None and m.canonical == "qk_norm"


def test_match_within_missing_skips():
    cat = _make_cat(
        CatalogEntry("qk_norm", "RMSNorm", "Qwen3Attention", "rmsnorm"),
    )
    # ancestor 链没 Qwen3Attention → 不该匹
    m = cat.match("RMSNorm", ancestors=["Qwen3Model", "Qwen3DecoderLayer"])
    assert m is None


def test_match_disambiguation_within_deepest_wins():
    """Qwen3 双 RMSNorm 场景: ancestor 链既有 DecoderLayer 又有 Attention,
    应优先 within=Qwen3Attention (内层) 的 entry."""
    cat = _make_cat(
        # 故意先放外层 entry (yaml 顺序无关)
        CatalogEntry("layernorm", "RMSNorm", "Qwen3DecoderLayer", "rmsnorm"),
        CatalogEntry("qk_norm", "RMSNorm", "Qwen3Attention", "rmsnorm"),
    )
    # 内层节点: ancestors = [..., DecoderLayer, Attention]
    m = cat.match("RMSNorm", ancestors=["Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention"])
    assert m is not None
    assert m.canonical == "qk_norm"   # within 在 ancestors index=2 (最深)


def test_match_disambiguation_outer_when_no_inner_ancestor():
    """同样两个 entry, 但 node 在外层 (没 Attention ancestor) → 应匹 layernorm."""
    cat = _make_cat(
        CatalogEntry("layernorm", "RMSNorm", "Qwen3DecoderLayer", "rmsnorm"),
        CatalogEntry("qk_norm", "RMSNorm", "Qwen3Attention", "rmsnorm"),
    )
    m = cat.match("RMSNorm", ancestors=["Qwen3Model", "Qwen3DecoderLayer"])
    assert m is not None and m.canonical == "layernorm"


def test_match_within_none_ranks_lowest():
    """有 within 的 entry 命中时, 无 within 的不该胜出."""
    cat = _make_cat(
        CatalogEntry("rms_any", "RMSNorm", None, "rmsnorm"),
        CatalogEntry("qk_norm", "RMSNorm", "Qwen3Attention", "rmsnorm"),
    )
    m = cat.match("RMSNorm", ancestors=["Qwen3Attention"])
    assert m.canonical == "qk_norm"

    # 但 ancestor 链没 Qwen3Attention 时, 无 within 兜底
    m2 = cat.match("RMSNorm", ancestors=["OtherWrapper"])
    assert m2.canonical == "rms_any"


def test_match_no_candidate_returns_none():
    cat = _make_cat(CatalogEntry("qkv_proj", "QKVParallelLinear", None, "dense_gemm"))
    assert cat.match("RowParallelLinear", ancestors=[]) is None


# ---- slice ----

def test_slice_for_op_kinds_filters():
    cat = _make_cat(
        CatalogEntry("qkv_proj", "QKVParallelLinear", None, "dense_gemm"),
        CatalogEntry("layernorm", "RMSNorm", None, "rmsnorm"),
        CatalogEntry("act_fn", "SiluAndMul", None, "swiglu"),
    )
    sl = cat.slice_for_op_kinds({"dense_gemm"})
    assert set(sl.keys()) == {"qkv_proj"}
    assert sl["qkv_proj"]["vllm"] == "QKVParallelLinear"


def test_slice_full_no_filter():
    cat = _make_cat(
        CatalogEntry("a", "ClsA", None, "k1"),
        CatalogEntry("b", "ClsB", None, "k2"),
    )
    sl = cat.slice_for_op_kinds(None)
    assert set(sl.keys()) == {"a", "b"}


# ---- bad YAML ----

def test_load_bad_yaml_raises():
    """缺 required 字段 vllm: → ValueError."""
    yaml_text = """
model_type: bad
entries:
  badentry:
    op_kind: rmsnorm
"""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.yaml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="catalog entry"):
            Catalog.load("bad", models_dir=Path(td))
