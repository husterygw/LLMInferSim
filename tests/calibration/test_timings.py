"""timings.py — LayerwiseProfileResults 树解析 (B.1).

全 mock: 构造 dict 形式的 model_stats tree, 不依赖 vLLM / torch.profiler.
"""
from __future__ import annotations

from llm_infer_sim.calibration.timings import (
    TimingSample,
    extract_samples,
    _strip_class_name,
)


def _node(name: str, cuda_time_us: float = 0.0, invocations: int = 1,
          children: list | None = None) -> dict:
    """Build a mock model_stats tree node matching vLLM's convert_stats_to_dict shape."""
    return {
        "entry": {
            "name": name,
            "cuda_time_us": cuda_time_us,
            "cpu_time_us": 0.0,
            "invocations": invocations,
        },
        "children": children or [],
    }


# ---- strip class name ----

def test_strip_class_name_with_repr():
    assert _strip_class_name("QKVParallelLinear(in_features=4096, ...)") == "QKVParallelLinear"


def test_strip_class_name_no_paren():
    assert _strip_class_name("RMSNorm") == "RMSNorm"


# ---- DFS & match ----

_SLICE = {
    "qkv_proj":  {"vllm": "QKVParallelLinear", "within": None, "op_kind": "dense_gemm"},
    "o_proj":    {"vllm": "RowParallelLinear", "within": "Qwen3Attention", "op_kind": "dense_gemm"},
    "down_proj": {"vllm": "RowParallelLinear", "within": "Qwen3MLP", "op_kind": "dense_gemm"},
    "layernorm": {"vllm": "RMSNorm", "within": "Qwen3DecoderLayer", "op_kind": "rmsnorm"},
    "qk_norm":   {"vllm": "RMSNorm", "within": "Qwen3Attention", "op_kind": "rmsnorm"},
    "embedding": {"vllm": "VocabParallelEmbedding", "within": None, "op_kind": "embedding"},
}


def test_extract_simple_match():
    tree = _node("Qwen3Model(...)", cuda_time_us=0, invocations=1, children=[
        _node("VocabParallelEmbedding(...)", cuda_time_us=42.0, invocations=1),
    ])
    samples = extract_samples(tree, _SLICE)
    assert len(samples) == 1
    assert samples[0].layer == "embedding"
    assert samples[0].op_kind == "embedding"
    assert samples[0].microseconds == 42.0


def test_extract_per_call_not_aggregated():
    """vLLM model_stats 是 per-call tree, 同 module 多次调用展开为兄弟节点.

    每个 entry 对应一次调用, cuda_time_us 是该次单独 latency. 我们对每次命中产
    1 个 sample, 聚合 (median 等) 留到 fit 阶段。
    """
    tree = _node("Qwen3Model(...)", children=[
        _node("QKVParallelLinear(...)", cuda_time_us=100.0),
        _node("QKVParallelLinear(...)", cuda_time_us=105.0),
        _node("QKVParallelLinear(...)", cuda_time_us=95.0),
    ])
    samples = extract_samples(tree, _SLICE)
    assert len(samples) == 3
    assert all(s.layer == "qkv_proj" for s in samples)
    assert [s.microseconds for s in samples] == [100.0, 105.0, 95.0]


def test_extract_skips_zero_cuda_time():
    """cuda_us=0 (CPU-only op 或 placeholder) 不出 sample."""
    tree = _node("Root", children=[
        _node("QKVParallelLinear(...)", cuda_time_us=0.0),
    ])
    samples = extract_samples(tree, _SLICE)
    assert samples == []


def test_extract_disambiguation_within_deepest_wins():
    """Qwen3 双 RMSNorm: layer 内 RMSNorm 在 DecoderLayer 直接孩子, qk_norm 在 Attention.

    DecoderLayer
      ├── RMSNorm (input_layernorm)   ← 外层, within=DecoderLayer
      ├── Qwen3Attention
      │     └── RMSNorm (qk_norm)     ← 内层, within=Attention 更深
      └── RMSNorm (post_attn_layernorm)
    """
    tree = _node("Qwen3Model(...)", children=[
        _node("Qwen3DecoderLayer(...)", children=[
            _node("RMSNorm(...)", cuda_time_us=10.0, invocations=1),  # outer
            _node("Qwen3Attention(...)", children=[
                _node("RMSNorm(...)", cuda_time_us=5.0, invocations=1),  # inner qk_norm
            ]),
            _node("RMSNorm(...)", cuda_time_us=10.0, invocations=1),  # outer post
        ]),
    ])
    samples = extract_samples(tree, _SLICE)
    # 应有 3 个 sample: 2 layernorm (外层 RMSNorm 两次) + 1 qk_norm
    layers = [s.layer for s in samples]
    assert layers.count("layernorm") == 2
    assert layers.count("qk_norm") == 1


def test_extract_unmatched_node_skipped():
    """不在 catalog 的 node 跳过."""
    tree = _node("Qwen3Model(...)", children=[
        _node("SomeUnknownClass(...)", cuda_time_us=100.0, invocations=1),
        _node("QKVParallelLinear(...)", cuda_time_us=50.0, invocations=1),
    ])
    samples = extract_samples(tree, _SLICE)
    assert len(samples) == 1
    assert samples[0].layer == "qkv_proj"


def test_extract_within_constraint_skips_outside():
    """o_proj 要求 within=Qwen3Attention, 在 MLP 内的 RowParallelLinear 不该匹 o_proj."""
    tree = _node("Qwen3Model(...)", children=[
        _node("Qwen3DecoderLayer(...)", children=[
            _node("Qwen3MLP(...)", children=[
                _node("RowParallelLinear(...)", cuda_time_us=20.0, invocations=1),
            ]),
        ]),
    ])
    samples = extract_samples(tree, _SLICE)
    # 应匹 down_proj (within=Qwen3MLP), 不应匹 o_proj (within=Qwen3Attention)
    assert len(samples) == 1
    assert samples[0].layer == "down_proj"


def test_extract_list_of_roots():
    """model_stats 也支持 list 顶层 (多 root)."""
    trees = [
        _node("EmbeddingWrapper(...)", children=[
            _node("VocabParallelEmbedding(...)", cuda_time_us=5.0, invocations=1),
        ]),
        _node("MLPWrapper(...)", children=[
            _node("QKVParallelLinear(...)", cuda_time_us=15.0, invocations=1),
        ]),
    ]
    samples = extract_samples(trees, _SLICE)
    assert len(samples) == 2
    assert {s.layer for s in samples} == {"embedding", "qkv_proj"}


def test_timing_sample_as_dict():
    s = TimingSample(layer="qkv_proj", op_kind="dense_gemm", microseconds=42.0)
    d = s.as_dict()
    assert d == {"layer": "qkv_proj", "op_kind": "dense_gemm", "microseconds": 42.0}


def test_extract_empty_tree():
    assert extract_samples([], _SLICE) == []
    assert extract_samples({"entry": {"name": "X(...)", "cuda_time_us": 0, "invocations": 0},
                            "children": []}, _SLICE) == []
