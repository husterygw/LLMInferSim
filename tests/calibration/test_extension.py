"""calibration/extension.py — LayerwiseProfileExtension.fire() (B.2).

mock 整套 model_runner + layerwise_profile, 验证:
  1. fire() 流程 (warmup + N iter forward + extract)
  2. iterations 参数有效
  3. catalog_slice 透传给 extract_samples
  4. 返回的是 plain dict (pickle-safe)
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from llm_infer_sim.calibration.extension import LayerwiseProfileExtension


class _FakeProfilerResults:
    """模拟 LayerwiseProfileResults."""
    def convert_stats_to_dict(self):
        return {
            "model_stats": [
                {
                    "entry": {
                        "name": "Qwen3Model(...)",
                        "cuda_time_us": 0.0, "cpu_time_us": 0.0, "invocations": 1,
                    },
                    "children": [
                        {
                            "entry": {
                                "name": "QKVParallelLinear(...)",
                                "cuda_time_us": 3000.0, "cpu_time_us": 0.0,
                                "invocations": 30,  # 3 iter × 10 layers
                            },
                            "children": [],
                        },
                    ],
                },
            ],
        }


class _FakeLayerwiseProfile:
    """模拟 vllm.profiler.layerwise_profile context manager."""
    def __init__(self, *args, **kwargs):
        self.results = _FakeProfilerResults()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeBlockTable:
    block_size = 16
    blocks_per_kv_block = 1


class _FakeInputBatch:
    def __init__(self):
        self.block_table = SimpleNamespace(block_tables=[_FakeBlockTable()])


class _FakeModelRunner:
    """mock vLLM v1 GPUModelRunner."""
    def __init__(self):
        self.input_batch = _FakeInputBatch()
        self.execute_calls = 0

    def execute_model(self, scheduler_output):
        self.execute_calls += 1
        return None  # 模拟返回 None (vLLM 某些路径), 触发 sample_tokens 兜底

    def sample_tokens(self, _):
        pass


@pytest.fixture
def patched_vllm(monkeypatch):
    """patch vLLM SamplingParams / SchedulerOutput / layerwise_profile.

    我们的 batch.assemble_scheduler_output 真去 import vllm.v1.core.sched.output. 测试
    跑得动靠 vLLM 0.20.1 已装 (有). 这里只额外 patch layerwise_profile.
    """
    fake_profiler_mod = mock.MagicMock()
    fake_profiler_mod.layerwise_profile = _FakeLayerwiseProfile
    monkeypatch.setitem(sys.modules, "vllm.profiler.layerwise_profile", fake_profiler_mod)


def _has_vllm_v1_output() -> bool:
    try:
        from vllm.v1.core.sched.output import SchedulerOutput  # noqa: F401
        return True
    except ImportError:
        return False


needs_vllm = pytest.mark.skipif(
    not _has_vllm_v1_output(),
    reason="vLLM v1 SchedulerOutput 不可 import",
)


@needs_vllm
def test_fire_returns_dict_list(patched_vllm):
    """fire() 返回 plain dict list (pickle 安全)."""
    ext = LayerwiseProfileExtension()
    ext.model_runner = _FakeModelRunner()

    shot = {"kind": "dense", "num_new_tokens": 128,
            "num_decode_seqs": 0, "kv_lens_prefill": [],
            "kv_lens_decode": [], "prefill_chunk": 0}
    slice_ = {
        "qkv_proj": {"vllm": "QKVParallelLinear", "within": None, "op_kind": "dense_gemm"},
    }
    out = ext.fire(shot, slice_, kind="dense", iterations=3)
    assert isinstance(out, list)
    assert all(isinstance(x, dict) for x in out)
    assert len(out) == 1
    assert out[0]["layer"] == "qkv_proj"
    assert out[0]["op_kind"] == "dense_gemm"
    # vLLM model_stats 树 entry 是 per-call, 不 / invocations: cuda_us 直接用
    assert out[0]["microseconds"] == 3000.0


@needs_vllm
def test_fire_warmup_plus_iterations(patched_vllm):
    """execute_model 被调 1 (warmup) + iterations 次."""
    ext = LayerwiseProfileExtension()
    runner = _FakeModelRunner()
    ext.model_runner = runner

    shot = {"kind": "dense", "num_new_tokens": 64,
            "num_decode_seqs": 0, "kv_lens_prefill": [],
            "kv_lens_decode": [], "prefill_chunk": 0}
    slice_: dict = {}
    ext.fire(shot, slice_, kind="dense", iterations=5)
    assert runner.execute_calls == 1 + 5   # warmup + 5 iter


@needs_vllm
def test_fire_iterations_clamps_to_one(patched_vllm):
    """iterations=0 / 负数 → clamp 到 1."""
    ext = LayerwiseProfileExtension()
    runner = _FakeModelRunner()
    ext.model_runner = runner

    shot = {"kind": "dense", "num_new_tokens": 64,
            "num_decode_seqs": 0, "kv_lens_prefill": [],
            "kv_lens_decode": [], "prefill_chunk": 0}
    ext.fire(shot, {}, kind="dense", iterations=0)
    assert runner.execute_calls == 1 + 1   # warmup + clamped 1


@needs_vllm
def test_fire_passes_slice_to_extract(patched_vllm):
    """slice 中没的 vllm class 应不出 sample."""
    ext = LayerwiseProfileExtension()
    ext.model_runner = _FakeModelRunner()
    # slice 只含 RMSNorm, 但 fake tree 全是 QKVParallelLinear → 不该出 sample
    slice_only_rmsnorm = {
        "layernorm": {"vllm": "RMSNorm", "within": None, "op_kind": "rmsnorm"},
    }
    shot = {"kind": "dense", "num_new_tokens": 64,
            "num_decode_seqs": 0, "kv_lens_prefill": [],
            "kv_lens_decode": [], "prefill_chunk": 0}
    out = ext.fire(shot, slice_only_rmsnorm, kind="dense", iterations=1)
    assert out == []


@needs_vllm
def test_fire_handles_sample_tokens_missing(patched_vllm):
    """model_runner 没 sample_tokens 也不该挂 (sample_tokens 是可选 fallback)."""
    class _RunnerNoSample:
        def __init__(self):
            self.input_batch = _FakeInputBatch()
        def execute_model(self, _):
            return None  # 触发 sample_tokens 兜底路径
        # 故意不实现 sample_tokens

    ext = LayerwiseProfileExtension()
    ext.model_runner = _RunnerNoSample()
    shot = {"kind": "dense", "num_new_tokens": 64,
            "num_decode_seqs": 0, "kv_lens_prefill": [],
            "kv_lens_decode": [], "prefill_chunk": 0}
    out = ext.fire(shot, {}, kind="dense", iterations=1)
    assert isinstance(out, list)
