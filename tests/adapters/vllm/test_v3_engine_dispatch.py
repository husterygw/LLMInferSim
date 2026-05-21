"""Step 4 foundation: 验证 profile_extractor 同时输出 LegacyDeployConfig + V3 DeployConfig.

V3 DeployConfig 字段从 vllm_config.parallel_config / cache_config / scheduler_config 推.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.profiles.deploy import DeployConfig, LegacyDeployConfig


def _qwen3_4b_vllm_config(tp_size: int = 1, dp_size: int = 1, enable_ep: bool = False):
    hf = SimpleNamespace(
        model_type="qwen3",
        num_attention_heads=32, num_key_value_heads=8,
        hidden_size=2560, num_hidden_layers=36,
        intermediate_size=9728, vocab_size=151936, head_dim=128,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-4B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            enable_expert_parallel=enable_ep,
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_seqs=64,
        ),
    )


def test_extract_produces_both_legacy_and_v3_deploy():
    """bundle.deploy = LegacyDeployConfig (旧), bundle.deploy_v3 = V3 DeployConfig (新)."""
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    assert isinstance(bundle.deploy, LegacyDeployConfig)
    assert bundle.deploy_v3 is not None
    assert isinstance(bundle.deploy_v3, DeployConfig)


def test_v3_deploy_tp_dp_ep_match_legacy():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config(tp_size=4, dp_size=2, enable_ep=True))
    d_legacy = bundle.deploy
    d_v3 = bundle.deploy_v3
    assert d_v3.tp_size == d_legacy.parallel.tp_size == 4
    assert d_v3.dp_size == d_legacy.parallel.dp_size == 2
    # vLLM ep = tp * dp when enable_ep
    assert d_v3.ep_size == d_legacy.parallel.ep_size == 8


def test_v3_deploy_scheduler_block_fields():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config())
    d = bundle.deploy_v3
    assert d.block_size == 16
    assert d.max_num_batched_tokens == 8192
    assert d.max_num_seqs == 64
    assert d.execution_mode == "cudagraph"   # vLLM v1 default
    assert d.backend == "vllm"


def test_v3_deploy_no_expert_parallel_means_ep1():
    bundle = extract_profile_bundle(_qwen3_4b_vllm_config(tp_size=2, dp_size=1, enable_ep=False))
    assert bundle.deploy_v3.ep_size == 1
