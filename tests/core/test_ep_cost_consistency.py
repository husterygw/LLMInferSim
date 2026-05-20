"""阶段 6-γ: EP (expert parallel) cost 公式数字一致性 (详设 §4.1.3 + §4.7.4)。

固化以下关键正确性 (手算 vs layer_builder 实际输出):
  1. ep>1 时 expert_dim_per_device = expert_dim (不切 TP), 跟 ep=1 时切 tp 反着
  2. routed_experts.flops = tokens × top_k × 3 × 2 × h × expert_dim // ep
  3. routed_experts.mem_bytes:
        weight = distinct(T,k,N) × 3 × h × expert_dim × w_byte / ep
        act    = 2 × (T × top_k // ep) × h × a_byte
  4. ep_alltoall_dispatch + ep_alltoall_combine 都注入, 每个 comm_bytes = T × h × a_byte
  5. routed_expert_allreduce 在 ep>1 时不再插 (被 AllToAll 替代)
  6. attn_allreduce 仍在 (跟 EP 无关, 由 TP 决定)

按记忆 feedback_cost_formula_handcheck.md 必做。
"""
from types import SimpleNamespace

import pytest

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost_model.layer_builder import moe_layer_time


@pytest.fixture
def qwen3_30b_a3b_ep2_bundle():
    """Qwen3-30B-A3B + tp=2 + enable_expert_parallel=True 的 bundle。"""
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32, num_key_value_heads=4,
        hidden_size=2048, num_hidden_layers=48,
        intermediate_size=6144, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, mlp_only_layers=[],
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-30B-A3B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, data_parallel_size=1,
            enable_expert_parallel=True,
        ),
    )
    return extract_profile_bundle(vc)


def _find_op(lr, name):
    for op in lr.ops:
        if op.name == name:
            return op
    raise AssertionError(f"op {name!r} not found in {[o.name for o in lr.ops]}")


def _has_op(lr, name) -> bool:
    return any(op.name == name for op in lr.ops)


# ------- enable_ep extraction -------

def test_profile_extractor_reads_enable_expert_parallel(qwen3_30b_a3b_ep2_bundle):
    """profile_extractor 应该把 vllm_config.parallel_config.enable_expert_parallel
    透传到 ParallelConfig.enable_ep / LegacyDeployConfig.ep。"""
    deploy = qwen3_30b_a3b_ep2_bundle.deploy
    assert deploy.parallel.enable_ep is True
    assert deploy.ep == 2   # ep = tp × dp = 2 × 1 = 2


# ------- routed_experts under EP -------

def test_routed_experts_expert_dim_not_sliced_under_ep(qwen3_30b_a3b_ep2_bundle):
    """ep>1 时 expert_dim_per_device = expert_dim (不切 tp), 跟 ep=1 时反着。

    layer_builder.py: `expert_dim_per_device = model.expert_dim // tp if ep == 1
                       else model.expert_dim`
    """
    b = qwen3_30b_a3b_ep2_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    tokens = 4
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, hw)
    op = _find_op(lr, "routed_experts")

    expert_dim_per_device = m.expert_dim   # NOT // tp
    expected_flops = (
        tokens * m.num_activated_experts * 3 * 2 * m.hidden_dim
        * expert_dim_per_device // deploy.ep
    )
    assert op.flops == expected_flops


def test_routed_experts_weight_uses_distinct_div_ep(qwen3_30b_a3b_ep2_bundle):
    """EP 下 weight read = distinct × per-expert / ep。"""
    from llm_infer_sim.core.cost_model.moe_routing import estimate_distinct_experts
    b = qwen3_30b_a3b_ep2_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    expert_dim_per_device = m.expert_dim   # ep > 1: 不切 tp

    for tokens in (4, 128):
        lr = moe_layer_time(0, "prefill", tokens, 128, m, deploy, hw)
        op = _find_op(lr, "routed_experts")
        distinct = estimate_distinct_experts(
            tokens, m.num_activated_experts, m.num_experts, skew=0.0,
        )
        expected_weight = int(
            distinct * 3 * m.hidden_dim * expert_dim_per_device
            * deploy.w_byte / deploy.ep
        )
        tokens_per_device = tokens * m.num_activated_experts // deploy.ep
        expected_act = 2 * tokens_per_device * m.hidden_dim * deploy.a_byte
        assert op.mem_bytes == expected_weight + expected_act, (
            f"tokens={tokens}: got {op.mem_bytes}, "
            f"expected {expected_weight + expected_act} "
            f"(distinct={distinct:.2f})"
        )


def test_routed_experts_act_scales_with_tokens_per_device(qwen3_30b_a3b_ep2_bundle):
    """EP 下 act_in = act_out = (tokens × top_k // ep) × h × a_byte。

    AllToAll dispatch 后, 每 rank 收到 tokens × top_k / ep 个 token 副本。
    """
    b = qwen3_30b_a3b_ep2_bundle
    m, deploy, hw = b.model, b.deploy, b.hw
    tokens = 4
    lr = moe_layer_time(0, "decode", tokens, 128, m, deploy, hw)
    op = _find_op(lr, "routed_experts")

    tokens_per_device = tokens * m.num_activated_experts // deploy.ep
    expected_act_each = tokens_per_device * m.hidden_dim * deploy.a_byte
    assert op.load_act == expected_act_each
    assert op.store_act == expected_act_each


# ------- AllToAll comm injection -------

def test_ep_alltoall_dispatch_combine_both_present(qwen3_30b_a3b_ep2_bundle):
    """ep>1 时必须同时插入 dispatch + combine 两个 AllToAll ops。"""
    b = qwen3_30b_a3b_ep2_bundle
    lr = moe_layer_time(0, "decode", tokens=4, ctx_len=128,
                        model=b.model, deploy=b.deploy, hw=b.hw)
    assert _has_op(lr, "ep_alltoall_dispatch")
    assert _has_op(lr, "ep_alltoall_combine")


def test_ep_alltoall_comm_bytes(qwen3_30b_a3b_ep2_bundle):
    """AllToAll comm_bytes = tokens × h × a_byte (per rank 发/收量)。"""
    b = qwen3_30b_a3b_ep2_bundle
    m, deploy = b.model, b.deploy

    for tokens in (4, 128):
        lr = moe_layer_time(0, "prefill" if tokens > 1 else "decode",
                            tokens, 128, m, deploy, b.hw)
        for op_name in ("ep_alltoall_dispatch", "ep_alltoall_combine"):
            op = _find_op(lr, op_name)
            expected = tokens * m.hidden_dim * deploy.a_byte
            assert op.comm_bytes == expected, (
                f"tokens={tokens} op={op_name}: got {op.comm_bytes}, expected {expected}"
            )
            assert op.comm_type == "alltoall"


# ------- routed_expert_allreduce gone under EP -------

def test_routed_expert_allreduce_absent_under_ep(qwen3_30b_a3b_ep2_bundle):
    """ep>1 时 routed_expert_allreduce 被 AllToAll 替代, 不应再出现。"""
    b = qwen3_30b_a3b_ep2_bundle
    lr = moe_layer_time(0, "decode", tokens=4, ctx_len=128,
                        model=b.model, deploy=b.deploy, hw=b.hw)
    assert not _has_op(lr, "routed_expert_allreduce")


def test_attn_allreduce_still_present_under_ep(qwen3_30b_a3b_ep2_bundle):
    """attn_allreduce 跟 EP 无关, 只看 TP, ep>1 不影响 attention 的 allreduce 注入。"""
    b = qwen3_30b_a3b_ep2_bundle
    assert b.deploy.tp == 2
    lr = moe_layer_time(0, "decode", tokens=4, ctx_len=128,
                        model=b.model, deploy=b.deploy, hw=b.hw)
    assert _has_op(lr, "attn_allreduce")


# ------- ep=1 vs ep>1 切换的反向防御 -------

def test_ep1_uses_allreduce_ep2_uses_alltoall():
    """同 model 在 ep=1 / ep=2 之间切换, 通信 op 形态正确切换。"""
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32, num_key_value_heads=4,
        hidden_size=2048, num_hidden_layers=48,
        intermediate_size=6144, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, mlp_only_layers=[],
    )

    def _bundle_with(enable_ep: bool):
        vc = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf, model="x"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2, data_parallel_size=1,
                enable_expert_parallel=enable_ep,
            ),
        )
        return extract_profile_bundle(vc)

    b1 = _bundle_with(False)  # ep=1
    b2 = _bundle_with(True)   # ep=2

    lr1 = moe_layer_time(0, "decode", 4, 128, b1.model, b1.deploy, b1.hw)
    lr2 = moe_layer_time(0, "decode", 4, 128, b2.model, b2.deploy, b2.hw)

    # ep=1: routed_expert_allreduce 在, AllToAll 不在
    assert _has_op(lr1, "routed_expert_allreduce")
    assert not _has_op(lr1, "ep_alltoall_dispatch")
    assert not _has_op(lr1, "ep_alltoall_combine")
    # ep=2: 反过来
    assert not _has_op(lr2, "routed_expert_allreduce")
    assert _has_op(lr2, "ep_alltoall_dispatch")
    assert _has_op(lr2, "ep_alltoall_combine")


def test_dp_doubles_ep_world_passes_to_alltoall_time(monkeypatch):
    """DP+EP 场景: tp=2 dp=1 ep=2  vs  tp=2 dp=2 ep=4. 验证 cost 路径里
    `alltoall_time(bytes, n, hw)` 的 n 严格等于 ep_world = tp × dp, 不是 tp。

    注: AllToAll cost 与 n 的关系并非单调 (大数据下 n² 项让带宽分摊更优, 总时间可
    能反而降), 所以验证"参数值"而非"输出大小"。
    """
    import llm_infer_sim.core.cost_model.layer_builder as lb
    captured: list[int] = []
    original = lb.alltoall_time

    def spy(bytes_, n, hw, **kw):
        captured.append(n)
        return original(bytes_, n, hw, **kw)

    monkeypatch.setattr(lb, "alltoall_time", spy)

    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=32, num_key_value_heads=4,
        hidden_size=2048, num_hidden_layers=48,
        intermediate_size=6144, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=768, mlp_only_layers=[],
    )

    def _bundle(tp, dp):
        vc = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf, model="x"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp, data_parallel_size=dp,
                enable_expert_parallel=True,
            ),
        )
        return extract_profile_bundle(vc)

    b_small = _bundle(tp=2, dp=1)  # ep_world=2
    b_big = _bundle(tp=2, dp=2)    # ep_world=4
    assert b_small.deploy.ep == 2 and b_big.deploy.ep == 4

    captured.clear()
    lr1 = moe_layer_time(0, "decode", 4, 128, b_small.model, b_small.deploy, b_small.hw)
    lb._compute_layer_time(lr1.ops, b_small.hw, b_small.deploy)
    n_used_small = set(captured)
    assert n_used_small == {2}, f"ep_world=2 应调 alltoall_time(n=2), 实际 {n_used_small}"

    captured.clear()
    lr2 = moe_layer_time(0, "decode", 4, 128, b_big.model, b_big.deploy, b_big.hw)
    lb._compute_layer_time(lr2.ops, b_big.hw, b_big.deploy)
    n_used_big = set(captured)
    assert n_used_big == {4}, f"ep_world=4 (=tp×dp) 应调 alltoall_time(n=4), 实际 {n_used_big}"
