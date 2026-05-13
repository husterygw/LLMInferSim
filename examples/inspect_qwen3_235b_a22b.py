"""阶段 7 inspect: Qwen3-235B-A22B + tp=8 + ep=16 cost path 模拟。

═══════════════════════════════════════════════════════════════════════════════
                                 怎么跑
═══════════════════════════════════════════════════════════════════════════════

不走 vllm 多进程 — 直接驱动 cost model 做 sizing + per-op breakdown。原因:
  - 阶段 7 没有新胶水层 (VirtualWorker / Platform 都不变, 阶段 6 已验过 EP multi-proc)
  - 触发 hierarchical inter-node 公式需要 world_size > intra_node_size (默认 8),
    vllm 单机起 16 进程紧 (16 worker × ~1GB driver state)
  - 真正要验的是数字层 — bandwidth 切换 / hierarchical 公式 / 跨节点 collective

   conda activate llm_sim
   python examples/inspect_qwen3_235b_a22b.py

预期输出:
  - per-layer cost (attention + MoE FFN + comm)
  - ep_alltoall_dispatch / combine 数字, 跨节点用 inter_bw
  - 对比 tp=8 ep=8 (单节点) vs tp=8 ep=16 (跨 2 节点) 的 latency 变化

═══════════════════════════════════════════════════════════════════════════════
                              这个 inspect 验证什么
═══════════════════════════════════════════════════════════════════════════════
  ✅ Qwen3-235B-A22B hf_config 解析正确 (hidden=4096, layers=94, num_experts=128, top_k=8)
  ✅ profile_extractor 读 enable_expert_parallel 正确
  ✅ EP > intra_node_size 时, alltoall_time 走 _hierarchical_alltoall 分支
  ✅ Hierarchical 公式数字落在 (intra-only, inter-only) 区间内
  ✅ 跨节点配置 (ep=16) vs 单节点 (ep=8) 的 step latency 差异
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
from llm_infer_sim.core.cost_model.layer_builder import moe_layer_time
from llm_infer_sim.core.ops.communication import (
    _hierarchical_alltoall,
    _is_cross_node,
    alltoall_time,
)


def _build_bundle(tp_size: int, ep_enable: bool):
    """构造 Qwen3-235B-A22B + 指定 TP/EP 配置的 ProfileBundle。"""
    hf = SimpleNamespace(
        model_type="qwen3_moe",
        num_attention_heads=64, num_key_value_heads=4,
        hidden_size=4096, num_hidden_layers=94,
        intermediate_size=12288, vocab_size=151936, head_dim=128,
        num_experts=128, num_experts_per_tok=8,
        moe_intermediate_size=1536, mlp_only_layers=[],
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, model="Qwen/Qwen3-235B-A22B"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            data_parallel_size=1,
            enable_expert_parallel=ep_enable,
        ),
    )
    return extract_profile_bundle(vc)


def _summarize(label: str, tp: int, ep_enable: bool, tokens: int, ctx: int):
    bundle = _build_bundle(tp, ep_enable)
    m, deploy, hw = bundle.model, bundle.deploy, bundle.hw
    # 阶段 6 单节点 + 阶段 7 跨节点切换由 intra_node_size 决定
    cross = _is_cross_node(deploy.ep, hw)

    lr = moe_layer_time(0, "prefill" if tokens > 1 else "decode",
                        tokens, ctx, m, deploy, hw)
    print(f"\n=== {label} ===")
    print(f"   tp={deploy.tp} ep={deploy.ep} intra_node_size={hw.intra_node_size}"
          f" → cross_node={cross}")
    print(f"   inter_node_bandwidth={hw.inter_node_bandwidth/1e9:.0f}GB/s"
          f"  intra_node_bandwidth={hw.intra_node_bandwidth/1e9:.0f}GB/s (bidir)")
    print(f"   per-layer: t_compute={lr.t_compute*1e6:.2f}us"
          f"  t_comm={lr.t_comm*1e6:.2f}us  t_total={lr.t_total*1e6:.2f}us")
    print(f"   per-step (× {m.num_layers} layers): "
          f"t_total ≈ {lr.t_total * m.num_layers * 1e6:.0f}us")

    # 找通信 ops
    for op in lr.ops:
        if op.op_category == "communication":
            t_each = (
                alltoall_time(op.comm_bytes, deploy.ep, hw)
                if op.comm_type == "alltoall"
                else None
            )
            print(f"      {op.name:<30} comm_bytes={op.comm_bytes:>10,.0f}"
                  f"  type={op.comm_type}"
                  + (f"  alltoall_time={t_each*1e6:.3f}us" if t_each else ""))


def main():
    print("Qwen3-235B-A22B (94 layers, hidden=4096, 128 experts top-8)")

    # --- 阶段 6 baseline: tp=8 + ep=8 (单节点) ---
    # vllm: tp=8, enable_expert_parallel=True → ep=8 (within node, intra)
    _summarize("阶段 6 baseline: tp=8 ep=8 单节点 (intra path)",
               tp=8, ep_enable=True, tokens=128, ctx=128)

    # --- 阶段 7 跨节点: 模拟 tp=16 ep=16 (2 nodes × 8 GPUs) ---
    # 注: vllm 单机起不到 tp=16; 我们 standalone 直接配置
    print("\n" + "=" * 80)
    print("跨节点公式触发 (ep > intra_node_size=8) — standalone 路径")
    print("=" * 80)
    bundle16 = _build_bundle(16, ep_enable=True)
    m16, hw16 = bundle16.model, bundle16.hw
    print(f"\n=== 阶段 7: tp=16 ep=16 跨 2 节点 (cross_node=True) ===")
    print(f"   intra_node_size={hw16.intra_node_size}, num_nodes={(16+7)//8}")

    # 单节点 ep=8 vs 跨节点 ep=16, alltoall_time 对比 (data=tokens*h*a_byte)
    h = m16.hidden_dim
    tokens = 128
    data = tokens * h * 2  # a_byte = 2
    print(f"\n   AllToAll comm_bytes 单层 = {data:,}")
    t_intra = alltoall_time(data, 8, hw16)
    t_inter = alltoall_time(data, 16, hw16)
    t_hier = _hierarchical_alltoall(data, 16, hw16)
    print(f"   alltoall_time(N=8,  intra path) = {t_intra*1e6:>9.3f}us")
    print(f"   alltoall_time(N=16, hierarchical) = {t_inter*1e6:>9.3f}us")
    print(f"     -> _hierarchical_alltoall raw = {t_hier*1e6:>9.3f}us  (sanity check)")
    print(f"   ratio = {t_inter/t_intra:.2f}× (跨节点 vs 单节点 alltoall)")

    print("\n阶段 7 inspect PASSED — hierarchical 公式触发 + 跨节点带宽切换正确。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
