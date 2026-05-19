"""vLLM attention runner — 测真实 FLASH_ATTN/FLASHINFER kernel latency.

直接调 `impl.forward(layer, q, k_new, v_new, kv_cache, attn_metadata, output=...)`,
跟 vLLM step 里跑的是同一条 path. kernel_source 取决于 backend 选择
(RTX 4090 上一般 "FLASH_ATTN").

case.params 必含: phase, batch_size, isl, kv_prefill, n_decode, kv_decode,
                  num_heads, num_kv_heads, head_dim, dtype, tp, execution_mode

第一版 BF16, kv_cache_dtype=auto (BF16), tp=1, 无 sliding window.
"""
from __future__ import annotations

from typing import Optional

from collector.harness import BenchConfig, BenchResult, measure
from collector.runners._vllm_attn import (
    BatchSpec,
    MockAttentionLayer,
    create_and_prepopulate_kv_cache,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    resolve_backend,
)
from collector.runners._vllm_dist import torch_dtype
from collector.schemas import (
    Case,
    ExecutionMode,
    Framework,
    Metrics,
    OpKind,
    RawRecord,
)


def build_record(
    case: Case,
    bench: BenchResult,
    *,
    framework_version: str,
    device_name: str,
    kernel_source: str,
) -> RawRecord:
    return RawRecord(
        case_id=case.case_id,
        op_kind=OpKind.ATTENTION,
        framework=Framework.VLLM,
        framework_version=framework_version,
        device=device_name,
        execution_mode=(
            ExecutionMode.CUDAGRAPH if bench.used_cuda_graph else ExecutionMode.EAGER
        ),
        kernel_source=kernel_source,
        params=dict(case.params),
        metrics=Metrics(
            latency_us_p50=bench.latency_us_p50,
            latency_us_p10=bench.latency_us_p10,
            latency_us_p90=bench.latency_us_p90,
            used_cuda_graph=bench.used_cuda_graph,
            n_warmups=bench.n_warmups,
            n_iters=bench.n_iters,
        ),
        metadata={"fallback_reason": bench.fallback_reason},
    )


def _build_batch_spec(p: dict) -> BatchSpec:
    """case.params → BatchSpec.

    prefill: 1 个 seq (batch_size 一般 = 1), seq_len=isl, query_len=isl, ctx_len=0
             (全段 prefill, KV cache 空; 不模拟 chunked).
    decode:  n_decode 个 seq, seq_len = kv_decode, query_len = 1, ctx_len = kv_decode-1
    """
    phase = p["phase"]
    if phase == "prefill":
        bs = int(p["batch_size"])
        isl = int(p["isl"])
        # 假设 kv_prefill=0 → 整段做 prefill
        return BatchSpec(seq_lens=[isl] * bs, query_lens=[isl] * bs)
    if phase == "decode":
        n = int(p["n_decode"])
        kv_decode = int(p["kv_decode"])
        return BatchSpec(seq_lens=[kv_decode] * n, query_lens=[1] * n)
    raise ValueError(f"unknown phase: {phase!r}")


def run_case(case: Case, device: int) -> RawRecord:
    p = case.params
    if p.get("tp", 1) != 1:
        raise NotImplementedError(
            f"vllm_attention runner: tp={p['tp']} 需 distributed runner, 本 runner 只 tp=1"
        )
    if p["dtype"] != "bf16":
        raise NotImplementedError(f"dtype={p['dtype']} not supported (BF16 only)")

    import torch
    import vllm
    from vllm.config import set_current_vllm_config

    dtype = torch_dtype(p["dtype"])
    device_str = f"cuda:{device}"
    torch.cuda.set_device(device)
    dev = torch.device(device_str)

    num_heads = int(p["num_heads"])
    num_kv_heads = int(p["num_kv_heads"])
    head_dim = int(p["head_dim"])
    block_size = 16

    batch_spec = _build_batch_spec(p)
    max_seq_len = max(batch_spec.seq_lens)
    bs = batch_spec.batch_size

    vllm_config = create_vllm_config(
        max_model_len=max(max_seq_len, 256),
        block_size=block_size,
        num_gpu_blocks=8192,
        max_num_seqs=max(bs, 64),
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    with set_current_vllm_config(vllm_config):
        backend_name, builder_cls, impl_cls = resolve_backend(
            head_dim, dtype, block_size,
        )

        # build Q / K_new / V_new and per-seq context K/V (used to pre-populate paged KV cache)
        all_q, all_k_new, all_v_new = [], [], []
        k_ctxs, v_ctxs = [], []
        for i in range(bs):
            s_len = batch_spec.seq_lens[i]
            q_len = batch_spec.query_lens[i]
            ctx_len = s_len - q_len
            q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=dev)
            k_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=dev)
            v_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=dev)
            all_q.append(q)
            all_k_new.append(k_full[ctx_len:])
            all_v_new.append(v_full[ctx_len:])
            k_ctxs.append(k_full[:ctx_len])
            v_ctxs.append(v_full[:ctx_len])

        query = torch.cat(all_q, dim=0)
        key = torch.cat(all_k_new, dim=0)
        value = torch.cat(all_v_new, dim=0)

        common_meta = create_common_attn_metadata(batch_spec, block_size, dev)
        kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
        required_blocks = 1 + sum(
            (s + block_size - 1) // block_size for s in batch_spec.seq_lens
        )
        num_blocks = max(vllm_config.cache_config.num_gpu_blocks or 0, required_blocks)

        kv_cache = create_and_prepopulate_kv_cache(
            k_contexts=k_ctxs, v_contexts=v_ctxs,
            block_size=block_size, num_kv_heads=num_kv_heads,
            head_size=head_dim, dtype=dtype, device=dev,
            num_blocks=num_blocks, common_attn_metadata=common_meta,
        )

        builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, dev)
        attn_metadata = builder.build(
            common_prefix_len=0, common_attn_metadata=common_meta,
        )

        scale = 1.0 / (head_dim ** 0.5)
        impl = impl_cls(
            num_heads=num_heads, head_size=head_dim, scale=scale,
            num_kv_heads=num_kv_heads, alibi_slopes=None,
            sliding_window=None, kv_cache_dtype="auto",
        )

        mock_layer = MockAttentionLayer(dev)
        output = torch.empty_like(query)

        def kernel_func() -> None:
            impl.forward(
                mock_layer, query, key, value, kv_cache, attn_metadata,
                output=output,
            )

        # dry run
        kernel_func()

        mode = p.get("execution_mode", "cudagraph")
        if mode == "eager":
            cfg = BenchConfig(n_warmups=3, n_iters=10, use_cuda_graph=False)
        elif mode == "cudagraph":
            cfg = BenchConfig(n_warmups=3, n_iters=10,
                              use_cuda_graph=True, allow_graph_fail=False)
        else:
            raise NotImplementedError(f"execution_mode {mode!r}")
        bench = measure(kernel_func, cfg)

    device_name = torch.cuda.get_device_name(device)
    return build_record(
        case, bench,
        framework_version=str(vllm.__version__),
        device_name=device_name,
        kernel_source=f"vllm_{backend_name}".lower(),
    )
