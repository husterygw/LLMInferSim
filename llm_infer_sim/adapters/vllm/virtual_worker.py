"""VirtualWorker — vLLM v1 WorkerBase 的 CPU-only 虚拟实现。

阶段 0 spike 范围:
  - init_device: gloo PG 真实 multi-process collective ready
  - load_model: config-only, 不实例化 nn.Module / 不分配权重
  - get_kv_cache_spec / determine_available_memory / initialize_from_config:
    构造合法 KV spec + 虚拟 HBM 容量, 不分配真实 tensor
  - execute_model: 委托给 VirtualModelRunner (单独文件, 详 §4.3.4)
  - 进程不 crash, 让 EngineCore.step() 跑通

后续阶段才补:
  - feature gate / hf_config 离线注入 (Qwen 阶段)
  - MLA / sliding window / hybrid KV spec 分支 (DeepSeek / Mistral 阶段)
  - 多 worker 真实落地 (阶段 4 起)
"""
from __future__ import annotations

import sys

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase


def _log(msg: str) -> None:
    """阶段 0: 直接 stderr, 绕开 vllm.init_logger 的命名空间过滤。"""
    print(f"[VirtualWorker] {msg}", file=sys.stderr, flush=True)


class VirtualWorker(WorkerBase):
    """CPU-only 虚拟 worker。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.device = torch.device("cpu")
        self.step_counter = 0
        self._kv_cache_config: KVCacheConfig | None = None
        self._virtual_model_runner = None  # load_model 时实例化

    # ------- lifecycle -------

    def init_device(self) -> None:
        from vllm.distributed import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )

        # 即使 tp=pp=1 也要建 PG, vLLM 内部到处依赖 model_parallel_state
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )
        _log(
            f"device=cpu rank={self.rank}/{self.parallel_config.world_size} gloo PG ready"
        )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Config-only: 不构造 nn.Module, 同时实例化 VirtualModelRunner。"""
        from llm_infer_sim.adapters.vllm.virtual_model_runner import VirtualModelRunner

        self._virtual_model_runner = VirtualModelRunner(self.vllm_config)
        _log(f"config-only load (model={self.model_config.model})")

    def get_model(self) -> torch.nn.Module:
        # 上游某些路径会拿一下 model 实例; 给一个 Identity 占位
        return torch.nn.Identity()

    # ------- KV cache -------

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """返回每层的 KV cache spec (详设 §4.3.3 + 阶段 8-γ MLA 分支)。

        阶段 0-7: 所有模型走 `FullAttentionSpec`(标准 MHA/GQA per-head KV)
        阶段 8-γ 新增: MLA 模型(kv_lora_rank > 0)走 `MLAAttentionSpec`,
                     每 token 只存 (c_kv + qk_rope) bytes,远小于 num_kv_heads × head_dim × 2(K+V)。
                     V3 example: 标准 MHA 用 128×128×2×2=65536 bytes,MLA 用 (512+64)×2=1152,**~57× 小**
        """
        model_cfg = self.model_config
        cache_cfg = self.cache_config
        hf = model_cfg.hf_config

        num_layers = model_cfg.get_num_layers(self.parallel_config)
        block_size = cache_cfg.block_size

        if cache_cfg.cache_dtype == "auto":
            kv_dtype = model_cfg.dtype
        else:
            kv_dtype = getattr(torch, cache_cfg.cache_dtype, torch.float16)

        # 检测 MLA: kv_lora_rank 存在且 > 0
        kv_lora_rank = int(getattr(hf, "kv_lora_rank", 0) or 0)
        is_mla = kv_lora_rank > 0

        if is_mla:
            qk_rope = int(getattr(hf, "qk_rope_head_dim", 0) or 0)
            # MLA: c_kv 在 heads 间共享 (MQA-style), 所以 num_kv_heads=1
            # head_size = c_kv (kv_lora_rank) + qk_rope (rope_k 共享)
            mla_head_size = kv_lora_rank + qk_rope
            spec = {
                f"model.layers.{i}.self_attn.attn": MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,                    # MLA: c_kv shared across heads
                    head_size=mla_head_size,           # = kv_lora_rank + qk_rope
                    dtype=kv_dtype,
                )
                for i in range(num_layers)
            }
            _log(
                f"kv_cache_spec built MLA (layers={num_layers}, block_size={block_size}, "
                f"head_size={mla_head_size}={kv_lora_rank}+{qk_rope}, "
                f"per-token-bytes/layer={mla_head_size * 2}, dtype={kv_dtype})"
            )
        else:
            num_kv_heads = model_cfg.get_num_kv_heads(self.parallel_config)
            head_size = model_cfg.get_head_size()
            spec = {
                f"model.layers.{i}.self_attn.attn": FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=kv_dtype,
                )
                for i in range(num_layers)
            }
            _log(
                f"kv_cache_spec built (layers={num_layers}, block_size={block_size}, "
                f"num_kv_heads={num_kv_heads}, head_size={head_size}, dtype={kv_dtype})"
            )
        return spec

    def determine_available_memory(self) -> int:
        """返回 *KV cache 可用* 字节数 (详设 §4.3.3 阶段 4 真实化)。

        公式 (阶段 4 起):
          available_kv = HBM × gpu_memory_utilization
                       - per_rank_weight_bytes(model, w_byte, tp)
                       - activation_buffer(model, max_num_batched_tokens, a_byte)

        阶段 0-3 旧公式 `× 0.5` 占位已替换。
        阶段 5 (MoE) / 阶段 6 (EP) 起需细化 expert 切分逻辑。
        """
        from llm_infer_sim.adapters.vllm.profile_extractor import extract_profile_bundle
        from llm_infer_sim.core.profiles.sizing import (
            estimate_activation_bytes,
            per_rank_param_bytes,
        )

        bundle = extract_profile_bundle(self.vllm_config)
        model = bundle.model
        deploy = bundle.deploy
        hw = bundle.hw

        hbm = int(hw.mem_capacity_gb * 1024 * 1024 * 1024)
        utilization = self.cache_config.gpu_memory_utilization
        budget = int(hbm * utilization)

        tp = deploy.tp
        ep = deploy.ep
        # dtype-aware + EP-aware: routed expert 按 ep 切 + 用 expert_w_byte (V4 fp4=0.5).
        # expert_w_byte 在 per_rank_param_bytes 内部从 model.expert_fp4 推断.
        weight_bytes = per_rank_param_bytes(model, deploy.w_byte, tp, ep_size=ep)
        max_batched = getattr(self.scheduler_config, "max_num_batched_tokens", 2048)
        activation_bytes = estimate_activation_bytes(model, max_batched, deploy.a_byte)

        available = budget - weight_bytes - activation_bytes
        if available <= 0:
            _log(
                f"[WARN] estimated weights({weight_bytes/1e9:.1f}GB) + "
                f"activations({activation_bytes/1e9:.2f}GB) > budget({budget/1e9:.1f}GB);"
                f" falling back to 10% of HBM for KV"
            )
            available = max(int(hbm * 0.1), 1)

        _log(
            f"virtual available_memory={available/1e9:.1f} GB "
            f"(hbm={hw.mem_capacity_gb:.0f}GB util={utilization:.2f} "
            f"weights/rank={weight_bytes/1e9:.1f}GB tp={tp} "
            f"activation={activation_bytes/1e9:.2f}GB)"
        )
        return available

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # 不分配真实 tensor, 只把 num_blocks 喂给 cache_config
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self._kv_cache_config = kv_cache_config
        _log(
            f"kv_cache_config accepted (num_blocks={kv_cache_config.num_blocks}, no tensor allocated)"
        )

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_supported_tasks(self) -> tuple[str, ...]:
        # vLLM 0.20 通过 collective_rpc 询问支持的 task 类型
        return ("generate",)

    # ------- step execution -------

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        """委托给 VirtualModelRunner.execute_step (详 §4.3.4)。"""
        self.step_counter += 1
        if self._virtual_model_runner is None:
            # 防御: load_model 应该已经实例化, 但若被绕过则 lazy 兜底
            from llm_infer_sim.adapters.vllm.virtual_model_runner import VirtualModelRunner
            self._virtual_model_runner = VirtualModelRunner(self.vllm_config)
        return self._virtual_model_runner.execute_step(
            scheduler_output, self.step_counter, rank=self.rank,
        )

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        raise NotImplementedError(
            "VirtualWorker performs sampling inside execute_model"
        )

    def execute_dummy_batch(self) -> None:
        """vLLM v1 DP idle rank 同步用 dummy batch (详 vllm/v1/engine/core.py:1758).

        当 DP 集群里某 rank 没活但其他 rank 有时, idle rank 会调这个 keep gloo
        collective 同步。我们 cost model 不模拟 dummy batch (它在真实 GPU 上是空跑
        kernel + 跨 rank allreduce, 时间近 0), 直接 noop。
        """
        return None

    # ------- 阶段 3 D 块: 报告抽取 (供 collective_rpc 调用) -------

    def _get_virtual_runner_report(self) -> str:
        """examples/run_qwen3_4b_chunked.py 通过 collective_rpc 拉取每 rank 报告。"""
        if self._virtual_model_runner is None:
            return "(runner not initialized — no steps executed)"
        return self._virtual_model_runner.get_report().generate_console_report()

    def _get_pd_stats(self) -> dict:
        """examples/run_pd_disagg_loopback.py 拉 PD 累计传输统计."""
        if self._virtual_model_runner is None:
            return {"pd_enabled": False}
        return self._virtual_model_runner.get_pd_stats()

    # ------- PD 分离 stub (详设 §7.6) -------

    def get_kv_connector_handshake_metadata(self):
        """vLLM v1 在 engine init 时通过 collective_rpc 拉 connector metadata.
        我们不跑真 connector, 返回 None 让 engine 当作 no-op。"""
        return None

    def has_kv_connector(self) -> bool:
        """配套 stub: 上游某些路径会查"我有 connector 吗"."""
        return False

    def _get_per_request_metrics(self) -> list[dict]:
        """examples/run_prefix_caching.py 用: 拉每 request 的 ttft / arrival_time /
        first_token_time 等 sim-time 指标, 供 batch 间对比验证 prefix caching 命中。"""
        if self._virtual_model_runner is None:
            return []
        collector = self._virtual_model_runner.metrics
        out: list[dict] = []
        for rm in collector.requests.values():
            out.append({
                "request_id": rm.request_id,
                "arrival_time": rm.arrival_time,
                "first_token_time": rm.first_token_time,
                "completion_time": rm.completion_time,
                "ttft": rm.ttft,
                "tpot": rm.tpot,
                "e2e_latency": rm.e2e_latency,
                "output_tokens": rm.output_tokens,
                "completed": rm.completed,
                "num_steps_scheduled": len(rm.per_token_latencies),
            })
        return out
