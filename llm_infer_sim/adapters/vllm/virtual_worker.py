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
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
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
        model_cfg = self.model_config
        cache_cfg = self.cache_config

        num_layers = model_cfg.get_num_layers(self.parallel_config)
        num_kv_heads = model_cfg.get_num_kv_heads(self.parallel_config)
        head_size = model_cfg.get_head_size()

        if cache_cfg.cache_dtype == "auto":
            kv_dtype = model_cfg.dtype
        else:
            kv_dtype = getattr(torch, cache_cfg.cache_dtype, torch.float16)

        block_size = cache_cfg.block_size
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
            f"kv_cache_spec built (layers={num_layers}, block_size={block_size}, dtype={kv_dtype})"
        )
        return spec

    def determine_available_memory(self) -> int:
        """返回 *KV cache 可用* 字节数 (粗估)。

        阶段 0/1/2: 从 hardware profile 的 mem_capacity_gb 读 HBM 容量
        (env LLM_INFER_SIM_HW 选定, 默认 H100=80GB), 留一半给权重 + 激活粗估。
        阶段 4 起: 改为真实模型权重 + 激活估算 (详设 §4.3.3)。
        """
        import os
        from llm_infer_sim.core.profiles.hardware import get_hardware_profile

        hw_name = os.environ.get("LLM_INFER_SIM_HW", "H100")
        hw = get_hardware_profile(hw_name)
        hbm = int(hw.mem_capacity_gb * 1024 * 1024 * 1024)
        utilization = self.cache_config.gpu_memory_utilization
        # todo: 权重后续阶段补上后, 这里的 0.5 可以改成 config 注入的预留比例, 让用户模拟不同程度的内存压力
        available = int(hbm * utilization * 0.5)  # 留一半给权重 + 激活粗估
        _log(
            f"virtual available_memory={available/1e9:.1f} GB "
            f"(hw={hw_name} hbm={hw.mem_capacity_gb:.0f}GB utilization={utilization:.2f})"
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

    # ------- 阶段 3 D 块: 报告抽取 (供 collective_rpc 调用) -------

    def _get_virtual_runner_report(self) -> str:
        """examples/run_qwen3_4b_chunked.py 通过 collective_rpc 拉取每 rank 报告。"""
        if self._virtual_model_runner is None:
            return "(runner not initialized — no steps executed)"
        return self._virtual_model_runner.get_report().generate_console_report()
