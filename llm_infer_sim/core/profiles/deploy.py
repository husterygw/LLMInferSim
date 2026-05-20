"""Deployment and parallelism configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Literal


# PD 分离 connector → 默认带宽 preset (GB/s + 起始 latency us).
# 名字与 vLLM 0.20.1 KVConnectorFactory.register_connector 完全一致.
# - P2pNcclConnector: 同集群 GPU NCCL P2P, 走 IB/NVLink, ~25-50 GB/s
# - LMCacheConnectorV1 / LMCacheMPConnector: 走 RDMA 经 CPU, ~12 GB/s
# - MooncakeConnector: RDMA 直传, ~25 GB/s
# - NixlConnector: NVIDIA NIXL 新方案, 类 NCCL
# - OffloadingConnector / SimpleCPUOffloadConnector: 走 CPU mem, ~8 GB/s (host RAM)
# - HF3FSKVConnector: HuggingFace 3FS 分布式 fs
# 实测值可在 hardware preset 里 override; 这里仅作 fallback。
PD_CONNECTOR_PRESETS: dict[str, tuple[float, float]] = {
    # connector_name → (bandwidth_GBps, startup_latency_us)
    "P2pNcclConnector":          (25.0, 5.0),
    "LMCacheConnectorV1":        (12.0, 100.0),
    "LMCacheMPConnector":        (12.0, 100.0),
    "MooncakeConnector":         (25.0, 30.0),
    "NixlConnector":             (25.0, 5.0),
    "OffloadingConnector":       (8.0, 50.0),
    "SimpleCPUOffloadConnector": (8.0, 50.0),
    "MultiConnector":            (15.0, 50.0),  # 复合, 走最慢通道近似
    "FlexKVConnectorV1":         (15.0, 50.0),
    "MoRIIOConnector":           (15.0, 50.0),
    "HF3FSKVConnector":          (3.0, 1000.0),  # 分布式 fs, 慢但容量大
}


@dataclass
class PDDisaggConfig:
    """Prefill-Decode 分离配置 (详设 §7.6).

    None / role==None 表示未启用 PD 分离。启用时:
      - role=kv_producer: 当前进程是 prefill node, 在 prefill 完成时 send KV
      - role=kv_consumer: 当前进程是 decode node, 在首次 decode 之前等 recv KV
      - role=kv_both: 兼任 producer + consumer (同进程跑 prefill+decode)

    connector_bandwidth_gbps / connector_latency_us 若设, 优先于 hardware
    preset; 否则按 connector_name 从 PD_CONNECTOR_PRESETS 取。
    """
    role: Literal["kv_producer", "kv_consumer", "kv_both"] | None = None
    connector_name: str | None = None     # 如 "P2pNcclConnector"
    kv_parallel_size: int = 1
    connector_bandwidth_gbps: float | None = None
    connector_latency_us: float | None = None

    @property
    def enabled(self) -> bool:
        return self.role is not None

    @property
    def is_producer(self) -> bool:
        return self.role in ("kv_producer", "kv_both")

    @property
    def is_consumer(self) -> bool:
        return self.role in ("kv_consumer", "kv_both")

    def resolve_bandwidth(self) -> float:
        """resolve GB/s, 显式 > preset > fallback 10 GB/s."""
        if self.connector_bandwidth_gbps is not None:
            return self.connector_bandwidth_gbps
        if self.connector_name in PD_CONNECTOR_PRESETS:
            return PD_CONNECTOR_PRESETS[self.connector_name][0]
        return 10.0

    def resolve_latency_us(self) -> float:
        """resolve startup latency (us)."""
        if self.connector_latency_us is not None:
            return self.connector_latency_us
        if self.connector_name in PD_CONNECTOR_PRESETS:
            return PD_CONNECTOR_PRESETS[self.connector_name][1]
        return 50.0


@dataclass
class ParallelConfig:
    """Parallelism strategy (TP + DP, EP = TP × DP).

    Reference: vLLM parallel_state.py::initialize_model_parallel
    """

    tp_size: int = 1   # Tensor Parallelism
    dp_size: int = 1   # Data Parallelism
    enable_ep: bool = False  # 仅 MoE 模型开启；dense 模型无 EP

    @property
    def ep_size(self) -> int:
        """EP group size. MoE 开启时 = TP × DP，否则 = 1。"""
        if not self.enable_ep:
            return 1
        return self.tp_size * self.dp_size

    @property
    def total_devices(self) -> int:
        return self.tp_size * self.dp_size

    def validate_for_model(
        self,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int = 0,
        ffn_dim: int = 0,
    ):
        """Validate parallel config against model architecture."""
        errors = []
        if num_heads % self.tp_size != 0:
            errors.append(
                f"num_heads({num_heads}) must be divisible by tp_size({self.tp_size})"
            )
        if num_kv_heads % self.tp_size != 0:
            errors.append(
                f"num_kv_heads({num_kv_heads}) must be divisible by tp_size({self.tp_size})"
            )
        if ffn_dim > 0 and ffn_dim % self.tp_size != 0:
            errors.append(
                f"ffn_dim({ffn_dim}) must be divisible by tp_size({self.tp_size})"
            )
        if num_experts > 0 and self.ep_size > 1:
            if num_experts % self.ep_size != 0:
                errors.append(
                    f"num_experts({num_experts}) must be divisible by ep_size({self.ep_size}=tp×dp)"
                )
        if errors:
            raise ValueError(
                "Parallel config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


@dataclass
class LegacyDeployConfig:
    """Runtime deployment configuration."""

    batch_size: int = 1
    input_len: int = 1024
    output_len: int = 128

    # Quantization (byte widths)
    w_byte: float = 2.0   # weight (主体, 量化层): 2=fp16, 1=fp8/int8, 0.5=fp4/int4
    a_byte: float = 2.0   # activation (GEMM input, 量化层): 跟 quant 一致 (fp8 → 1.0)
    kv_byte: float = 2.0  # KV cache: 2=fp16, 1=fp8, 0.5=fp4
    # 非量化层 (lm_head / embed / norm) 始终走基础 dtype.
    # 从 model_config.dtype 推导, 跟 quant_method 解耦. 例 fp8 量化模型:
    #   w_byte = 1.0, base_w_byte = 2.0 (lm_head/embed/norm 仍是 bf16).
    base_w_byte: float = 2.0
    # Activation BUFFER 字节 (HBM 中的 X tensor 实际占地). 跟 a_byte 区分:
    #   - dynamic fp8 量化: GEMM 输入 fp8 (a_byte=1) 但 buffer 短时 bf16 (base_a_byte=2)
    #   - static fp8 / weight-only: 两个值相等
    # 用于 estimate_activation_bytes (KV budget 计算需要 peak buffer 大小).
    base_a_byte: float = 2.0
    # V4-specific: indexer K cache byte width. fp8 = 1.0 默认, fp4 = 0.5.
    # 从 vllm_config.attention_config.use_fp4_indexer_cache 推导(阶段 9 加).
    indexer_kv_byte: float = 1.0

    # compressed-tensors / awq / gptq 的 `ignore` 或 `modules_to_not_convert`:
    # 解析后的"已被 base 路径覆盖的" pattern (lm_head / embed / norm).
    covered_non_quantized: list[str] = field(default_factory=list)
    # 不在已知 base 集合里、需要用户关注的 pattern (例: 某层特定 Linear).
    # 当前 sizing 不为这些做 bytes correction (需 per-Linear graph 才能精确),
    # 用 list 记录 + extract 时 log warn, 让用户知道有 mismatch.
    unhandled_non_quantized_modules: list[str] = field(default_factory=list)

    # Parallelism
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    # PD 分离 (默认未启用 → role=None)
    pd: PDDisaggConfig = field(default_factory=PDDisaggConfig)

    # Features
    use_flash_attention: bool = False
    overlap_comm: bool = False

    @property
    def tp(self) -> int:
        return self.parallel.tp_size

    @property
    def dp(self) -> int:
        return self.parallel.dp_size

    @property
    def ep(self) -> int:
        return self.parallel.ep_size


@dataclass(frozen=True)
class DeployConfig:
    """V3 §4.6 minimal deployment config (search-ready)."""

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 1

    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    block_size: int = 16
    num_gpu_blocks: int | None = None

    execution_mode: str = "eager"
    backend: str = "vllm"
    backend_version: str | None = None
