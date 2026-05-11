"""Deployment and parallelism configuration dataclasses."""

from dataclasses import dataclass, field


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
class DeployConfig:
    """Runtime deployment configuration."""

    batch_size: int = 1
    input_len: int = 1024
    output_len: int = 128

    # Quantization (byte widths)
    w_byte: float = 2.0   # weight: 2=fp16, 1=int8, 0.5=int4
    a_byte: float = 2.0   # activation
    kv_byte: float = 2.0  # KV cache

    # Parallelism
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

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
