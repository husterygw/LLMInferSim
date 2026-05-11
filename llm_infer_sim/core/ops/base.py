"""Operator profile dataclass with 5-way memory access decomposition."""

from dataclasses import dataclass


@dataclass
class OperatorProfile:
    """Profile of a single operator's compute/memory/communication characteristics."""

    name: str
    op_category: str = "matmul"  # matmul|attention|norm|activation|embedding|communication

    # Compute
    flops: int = 0  # total FLOPs (FMA counts as 2)

    # Memory access — 5-way decomposition (from llm-viewer)
    load_weight: int = 0
    load_act: int = 0
    store_act: int = 0
    load_kv_cache: int = 0
    store_kv_cache: int = 0

    # Per-op precision override for roofline (overrides global w_bit/a_bit settings)
    # "": use global quantization settings  "fp8": INT8 TC peak  "bf16": FP16 TC peak
    # "fp32": vector (CUDA core) peak
    op_precision: str = ""

    # Communication (from llm_inference_eval)
    comm_bytes: float = 0.0
    comm_type: str = ""  # allreduce|allgather|alltoall|p2p|""

    @property
    def mem_bytes(self) -> int:
        return self.load_weight + self.load_act + self.store_act + self.load_kv_cache + self.store_kv_cache

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.mem_bytes if self.mem_bytes > 0 else float("inf")
