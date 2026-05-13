"""Roofline analyzer + result (merged from llm-viewer roofline/{result,analyzer}.py)."""

from dataclasses import dataclass

from llm_infer_sim.core.ops.base import OperatorProfile
from llm_infer_sim.core.profiles.hardware import HardwareConfig


@dataclass
class RooflineResult:
    """Result of roofline analysis for a single operator."""

    name: str
    flops: int
    mem_bytes: int
    comm_bytes: float

    # Time decomposition
    t_compute: float = 0.0    # flops / effective_peak
    t_memory: float = 0.0     # mem_bytes / effective_bandwidth
    t_comm: float = 0.0       # communication latency
    kernel_overhead: float = 0.0
    total_time: float = 0.0   # max(t_compute, t_memory) + t_comm + overhead

    # Roofline metrics
    arithmetic_intensity: float = 0.0  # flops / mem_bytes
    achievable_performance: float = 0.0  # min(AI * BW, peak) OPS
    bottleneck: str = ""  # "compute-bound" | "memory-bound"

    # Memory access decomposition (pass-through from OperatorProfile)
    load_weight: int = 0
    load_act: int = 0
    store_act: int = 0
    load_kv_cache: int = 0
    store_kv_cache: int = 0

# ===== analyzer (merged from roofline/analyzer.py) =====
"""Roofline performance analyzer.

Phase 1: Exact behavioral parity with the original roofline_model.py.
          No efficiency calibration (efficiency=1.0), no vector_flops distinction,
          no kernel overhead. These are added in Phase 2.

Phase 2: Adds efficiency calibration, vector_flops/tensor_flops distinction,
          kernel overhead, and communication time.
"""



class RooflineAnalyzer:
    """Roofline model analyzer."""

    def __init__(self, hw: HardwareConfig, w_bit: int = 16, a_bit: int = 16, kv_bit: int = 16):
        self.hw = hw
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit

    def _select_peak(self, op: "OperatorProfile") -> float:
        """Select peak performance based on op precision and category.

        Priority order:
        1. Per-op precision override (op.op_precision != "")
        2. Global quantization settings (w_bit/a_bit/kv_bit)
        3. Element-wise ops always use vector (CUDA core) peak
        """
        # Per-op precision override
        if op.op_precision == "fp8":
            return self.hw.effective_peak_fp8
        if op.op_precision == "fp4":
            return self.hw.effective_peak_fp4 if self.hw.has_fp4_tc else self.hw.effective_peak_flops
        if op.op_precision == "fp32":
            return self.hw.effective_vector_flops
        if op.op_precision == "bf16" or op.op_precision == "fp16":
            # bf16 / fp16 共用 Tensor Core 16-bit peak. 显式列 "fp16" 防止
            # 当全局 w_bit=8 (fp8 quant) 时, 标注为 fp16 的 op 被 fall-through
            # 错误地按 fp8 peak (≈ 2× 虚高) 算。
            return self.hw.effective_peak_flops

        # Element-wise ops run on CUDA cores, not tensor cores
        if op.op_category in ("norm", "activation", "softmax"):
            return self.hw.effective_vector_flops

        # Global quantization: peak determined by weight/activation precision only
        # (kv_bit excluded — KV cache precision does not affect main GEMM Tensor Core).
        # Mirrors old model_analyzer.py logic:
        #   w>8 or a>8  → FP16;  w≤4 and a≤4  → FP4;  w≤8 and a≤8  → INT8/FP8
        if self.w_bit > 8 or self.a_bit > 8:
            return self.hw.effective_peak_flops
        if self.w_bit <= 4 and self.a_bit <= 4 and self.hw.has_fp4_tc:
            return self.hw.effective_peak_fp4
        if self.w_bit <= 8 and self.a_bit <= 8:
            return self.hw.effective_peak_int8
        return self.hw.effective_peak_flops

    def _get_kernel_overhead(self, op_category: str) -> float:
        """Get kernel launch overhead for this op category."""
        overheads = self.hw.kernel_overhead
        if not overheads:
            return 0.0
        return overheads.get(op_category, overheads.get("default", 0.0))

    def analyze(self, op: OperatorProfile) -> RooflineResult:
        """Analyze a single operator against the roofline model.

        Returns a RooflineResult with the same arithmetic_intensity, performance,
        and bound classification as the original roofline_analyze() function.
        """
        peak = self._select_peak(op)
        bandwidth = self.hw.effective_mem_bandwidth

        mem_bytes = op.mem_bytes
        flops = op.flops

        # Original roofline logic (exact replica)
        turning_point = peak / bandwidth if bandwidth > 0 else float("inf")
        arithmetic_intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")

        if arithmetic_intensity < turning_point:
            bound = "memory"
            performance = arithmetic_intensity * bandwidth
        else:
            bound = "compute"
            performance = peak

        if performance == 0:
            performance = 1e-30  # avoid division by zero

        inference_time = flops / performance

        # Time decomposition
        t_compute = flops / peak if peak > 0 else 0.0
        t_memory = mem_bytes / bandwidth if bandwidth > 0 else 0.0
        k_overhead = self._get_kernel_overhead(op.op_category)

        return RooflineResult(
            name=op.name,
            flops=flops,
            mem_bytes=mem_bytes,
            comm_bytes=op.comm_bytes,
            t_compute=t_compute,
            t_memory=t_memory,
            t_comm=0.0,
            kernel_overhead=k_overhead,
            total_time=inference_time + k_overhead,
            arithmetic_intensity=arithmetic_intensity,
            achievable_performance=performance,
            bottleneck=bound,
            load_weight=op.load_weight,
            load_act=op.load_act,
            store_act=op.store_act,
            load_kv_cache=op.load_kv_cache,
            store_kv_cache=op.store_kv_cache,
        )

    def analyze_batch(self, ops: list[OperatorProfile]) -> list[RooflineResult]:
        return [self.analyze(op) for op in ops]
