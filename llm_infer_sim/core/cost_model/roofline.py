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

    def __init__(
        self,
        hw: HardwareConfig,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        efficiency_profile=None,    # EfficiencyProfile | None — 详 §9.4.2 B.6
        execution_mode: str = "eager",   # "eager" | "cudagraph"
    ):
        self.hw = hw
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        # 阶段 X.1 起: 可选传 EfficiencyProfile, 走 per-op lookup_entry 精化.
        # None 时维持 hw scalar (compute_efficiency / mem_efficiency) 全局默认.
        self.efficiency_profile = efficiency_profile
        # Phase 5: cudagraph 模式下 kernel_overhead = 0 (跟通信侧 framework_oh 对称).
        # eager 模式按 hw.kernel_overhead 加 per-op dispatch 开销.
        self.execution_mode = execution_mode

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
        """Get kernel launch overhead for this op category.

        Phase 5: cudagraph 模式下返 0 (跟通信 framework_call_overhead 对称).
        eager 模式下查 hw.kernel_overhead 表(典型 2 µs/op).
        """
        if self.execution_mode == "cudagraph":
            return 0.0
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

        # 阶段 X.1 B.6: per-op efficiency 精化 (calibrated 模式)
        if self.efficiency_profile is not None and op.efficiency_key is not None:
            ratio = self._efficiency_refine_ratio(op)
            if ratio is not None and ratio > 0:
                t_compute *= ratio
                t_memory *= ratio
                inference_time *= ratio

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

    def _efficiency_refine_ratio(self, op: OperatorProfile) -> float | None:
        """计算 per-op efficiency 精化比例.

        raw_time 已经按 hw scalar default 算了 (hw.effective_* 内含 compute_efficiency).
        如 entry 命中, 我们要把 raw_time 调整到使用 entry.efficiency 的水平:
            actual_time = raw_time * (hw_default / entry.efficiency)

        Returns:
            ratio (float) 用作 t_compute / t_memory / inference_time 的乘数;
            None 表示没命中 entry, 保持 raw (hw default).
        """
        if op.efficiency_key is None or self.efficiency_profile is None:
            return None
        op_kind, shape_key = op.efficiency_key
        dtype = _dtype_from_bits(self.w_bit, self.a_bit)
        entry = self.efficiency_profile.lookup_entry(op_kind, dtype, shape_key)
        if entry is None or entry.efficiency <= 0:
            return None
        # hw default 当前 scalar:
        #   compute: hw.compute_efficiency  (默认 1.0, 或被 apply_to 设到 default_compute)
        #   memory: hw.mem_efficiency
        # 算 ratio 时按 op_category 选 compute / memory bottleneck 还需要更细;
        # B.6 v1 简化: 单 ratio 整体 scale, 使用 compute_efficiency 当 default.
        default_eff = (
            self.hw.compute_efficiency
            if op.op_category != "communication"
            else self.hw.comm_efficiency
        )
        if default_eff <= 0:
            default_eff = 1.0
        return default_eff / entry.efficiency

    def analyze_batch(self, ops: list[OperatorProfile]) -> list[RooflineResult]:
        return [self.analyze(op) for op in ops]


def _dtype_from_bits(w_bit: int, a_bit: int) -> str:
    """w_bit/a_bit → dtype 字符串, 跟 fit.py 桶 key 对齐."""
    # 取 weight / activation 中精度较高的当 dtype (cost 主导项)
    # bf16 默认 16, fp8 默认 8, fp4 默认 4
    width = max(w_bit, a_bit)
    if width >= 32:
        return "fp32"
    if width >= 16:
        return "bfloat16"
    if width >= 8:
        return "fp8"
    return "fp4"
