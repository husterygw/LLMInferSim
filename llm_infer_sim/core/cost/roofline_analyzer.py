"""Roofline analyzer for semantic Operator formulas."""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.operators.specs import OperatorFormula
from llm_infer_sim.core.profiles.hardware import HardwareConfig


@dataclass
class RooflineResult:
    name: str
    flops: int
    mem_bytes: int
    comm_bytes: float
    t_compute: float = 0.0
    t_memory: float = 0.0
    t_comm: float = 0.0
    kernel_overhead: float = 0.0
    total_time: float = 0.0
    arithmetic_intensity: float = 0.0
    achievable_performance: float = 0.0
    bottleneck: str = ""
    load_weight: int = 0
    load_act: int = 0
    store_act: int = 0
    load_kv_cache: int = 0
    store_kv_cache: int = 0


class RooflineAnalyzer:
    def __init__(
        self,
        hw: HardwareConfig,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        efficiency_profile=None,
        execution_mode: str = "eager",
    ):
        self.hw = hw
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.efficiency_profile = efficiency_profile
        self.execution_mode = execution_mode

    def _select_peak(self, formula: OperatorFormula) -> float:
        if formula.op_precision == "fp8":
            return self.hw.effective_peak_fp8
        if formula.op_precision == "fp4":
            return (
                self.hw.effective_peak_fp4
                if self.hw.has_fp4_tc
                else self.hw.effective_peak_flops
            )
        if formula.op_precision == "fp32":
            return self.hw.effective_vector_flops
        if formula.op_precision in ("bf16", "fp16"):
            return self.hw.effective_peak_flops

        if formula.op_category in ("norm", "activation", "softmax"):
            return self.hw.effective_vector_flops

        if self.w_bit > 8 or self.a_bit > 8:
            return self.hw.effective_peak_flops
        if self.w_bit <= 4 and self.a_bit <= 4 and self.hw.has_fp4_tc:
            return self.hw.effective_peak_fp4
        if self.w_bit <= 8 and self.a_bit <= 8:
            return self.hw.effective_peak_int8
        return self.hw.effective_peak_flops

    def _get_kernel_overhead(self, op_category: str) -> float:
        if self.execution_mode == "cudagraph":
            return 0.0
        overheads = self.hw.kernel_overhead
        if not overheads:
            return 0.0
        return overheads.get(op_category, overheads.get("default", 0.0))

    def analyze(self, name: str, formula: OperatorFormula) -> RooflineResult:
        peak = self._select_peak(formula)
        bandwidth = self.hw.effective_mem_bandwidth
        mem_bytes = formula.mem_bytes
        flops = formula.flops

        turning_point = peak / bandwidth if bandwidth > 0 else float("inf")
        arithmetic_intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")
        if arithmetic_intensity < turning_point:
            bound = "memory"
            performance = arithmetic_intensity * bandwidth
        else:
            bound = "compute"
            performance = peak
        if performance == 0:
            performance = 1e-30

        inference_time = flops / performance
        t_compute = flops / peak if peak > 0 else 0.0
        t_memory = mem_bytes / bandwidth if bandwidth > 0 else 0.0
        k_overhead = self._get_kernel_overhead(formula.op_category)

        return RooflineResult(
            name=name,
            flops=flops,
            mem_bytes=mem_bytes,
            comm_bytes=formula.comm_bytes,
            t_compute=t_compute,
            t_memory=t_memory,
            t_comm=0.0,
            kernel_overhead=k_overhead,
            total_time=inference_time + k_overhead,
            arithmetic_intensity=arithmetic_intensity,
            achievable_performance=performance,
            bottleneck=bound,
            load_weight=formula.load_weight,
            load_act=formula.load_act,
            store_act=formula.store_act,
            load_kv_cache=formula.load_kv_cache,
            store_kv_cache=formula.store_kv_cache,
        )
