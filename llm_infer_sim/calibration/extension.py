"""vLLM worker_extension_cls — 在每个 TP worker 里跑 layerwise_profile (B.2).

注册方式 (B.3 runner.spin_up 调):

    llm = vllm.LLM(
        model=...,
        worker_extension_cls="llm_infer_sim.calibration.extension.LayerwiseProfileExtension",
        ...,
    )

vLLM 给每个 TP rank 进程实例化一份 Extension, 注入 self.model_runner. host 用
`llm.collective_rpc("fire", args=(shot_dict, slice_, kind, iterations))` 触发,
每 rank 跑同样的 shot, 返回 per-rank timing list.

测量协议 (per shot):
  1. warmup forward (废弃) — JIT / paged buffer setup 摊销
  2. layerwise_profile 上下文里跑 N forwards (默认 3)
     vLLM 自动累加 cuda_time_us 跟 invocations
  3. extract_samples 用 catalog_slice 匹中 canonical layer, per_invocation_us = total / invocations
"""
from __future__ import annotations

from typing import Any


class LayerwiseProfileExtension:
    """Worker-side profile entry. vLLM 在每 TP worker 实例化, 注入 self.model_runner."""

    def fire(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        kind: str,
        iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """Run one profile shot, return TimingSample list (as dicts for pickle).

        Args:
            shot_dict: Shot.to_dict() 形式.
            slice_: Catalog.slice_for_op_kinds() 形式
                    {canonical: {"vllm": cls, "within": parent, "op_kind": ...}}.
            kind: shot.kind, 当前不用 (留给后续 MoE forge routing).
            iterations: 计入 profile 的 forward 次数; 1 也行但跨 invocation 噪声大.

        Returns:
            list[dict], 每 dict = TimingSample.as_dict() = {layer, op_kind, microseconds}.

        Raises:
            RuntimeError: model_runner / layerwise_profile API 不可用.
        """
        # 局部 import — 主模块加载时 (host 端) 不必拉 vLLM
        from llm_infer_sim.calibration.batch import assemble_scheduler_output
        from llm_infer_sim.calibration.shots import Shot
        from llm_infer_sim.calibration.timings import extract_samples

        try:
            from vllm.profiler.layerwise_profile import layerwise_profile
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "vllm.profiler.layerwise_profile 不可用. "
                "需要 vLLM >= 0.18 with torch profiler 支持."
            ) from e

        shot = Shot.hydrate(shot_dict)
        iterations = max(1, int(iterations))

        # ---- warmup (废弃结果) ----
        # 摊销 JIT 编译 / cudagraph capture (如果 vLLM 没强制 enforce_eager) /
        # paged-attention buffer 分配 / 第一次 nccl init.
        batch, _ = assemble_scheduler_output(shot, self.model_runner)
        warmup_out = self.model_runner.execute_model(batch)
        if warmup_out is None:
            # 部分 vLLM 路径下 execute_model 把 sampling 拆到 sample_tokens
            try:
                self.model_runner.sample_tokens(None)
            except (AttributeError, Exception):  # noqa: BLE001
                pass     # 无 sample_tokens 也无关

        # ---- 测量 ----
        with layerwise_profile() as profiler:
            for _ in range(iterations):
                # 每 iter 重建 SchedulerOutput, 避免 prior-iter KV writes 污染下次
                fresh_batch, _ = assemble_scheduler_output(shot, self.model_runner)
                out = self.model_runner.execute_model(fresh_batch)
                # vLLM 0.19+ async scheduling: execute_model 返 None 时 worker 处在
                # "等 sample_tokens" 状态, 下一次 execute_model 会抛 State error.
                # 每次都补一发 sample_tokens 让 worker 进入 "ready" 状态.
                if out is None:
                    try:
                        self.model_runner.sample_tokens(None)
                    except Exception:    # noqa: BLE001
                        pass

        results = profiler.results.convert_stats_to_dict()
        model_stats = results.get("model_stats", []) or []
        samples = extract_samples(model_stats, slice_)

        # 序列化 (collective_rpc 走 pickle, plain dict 最稳)
        return [s.as_dict() for s in samples]

    def fire_raw_tree(self, shot_dict: dict) -> dict:
        """Debug: 跑一次 shot 返 raw LayerwiseProfileResults model_stats tree.

        给 calibration 排错用 — 看 vLLM 实际 module class names 是否跟我们 catalog 对齐.
        """
        from llm_infer_sim.calibration.batch import assemble_scheduler_output
        from llm_infer_sim.calibration.shots import Shot

        try:
            from vllm.profiler.layerwise_profile import layerwise_profile
        except ImportError as e:
            return {"error": f"layerwise_profile unavailable: {e}"}

        shot = Shot.hydrate(shot_dict)
        # warmup
        batch, _ = assemble_scheduler_output(shot, self.model_runner)
        self.model_runner.execute_model(batch)
        try:
            self.model_runner.sample_tokens(None)
        except Exception:    # noqa: BLE001
            pass
        # measure
        with layerwise_profile() as profiler:
            fresh_batch, _ = assemble_scheduler_output(shot, self.model_runner)
            self.model_runner.execute_model(fresh_batch)
        results = profiler.results.convert_stats_to_dict()
        return results
