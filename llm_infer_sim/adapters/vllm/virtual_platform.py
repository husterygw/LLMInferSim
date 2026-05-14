"""VirtualPlatform — Phase 1 spike 最小版。

目标只验证一件事：CPU wheel + VLLM_USE_V1=1 + 通过 entry_point
注册的 VirtualPlatform 能被 vLLM 正确选中。

最小集做法:
  1. virtual_platform_plugin() 在 VLLM_VIRTUAL_BACKEND=1 时返回 qualname
  2. VirtualPlatform 继承 Platform, 提供必要的 classmethod
  3. check_and_update_config() 把 worker_cls 注入为 VirtualWorker

Phase 2+ 才补 feature gate / hf_config 离线注入 / 真实 cost model。
"""
from __future__ import annotations

import os

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.logger import init_logger

logger = init_logger(__name__)


class VirtualPlatform(Platform):
    _enum = PlatformEnum.OOT
    # device_name 必须是 torch 认识的字符串 (vLLM 内部会 torch.device(f"{device_name}:{rank}"))
    # 我们底层是 CPU tensor, 所以直接声称 cpu 即可。
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    dist_backend: str = "gloo"
    simple_compile_backend: str = "eager"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        return DeviceCapability(major=9, minor=0)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "VirtualPerformanceBackend"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"VIRTUAL-{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # 80GB —— 对应 H100 规格, 让 KV cache size 计算有合理基线
        # todo: 后续阶段可以改成 config 注入, 让用户模拟不同设备规格
        return 80 * 1024 * 1024 * 1024

    @classmethod
    def inference_mode(cls):
        import torch
        return torch.no_grad()

    @classmethod
    def set_device(cls, device) -> None:
        pass

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        import torch
        torch.manual_seed(seed)

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "llm_infer_sim.adapters.vllm.virtual_worker.VirtualWorker"
            )

        # 防御性: 任何路径触发 model loader 时也走 dummy, 不下载权重
        vllm_config.load_config.load_format = "dummy"

        # 没有真实模型, 不可能 capture cudagraph
        if vllm_config.compilation_config is not None:
            vllm_config.compilation_config.cudagraph_capture_sizes = []

        scheduler_config = vllm_config.scheduler_config
        if getattr(scheduler_config, "async_scheduling", False):
            scheduler_config.async_scheduling = False

        # 阶段 3 C 块: 6 类 fail-fast Feature Gate (详设 §7.5.2)
        cls._check_unsupported_features(vllm_config)

        logger.info(
            "VirtualPlatform: worker_cls set to %s, feature gate passed",
            parallel_config.worker_cls,
        )

    @classmethod
    def _check_unsupported_features(cls, vllm_config) -> None:
        """对应详设 §7.5.2 不支持 feature 清单的强制检查。

        每项检查发现违例时把错误描述塞进 errors 列表, 最后聚合 raise ValueError。
        每条错误都附 "替代:" 提示用户怎么继续。
        """
        errors: list[str] = []

        # ---- 1. LoRA ----
        if getattr(vllm_config, "lora_config", None) is not None:
            errors.append(
                "LoRA adapter (--enable-lora) is not supported: "
                "VirtualWorker uses config-only model loading and has no "
                "real nn.Module to inject LoRA layers into. "
                "替代: 在外部 cost model 中按 LoRA rank 估算 GEMM 增量。"
            )

        # ---- 2. Speculative decoding ----
        if getattr(vllm_config, "speculative_config", None) is not None:
            errors.append(
                "Speculative decoding (--speculative-config) is not supported: "
                "fake token sequence makes acceptance rate always 100%, "
                "which would produce misleading speedup numbers. "
                "替代: 在 cost model 中独立建模 draft+verify 开销, 并显式提供 "
                "acceptance_rate 参数。"
            )

        # ---- 3. Multi-modal ----
        model_cfg = vllm_config.model_config
        if getattr(model_cfg, "is_multimodal_model", False):
            errors.append(
                "Multi-modal model (vision/audio encoder) is not supported: "
                "encoder needs real nn.Module + real input tensors. "
                "替代: 仅支持纯文本 LLM; 如需评估 MM 模型, 单独建模 encoder 部分。"
            )

        # ---- 4. Guided decoding / Structured output ----
        decoding_cfg = getattr(vllm_config, "decoding_config", None)
        if decoding_cfg is not None and getattr(
            decoding_cfg, "guided_decoding_backend", None
        ):
            errors.append(
                "Guided decoding (--guided-decoding-backend) is not supported: "
                "logits processor needs real logits tensor; fake output cannot "
                "satisfy grammar/regex constraints. "
                "替代: 在 cost model 中按 vocab_size × constraint_overhead 估算。"
            )

        # ---- 5. Logprobs ----
        max_logprobs = getattr(model_cfg, "max_logprobs", 0) or 0
        if max_logprobs > 0:
            errors.append(
                f"Logprobs (--max-logprobs={max_logprobs}>0) is not supported: "
                "no real logits tensor available. "
                "替代: 设置 --max-logprobs 0; 或在 cost model 中按 "
                "top_k × bytes_per_logprob 估算 transfer 开销。"
            )

        # ---- 6. KV connector / disaggregated prefill (详设 §7.6) ----
        # MVP 已实现: P2pNcclConnector / LMCacheConnectorV1 / MooncakeConnector /
        # NixlConnector / OffloadingConnector 等 (详 PD_CONNECTOR_PRESETS).
        kvt = getattr(vllm_config, "kv_transfer_config", None)
        if kvt is not None:
            from llm_infer_sim.core.profiles.deploy import PD_CONNECTOR_PRESETS
            role = getattr(kvt, "kv_role", None)
            connector = getattr(kvt, "kv_connector", None)
            if role is None:
                # vLLM 允许 kv_transfer_config 但 role=None 当 noop; 我们也放行
                pass
            elif role not in ("kv_producer", "kv_consumer", "kv_both"):
                errors.append(
                    f"KV transfer: unknown kv_role={role!r}; 期望 "
                    f"kv_producer / kv_consumer / kv_both。"
                )
            elif connector not in PD_CONNECTOR_PRESETS:
                # 允许 env override 强行放行 (未知 connector 走 fallback 10 GB/s)
                if os.environ.get("VLLM_INFER_SIM_ALLOW_UNKNOWN_PD_CONNECTOR", "0") != "1":
                    errors.append(
                        f"KV transfer connector={connector!r} 不在预设带宽表中: "
                        f"{sorted(PD_CONNECTOR_PRESETS.keys())}; "
                        f"设 VLLM_INFER_SIM_ALLOW_UNKNOWN_PD_CONNECTOR=1 走 fallback 带宽。"
                    )

        # ---- 7. Unsupported attention backend (阶段 3.5, 详设 §4.8.1.1) ----
        # platform 启动期就拦, 比 cost model 内部 raise 更早, 错误信息更直接。
        attn_cfg = getattr(vllm_config, "attention_config", None)
        backend = getattr(attn_cfg, "backend", None) if attn_cfg is not None else None
        if backend is not None:
            from llm_infer_sim.adapters.vllm.profile_extractor import (
                _VLLM_BACKEND_MODE_MAP,
                _VLLM_UNSUPPORTED_BACKENDS,
            )
            name = backend.name
            if name in _VLLM_UNSUPPORTED_BACKENDS:
                errors.append(
                    f"Attention backend {name} is not supported (阶段 3.5): "
                    f"本系统当前仅支持 NVIDIA CUDA / FlashInfer 系列。"
                    f"已支持: {sorted(_VLLM_BACKEND_MODE_MAP.keys())}。"
                    f"替代: 设置 VLLM_ATTENTION_BACKEND=FLASH_ATTN 或留空走默认。"
                )
            elif name not in _VLLM_BACKEND_MODE_MAP:
                errors.append(
                    f"Unknown attention backend {name} (新 enum, 未在 §4.8.1.1 "
                    f"_VLLM_BACKEND_MODE_MAP 中映射)。请在 adapters/vllm/profile_extractor.py "
                    f"加映射, 或临时设置 VLLM_ATTENTION_BACKEND=FLASH_ATTN 绕过。"
                )

        if errors:
            raise ValueError(
                "VirtualPlatform: unsupported vLLM feature(s) detected:\n  - "
                + "\n  - ".join(errors)
                + "\n\n详见详设 §7.5.2 / 系方 §2.3.4 不支持 feature 清单。"
            )

    @classmethod
    def get_attn_backend_cls(cls, *args, **kwargs) -> str:
        # 不返回任何真实 attention backend
        return ""


def virtual_platform_plugin() -> str | None:
    """vLLM platform plugin entry_point.

    通过 VLLM_VIRTUAL_BACKEND=1 启用; 否则返回 None 让 vLLM 走默认平台
    探测路径 (这样安装了本包但没启用时不会影响其他用法)。
    """
    if os.environ.get("VLLM_VIRTUAL_BACKEND", "0") == "1":
        logger.info("VirtualPlatform plugin activated")
        return "llm_infer_sim.adapters.vllm.virtual_platform.VirtualPlatform"
    return None
