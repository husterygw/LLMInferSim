"""Smoke test (轻探针): 不启动模型, 只验证 entry_point + 平台选中。

run:
    cd inference-perf-sim/
    pip install -e .
    VLLM_VIRTUAL_BACKEND=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python tests/smoke_platform_selected.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("VLLM_VIRTUAL_BACKEND") != "1":
        print("FAIL: please run with VLLM_VIRTUAL_BACKEND=1", file=sys.stderr)
        return 1

    # 1. entry_point 注册可见性
    from importlib.metadata import entry_points

    eps = entry_points(group="vllm.platform_plugins")
    names = sorted(ep.name for ep in eps)
    print(f"[1] vllm.platform_plugins entry points: {names}")
    assert "virtual" in names, "virtual platform plugin not registered"

    # 2. plugin 函数返回 qualname
    from llm_infer_sim.adapters.vllm.virtual_platform import virtual_platform_plugin

    qualname = virtual_platform_plugin()
    print(f"[2] virtual_platform_plugin() -> {qualname}")
    assert qualname == "llm_infer_sim.adapters.vllm.virtual_platform.VirtualPlatform"

    # 3. vLLM 的 platform 解析路径选中我们的 plugin
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.platforms import current_platform

    print(f"[3] vllm.current_platform = {type(current_platform).__name__}")
    assert type(current_platform).__name__ == "VirtualPlatform", (
        f"current_platform should be VirtualPlatform, got {type(current_platform).__name__}"
    )

    # 4. check_and_update_config 注入 worker_cls
    from vllm.config import ParallelConfig

    pc = ParallelConfig()
    cfg = type("FakeCfg", (), {})()
    cfg.parallel_config = pc
    cfg.load_config = type("LC", (), {"load_format": "auto"})()
    cfg.compilation_config = type("CC", (), {"cudagraph_capture_sizes": [1, 2]})()
    cfg.scheduler_config = type("SC", (), {"async_scheduling": True})()
    current_platform.check_and_update_config(cfg)

    print(f"[4] worker_cls after check_and_update_config = {pc.worker_cls!r}")
    assert pc.worker_cls == "llm_infer_sim.adapters.vllm.virtual_worker.VirtualWorker"
    assert cfg.load_config.load_format == "dummy"
    assert cfg.compilation_config.cudagraph_capture_sizes == []
    assert cfg.scheduler_config.async_scheduling is False

    print("\nSMOKE TEST PASSED — VirtualPlatform is correctly selected & configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
