"""Guard: deleted legacy profile abstractions must not reappear in production code.

config_plan Steps 8-9 + Step D 删除了 ProfileBundle / DeployConfig /
EfficiencyProfile / BackendExecutionProfile 以及整个 core/profiles 包。
本测试在它们回流到 llm_infer_sim/ (生产源码) 时 fail-fast。

只扫生产源码 (llm_infer_sim/); tests/tools 里残留的历史 docstring 提及无害。
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

PROD_ROOT = Path(__file__).resolve().parents[2] / "llm_infer_sim"

# 唯一不与 live 名冲突的 token (EfficiencyProfile 是 MoEEfficiencyProfile 子串, 不列)。
# 当前生产源码里这些 token 出现次数 = 0; 任何回流都应让本测试红。
FORBIDDEN_TOKENS = (
    "DeployConfig",
    "ProfileBundle",
    "extract_profile_bundle",
    "BackendExecutionProfile",
)

# 已删除的模块/包 — 不应再可 import。
DELETED_MODULES = (
    "llm_infer_sim.core.profiles",
    "llm_infer_sim.core.runtime.backend",
)


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_forbidden_token_absent_from_production(token):
    offenders = []
    for path in PROD_ROOT.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if token in line:
                offenders.append(f"{path.relative_to(PROD_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        f"legacy symbol {token!r} reappeared in production:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("module", DELETED_MODULES)
def test_deleted_module_not_importable(module):
    assert importlib.util.find_spec(module) is None, (
        f"deleted module {module!r} is importable again — it must stay removed"
    )


# config_plan §5 + Step B/C: 生产装配链只吃结构化域对象 (ModelProfile / HardwareProfile /
# SimulationScenario), 不得用 ModelConfig ↔ HardwareConfig 的 to_legacy()/from_legacy()
# 折返。合法的 legacy 转换只允许出现在显式 ingest 边界 (profile_extractor 把 vLLM 配置喂进
# 结构化域) + hardware registry 域入口 + dataclass 自身的方法定义 —— 这些不在下表里。
ASSEMBLY_PATH_FILES = (
    "core/cost/engine.py",
    "core/operators/context.py",
    "adapters/vllm/virtual_model_runner.py",
    "adapters/vllm/virtual_worker.py",
)

_LEGACY_CALL = re.compile(r"\.(to|from)_legacy\(")


@pytest.mark.parametrize("rel", ASSEMBLY_PATH_FILES)
def test_assembly_path_free_of_legacy_conversion(rel):
    """生产装配文件不得调用 .to_legacy()/.from_legacy() (docstring 提及无害)。"""
    path = PROD_ROOT / rel
    offenders = [
        f"{rel}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _LEGACY_CALL.search(line)
    ]
    assert not offenders, (
        f"legacy to_legacy/from_legacy conversion reappeared in assembly path:\n"
        + "\n".join(offenders)
    )


# Phase 5.5: 冻结整个生产树的 legacy flat config 折返面。.to_legacy()/.from_legacy()
# 调用只允许出现在两个显式边界: profile_extractor (vLLM 配置 ingest) + hardware
# registry (硬件域入口)。dataclass 自身的 def to_legacy/def from_legacy 不带前导点,
# 不被 _LEGACY_CALL 命中。任何新文件引入折返调用都应让本测试红。
ALLOWED_LEGACY_BOUNDARY_FILES = frozenset(
    {
        "adapters/vllm/profile_extractor.py",
        "core/hardware/registry.py",
    }
)


def test_legacy_conversion_confined_to_allowed_boundaries():
    offenders: dict[str, list[str]] = {}
    for path in PROD_ROOT.rglob("*.py"):
        rel = path.relative_to(PROD_ROOT).as_posix()
        if rel in ALLOWED_LEGACY_BOUNDARY_FILES:
            continue
        hits = [
            f"{rel}:{i}: {line.strip()}"
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if _LEGACY_CALL.search(line)
        ]
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        "to_legacy/from_legacy 折返调用只允许出现在显式 ingest/域边界 "
        f"({sorted(ALLOWED_LEGACY_BOUNDARY_FILES)}); 新增使用面:\n"
        + "\n".join(line for hits in offenders.values() for line in hits)
    )
