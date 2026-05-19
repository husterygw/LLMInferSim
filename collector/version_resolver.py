"""按 framework_version 解析最终 runner_module path.

设计:
  - CollectorEntry.versions 是 VersionRoute tuple
  - 给定 actual_version, 选 max(min_version) <= actual_version 的那条
  - 都不匹配 → entry.run_case_module (默认 runner)

版本比较: tuple of int parts. 例:
  "0.19.1" → (0, 19, 1)
  "0.20.0" → (0, 20, 0)
  (0, 19, 1) < (0, 20, 0)   ✓

非数字 suffix 例 "0.19.0rc1" 当前 strip 掉 (只保留前面 digits), 后续如需精确语义
再引入 packaging.version.Version.
"""
from __future__ import annotations

import re

from collector.schemas import CollectorEntry, VersionRoute


_VERSION_PART_RE = re.compile(r"^(\d+)")


def parse_version(v: str) -> tuple[int, ...]:
    """字符串 → tuple[int]. 'rc1' / '+abc' 之类的 suffix 当前简单 strip.

    例:
      "0.19.1"     → (0, 19, 1)
      "0.20.0rc1"  → (0, 20, 0)
      "1.0"        → (1, 0)
      ""           → ()
    """
    if not v:
        return ()
    parts: list[int] = []
    for raw in v.split("."):
        m = _VERSION_PART_RE.match(raw)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def resolve_runner(entry: CollectorEntry, framework_version: str) -> str:
    """返该 entry 在 framework_version 下应用的 runner_module path.

    规则:
      1. 把 entry.versions 按 min_version desc 排序
      2. 取第一条 min_version <= framework_version 的
      3. 都不匹配 → entry.run_case_module
    """
    actual = parse_version(framework_version)
    # 选最大的、且 <= actual 的 min_version
    candidates: list[VersionRoute] = []
    for route in entry.versions:
        if parse_version(route.min_version) <= actual:
            candidates.append(route)
    if not candidates:
        return entry.run_case_module
    # 取 min_version 最大的那条 (最具体的匹配)
    best = max(candidates, key=lambda r: parse_version(r.min_version))
    return best.runner_module
