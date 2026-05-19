"""跨 profile case 合并 + dedup by case_id.

设计:
  - 同一 op-level shape 在多个 profile 中出现 → 只保留一个 case
  - 保留时把所有"贡献" profile_name 写到 case 的 source_profiles 字段 (metadata 用)
  - case_id 不变 (params hash 不含 profile_name)

为啥 Case dataclass 不直接加 source_profiles:
  - Case 是 framework-agnostic case schema, 不应承载 provenance
  - 走 RawRecord.metadata 路径(runner 跑 case 时知道 case 由哪些 profile 贡献,
    把 list 写进 metadata)

实现: dedup 返 (cases, attribution) 两份, attribution[case_id] = ["qwen3_4b", ...]
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from collector.schemas import Case


def merge_and_dedup(
    per_profile_cases: list[tuple[str, Iterable[Case]]],
) -> tuple[list[Case], dict[str, list[str]]]:
    """合并多 profile case 列表, dedup by case_id.

    Args:
        per_profile_cases: list of (profile_name, cases_iter) pairs.

    Returns:
        (unique_cases, source_profiles_by_case_id)
        - unique_cases: 去重 list (按首次出现顺序)
        - source_profiles_by_case_id: {case_id: [profile_name, ...]} 多个 profile
          贡献同 case 时按出现顺序记录
    """
    cases_by_id: OrderedDict[str, Case] = OrderedDict()
    sources: dict[str, list[str]] = {}

    for profile_name, cases in per_profile_cases:
        for c in cases:
            if c.case_id not in cases_by_id:
                cases_by_id[c.case_id] = c
                sources[c.case_id] = [profile_name]
            else:
                if profile_name not in sources[c.case_id]:
                    sources[c.case_id].append(profile_name)

    return list(cases_by_id.values()), sources
