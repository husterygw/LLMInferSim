"""Op-level case generators.

每个 module 对应一个 OpKind, 暴露:
  - get_cases_for_profile(profile, **opts) -> list[Case]   # 单 profile
  - get_cases(profiles, **opts) -> list[Case]              # 多 profile, dedup by case_id

case_id 仅基于 op-level 参数 (shape, dtype, parallel config, routing, ...),
**不含 profile_name**. 同 shape 不同 profile 自然 dedup.
profile 来源走 RawRecord.metadata["source_profiles"] 提供 provenance.
"""
