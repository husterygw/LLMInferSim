"""Shape profiles — 从模型 config 派生的算子 shape 数据.

profile **不是** OperatorDB 主键, 只是 case 生成的输入 + provenance.
同一个 GEMM/MoE shape 可能被多个 profile 引用, 在 case_id 层面会自动 dedup.

每个 profile module 只导出一个 ProfileSpec 常量 + 任何模型本身需要的辅助常量.
不在 profile 模块里生成 case (case 生成在 collector/cases/<op>.py).
"""
from collector.profiles._dims import ProfileSpec
