"""FakeTokenGenerator — 详设 §4.3.5。

阶段 0/2: fixed 模式 (恒定 token=1, 最小可用)。
阶段 4.5 (§10.5): deterministic_hash 模式 — token 由 prompt_token_ids + 已生成 token 数
的稳定 hash 决定, 让重复请求 (相同 prompt) 在 prefix cache ON 下命中"prompt + 输出"
的完整链 (而非仅 prompt 段)。

为什么"deterministic_hash"对 prefix caching 重要:
  vLLM PrefixCache block 键 = hash(token_ids[0..block_end]), 含已生成 token。
  fixed 模式下每个 req_id 第一步都 emit=1, 看起来确实"相同 prompt 同输出",
  但请求间命中只发生在 prompt 段 (decode 段不进 cache, vLLM 不缓存 in-flight
  请求的 output 直到该 block 满)。一旦 block 跨越 prompt/output 边界, fixed
  模式生成的 1,1,1,... 序列在不同请求间也是 1,1,1,..., 巧合命中。
  但当上层换 prompt 后第二次请求 (常见 agentic loop), prompt 端 deterministic
  + decode 端按 prompt 内容差异化, 真实 cache 命中行为更接近线上。

模式:
  - fixed:               所有 token = `fixed_token_id` (默认 1)
  - deterministic_hash:  token = md5(prompt_token_ids ++ generated_index) % vocab_size

切换环境变量: LLM_INFER_SIM_FAKE_TOKEN_MODE = fixed | deterministic_hash
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence


class FakeTokenGenerator:
    """生成 fake output token, 供 VirtualModelRunner 填充 ModelRunnerOutput。

    本类无内部状态: caller 每次传入 (prompt_token_ids, num_generated), 产出下一个
    output token。状态由 VirtualModelRunner._request_states 持有。

    注: fake token 必须 < vocab_size 才不会触发 vLLM detokenize / eos 边界异常。
    """

    VALID_MODES = ("fixed", "deterministic_hash")

    def __init__(
        self,
        mode: str = "fixed",
        vocab_size: int = 32000,
        fixed_token_id: int = 1,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"FakeTokenGenerator: mode={mode!r} 非法, 有效值: {self.VALID_MODES}"
            )
        if vocab_size <= 0:
            raise ValueError(f"FakeTokenGenerator: vocab_size 必须 > 0, 收到 {vocab_size}")
        self.mode = mode
        self.vocab_size = vocab_size
        self.fixed_token_id = max(0, min(fixed_token_id, vocab_size - 1))

    @classmethod
    def from_env(cls, vocab_size: int) -> "FakeTokenGenerator":
        """根据环境变量构造 (供 VirtualModelRunner 调用)。

        默认 fixed 模式向后兼容: 阶段 0/2 的 e2e 用例不需要切换。
        prefix caching / agentic loop 验证场景显式设 deterministic_hash。
        """
        mode = os.environ.get("LLM_INFER_SIM_FAKE_TOKEN_MODE", "fixed").strip().lower()
        return cls(mode=mode, vocab_size=vocab_size)

    def next_token(
        self,
        prompt_token_ids: Sequence[int],
        num_generated: int,
    ) -> int:
        """生成下一个 fake output token id。

        Args:
            prompt_token_ids: 输入 prompt 的 token id 序列 (不含已生成 output)。
                              deterministic_hash 模式必须给; fixed 模式忽略。
            num_generated: 该请求迄今已生成的 output token 数 (本 step 之前)。
                           = 本 step 即将输出的"第 (num_generated+1) 个"output。

        Returns:
            int in [0, vocab_size).
        """
        if self.mode == "fixed":
            return self.fixed_token_id

        # deterministic_hash: 稳定且与 PYTHONHASHSEED 无关
        # 用 md5(prompt + sep + num_generated) → 取前 4 字节 → mod vocab
        # md5 选择理由: stdlib 内置, 输出稳定, 速度对仿真足够 (vs sha256)。
        m = hashlib.md5()
        # prompt 部分: 用 ',' 分隔避免 [1,2] 与 [12] 冲突
        m.update((",".join(str(t) for t in prompt_token_ids)).encode("utf-8"))
        m.update(b"|")
        m.update(str(num_generated).encode("utf-8"))
        digest = m.digest()
        raw = int.from_bytes(digest[:4], "big")
        return raw % self.vocab_size
