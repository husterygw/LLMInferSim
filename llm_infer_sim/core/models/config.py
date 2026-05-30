"""ModelProfile — 模型域配置 (config_plan §4.1)。

模型配置只描述结构和量化, 不描述部署或 runtime。结构以嵌套 ModelArchitecture
表达 (attention / ffn / moe / mla)。

flat legacy `ModelConfig` (canonical 定义在本模块, config_plan Step 5 从
core/profiles/model_config.py 迁入) 保留为 compatibility dataclass;
from_legacy / to_legacy 在 ModelProfile ↔ ModelConfig 间转换。
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_infer_sim.core.models.quantization import QuantizationProfile


@dataclass
class ModelConfig:
    """Model architecture parameters for parallel analysis (flat compatibility form)."""

    name: str = "model"
    # HF architectures[0] (e.g. "Qwen3ForCausalLM" / "Qwen3MoeForCausalLM"); 跟 vLLM
    # 模型注册同口径, 供 model-graph registry 精确分发. "" = 未知 → registry 按结构兜底.
    arch: str = ""
    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    ffn_dim: int = 11008       # Dense FFN intermediate size
    num_layers: int = 32
    vocab_size: int = 32000

    # MoE
    is_moe: bool = False
    num_experts: int = 0
    num_activated_experts: int = 0  # top_k
    expert_dim: int = 0             # per-expert intermediate dim
    num_shared_experts: int = 0
    moe_layer_freq: int = 1         # every N layers is MoE (1 = all MoE)
    first_moe_layer: int = 0        # first MoE layer index

    # MLA (Multi-head Latent Attention) — set to kv_lora_rank + qk_rope_head_dim
    # If > 0, KV cache uses this dim instead of num_kv_heads * head_dim
    kv_latent_dim: int = 0
    kv_lora_rank: int = 0       # MLA: compressed KV dim (c_kv); 0 = standard MHA
    v_head_dim: int = 0         # MLA: per-head V dim (may differ from head_dim)
    qk_nope_head_dim: int = 0   # MLA: qk head dim without rope (e.g. 128 in DSV3)
    rope_head_dim: int = 0      # qk_rope_head_dim (64 in V3/V3.2)

    # Q low-rank projection (V3 用; V4 用过的 o_lora_rank 已删)
    q_lora_rank: int = 0        # Q low-rank bottleneck (1536 in V3); 0 = direct projection

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if layer_idx < self.first_moe_layer:
            return False
        return (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0


@dataclass(frozen=True)
class AttentionArchitecture:
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128


@dataclass(frozen=True)
class FFNArchitecture:
    ffn_dim: int = 11008


@dataclass(frozen=True)
class MoEArchitecture:
    num_experts: int = 0
    num_activated_experts: int = 0
    expert_dim: int = 0
    num_shared_experts: int = 0
    moe_layer_freq: int = 1
    first_moe_layer: int = 0


@dataclass(frozen=True)
class MLAArchitecture:
    kv_latent_dim: int = 0
    kv_lora_rank: int = 0
    v_head_dim: int = 0
    qk_nope_head_dim: int = 0
    rope_head_dim: int = 0
    q_lora_rank: int = 0


@dataclass(frozen=True)
class ModelArchitecture:
    hidden_dim: int = 4096
    num_layers: int = 32
    vocab_size: int = 32000
    attention: AttentionArchitecture = AttentionArchitecture()
    ffn: FFNArchitecture = FFNArchitecture()
    moe: MoEArchitecture | None = None
    mla: MLAArchitecture | None = None


@dataclass(frozen=True)
class ModelProfile:
    name: str
    arch: str
    architecture: ModelArchitecture
    quantization: QuantizationProfile

    @classmethod
    def from_legacy(
        cls,
        model: ModelConfig,
        quantization: QuantizationProfile | None = None,
    ) -> "ModelProfile":
        moe = (
            MoEArchitecture(
                num_experts=model.num_experts,
                num_activated_experts=model.num_activated_experts,
                expert_dim=model.expert_dim,
                num_shared_experts=model.num_shared_experts,
                moe_layer_freq=model.moe_layer_freq,
                first_moe_layer=model.first_moe_layer,
            )
            if model.is_moe
            else None
        )
        mla = (
            MLAArchitecture(
                kv_latent_dim=model.kv_latent_dim,
                kv_lora_rank=model.kv_lora_rank,
                v_head_dim=model.v_head_dim,
                qk_nope_head_dim=model.qk_nope_head_dim,
                rope_head_dim=model.rope_head_dim,
                q_lora_rank=model.q_lora_rank,
            )
            if model.kv_lora_rank > 0
            else None
        )
        return cls(
            name=model.name,
            arch=model.arch,
            architecture=ModelArchitecture(
                hidden_dim=model.hidden_dim,
                num_layers=model.num_layers,
                vocab_size=model.vocab_size,
                attention=AttentionArchitecture(
                    num_heads=model.num_heads,
                    num_kv_heads=model.num_kv_heads,
                    head_dim=model.head_dim,
                ),
                ffn=FFNArchitecture(ffn_dim=model.ffn_dim),
                moe=moe,
                mla=mla,
            ),
            quantization=quantization or QuantizationProfile.placeholder(),
        )

    # ---- flat read facade (config_plan Step F) ----
    # 让 model-graph / sizing / kv_block_allocator 直读结构化 ModelProfile, 无需
    # to_legacy() 重建 ModelConfig。每个 property 返回值与 to_legacy() 逐字段一致,
    # 故对 cost 路径 byte-identical。ModelConfig 与 ModelProfile 在这些读法上 duck-type 等价。
    @property
    def hidden_dim(self) -> int:
        return self.architecture.hidden_dim

    @property
    def num_heads(self) -> int:
        return self.architecture.attention.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.architecture.attention.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.architecture.attention.head_dim

    @property
    def ffn_dim(self) -> int:
        return self.architecture.ffn.ffn_dim

    @property
    def num_layers(self) -> int:
        return self.architecture.num_layers

    @property
    def vocab_size(self) -> int:
        return self.architecture.vocab_size

    @property
    def is_moe(self) -> bool:
        return self.architecture.moe is not None

    @property
    def num_experts(self) -> int:
        moe = self.architecture.moe
        return moe.num_experts if moe else 0

    @property
    def num_activated_experts(self) -> int:
        moe = self.architecture.moe
        return moe.num_activated_experts if moe else 0

    @property
    def expert_dim(self) -> int:
        moe = self.architecture.moe
        return moe.expert_dim if moe else 0

    @property
    def num_shared_experts(self) -> int:
        moe = self.architecture.moe
        return moe.num_shared_experts if moe else 0

    @property
    def moe_layer_freq(self) -> int:
        moe = self.architecture.moe
        return moe.moe_layer_freq if moe else 1

    @property
    def first_moe_layer(self) -> int:
        moe = self.architecture.moe
        return moe.first_moe_layer if moe else 0

    @property
    def kv_latent_dim(self) -> int:
        mla = self.architecture.mla
        return mla.kv_latent_dim if mla else 0

    @property
    def kv_lora_rank(self) -> int:
        mla = self.architecture.mla
        return mla.kv_lora_rank if mla else 0

    @property
    def v_head_dim(self) -> int:
        mla = self.architecture.mla
        return mla.v_head_dim if mla else 0

    @property
    def qk_nope_head_dim(self) -> int:
        mla = self.architecture.mla
        return mla.qk_nope_head_dim if mla else 0

    @property
    def rope_head_dim(self) -> int:
        mla = self.architecture.mla
        return mla.rope_head_dim if mla else 0

    @property
    def q_lora_rank(self) -> int:
        mla = self.architecture.mla
        return mla.q_lora_rank if mla else 0

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.is_moe:
            return False
        if layer_idx < self.first_moe_layer:
            return False
        return (layer_idx - self.first_moe_layer) % self.moe_layer_freq == 0

    def to_legacy(self) -> ModelConfig:
        a = self.architecture
        moe = a.moe
        mla = a.mla
        return ModelConfig(
            name=self.name,
            arch=self.arch,
            hidden_dim=a.hidden_dim,
            num_heads=a.attention.num_heads,
            num_kv_heads=a.attention.num_kv_heads,
            head_dim=a.attention.head_dim,
            ffn_dim=a.ffn.ffn_dim,
            num_layers=a.num_layers,
            vocab_size=a.vocab_size,
            is_moe=moe is not None,
            num_experts=moe.num_experts if moe else 0,
            num_activated_experts=moe.num_activated_experts if moe else 0,
            expert_dim=moe.expert_dim if moe else 0,
            num_shared_experts=moe.num_shared_experts if moe else 0,
            moe_layer_freq=moe.moe_layer_freq if moe else 1,
            first_moe_layer=moe.first_moe_layer if moe else 0,
            kv_latent_dim=mla.kv_latent_dim if mla else 0,
            kv_lora_rank=mla.kv_lora_rank if mla else 0,
            v_head_dim=mla.v_head_dim if mla else 0,
            qk_nope_head_dim=mla.qk_nope_head_dim if mla else 0,
            rope_head_dim=mla.rope_head_dim if mla else 0,
            q_lora_rank=mla.q_lora_rank if mla else 0,
        )
