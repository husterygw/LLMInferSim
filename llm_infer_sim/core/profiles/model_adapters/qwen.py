def _base_params(model_params):
    # Qwen3.5 MoE models expose text fields under text_config.
    return getattr(model_params, "text_config", model_params)

def get_num_attention_heads(model_params):
    return getattr(_base_params(model_params), "num_attention_heads")

def get_hidden_size(model_params):
    return getattr(_base_params(model_params), "hidden_size")

def get_num_key_value_heads(model_params):
    base = _base_params(model_params)
    return getattr(base, "num_key_value_heads", get_num_attention_heads(model_params))

def get_num_hidden_layers(model_params):
    return getattr(_base_params(model_params), "num_hidden_layers")

def get_intermediate_size(model_params):
    base = _base_params(model_params)
    intermediate_size = getattr(base, "intermediate_size", None)
    if intermediate_size is not None:
        return intermediate_size
    shared_expert_intermediate_size = getattr(base, "shared_expert_intermediate_size", 0)
    moe_intermediate_size = getattr(base, "moe_intermediate_size", None)
    num_experts_per_tok = getattr(base, "num_experts_per_tok", None)
    if moe_intermediate_size is not None and num_experts_per_tok is not None:
        return shared_expert_intermediate_size + moe_intermediate_size * num_experts_per_tok
    raise AttributeError("Qwen config missing intermediate_size/moe_intermediate_size fields")

def get_vocab_size(model_params):
    return getattr(_base_params(model_params), "vocab_size")
