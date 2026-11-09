"""Qwen3 MoE with per-layer causal windows and unchanged weight names."""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def _window_mask(attention, args, kwargs):
    # FlashAttention 2 consumes the window directly from Qwen3MoeAttention.
    if attention.config._attn_implementation == "flash_attention_2":
        return
    hidden = args[0] if args else kwargs["hidden_states"]
    mask = kwargs.get("attention_mask")
    cache = kwargs.get("past_key_values")
    past = cache.get_seq_length(attention.layer_idx) if cache is not None else 0
    queries = torch.arange(hidden.size(1), device=hidden.device) + past
    keys = torch.arange(mask.size(-1) if mask is not None else past + hidden.size(1), device=hidden.device)
    distance = queries[:, None] - keys[None, :]
    allowed = ((distance >= 0) & (distance < attention.sliding_window))[None, None]
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.masked_fill(~allowed, torch.finfo(mask.dtype).min)
    elif attention.config._attn_implementation == "eager":
        allowed = allowed if mask is None else allowed & mask
        mask = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
    else:
        mask = allowed if mask is None else allowed & mask
    kwargs["attention_mask"] = mask
    return args, kwargs


class Qwen3WindowedMoeForCausalLM(Qwen3MoeForCausalLM):
    @classmethod
    def _can_set_experts_implementation(cls):
        # The experts still come from the parent module. Transformers checks
        # the class's source file, where this attention-only subclass has no
        # expert implementation decorator of its own.
        return Qwen3MoeForCausalLM._can_set_experts_implementation()

    def __init__(self, config):
        super().__init__(config)
        window = config.local_attention_window
        global_layers = set(config.global_attention_layers)
        # Keep the backbone's shared mask causal so global layers see the
        # entire prefix. Only local layers get a second window constraint.
        if config.sliding_window is not None:
            raise ValueError("Mixed Qwen3 MoE requires the backbone's shared mask to be global")
        for index, layer in enumerate(self.model.layers):
            if index not in global_layers:
                layer.self_attn.sliding_window = window
                layer.self_attn.register_forward_pre_hook(_window_mask, with_kwargs=True)
