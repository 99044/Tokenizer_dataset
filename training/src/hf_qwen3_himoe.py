"""Grouped Qwen3 MoE routing with Hi-MoE's two router regularizers."""

from types import MethodType

import torch
import torch.nn.functional as F
from transformers import Qwen3MoeForCausalLM


def group_bounds(experts, groups):
    if groups < 1 or groups > experts:
        raise ValueError("invalid expert-group count")
    sizes = [experts // groups + (index < experts % groups) for index in range(groups)]
    start = 0
    for size in sizes:
        yield start, start + size
        start += size


def grouped_topk(probabilities, groups, per_group):
    values, indices = [], []
    for start, stop in group_bounds(probabilities.shape[-1], groups):
        if stop - start < per_group:
            raise ValueError("expert group is smaller than its Top-K")
        value, index = probabilities[:, start:stop].topk(per_group, dim=-1)
        values.append(value)
        indices.append(index + start)
    return torch.cat(values, dim=-1), torch.cat(indices, dim=-1)


def _grouped_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden_states, self.weight)
    settings = self.himoe_settings
    corrected = (logits.float() - settings["bias_strength"] * self.himoe_logits_ema) / settings["temperature"]
    probabilities = corrected.softmax(dim=-1)
    values, indices = grouped_topk(probabilities, settings["groups"], settings["per_group"])
    if self.norm_topk_prob:
        values = values / values.sum(dim=-1, keepdim=True)
    if self.training:
        with torch.no_grad():
            beta = settings["ema_decay"]
            self.himoe_logits_ema.mul_(beta).add_(logits.float().mean(dim=0), alpha=1 - beta)
    return corrected, values.to(logits.dtype), indices


def himoe_gates(model):
    """Yield only the routers that exist in the decoder."""
    for layer in model.model.layers:
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            yield gate


def install_himoe(model):
    settings = getattr(model.config, "hi_moe", None)
    if not settings:
        raise ValueError("missing hi_moe configuration")
    if settings["groups"] * settings["per_group"] != model.config.num_experts_per_tok:
        raise ValueError("Hi-MoE group Top-K must preserve the active expert count")
    gates = list(himoe_gates(model))
    if not gates:
        raise ValueError("Hi-MoE requires at least one routed layer")
    for gate in gates:
        gate.himoe_settings = settings
        gate.register_buffer("himoe_logits_ema", torch.zeros(model.config.num_experts, dtype=torch.float32))
        gate.forward = MethodType(_grouped_forward, gate)
    return model


class Qwen3HiMoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        install_himoe(self)


Qwen3HiMoeForCausalLM.register_for_auto_class("AutoModelForCausalLM")
