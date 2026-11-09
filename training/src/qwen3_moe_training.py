"""Use the stock Qwen3 MoE in the masked-CE DDP runner."""

from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from hf_qwen3_himoe import Qwen3HiMoeForCausalLM, grouped_topk


class Qwen3MoeTrainingConfig:
    def __init__(self, hf_config):
        # Transformers may insert defaults into its input mapping. Preserve the
        # declared architecture verbatim for family checks and exact resumes.
        self.hf = Qwen3MoeConfig.from_dict(deepcopy(hf_config))
        self.hf._attn_implementation = "sdpa"
        self.init_seed = None

    def build(self, init_device="cpu"):
        if self.init_seed is not None:
            torch.manual_seed(self.init_seed)
        with torch.device(init_device):
            return Qwen3MoeTrainingModel(
                Qwen3HiMoeForCausalLM(self.hf) if getattr(self.hf, "hi_moe", None) else
                AutoModelForCausalLM.from_config(self.hf, attn_implementation="sdpa")
            )


class Qwen3MoeTrainingModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def init_weights(self, max_seq_len, device):
        if max_seq_len > self.lm.config.max_position_embeddings:
            raise ValueError("Training context exceeds the Qwen3 MoE configuration")
        rotary = self.lm.model.rotary_emb
        inv_freq = rotary.inv_freq
        self.to(device=device, dtype=torch.bfloat16)
        if getattr(self.lm.config, "hi_moe", None):
            for layer in self.lm.model.layers:
                layer.mlp.gate.himoe_logits_ema = layer.mlp.gate.himoe_logits_ema.float()
        rotary.inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        rotary.original_inv_freq = rotary.inv_freq.clone()

    def forward(self, input_ids, *, route=False):
        if input_ids.is_cuda and self.lm.config._attn_implementation == "sdpa":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return self._forward(input_ids, route)
        return self._forward(input_ids, route)

    def _forward(self, input_ids, route):
        # Calling the backbone avoids the stock micro-batch auxiliary loss.
        output = self.lm.model(input_ids=input_ids, use_cache=False, output_router_logits=route)
        logits = self.lm.lm_head(output.last_hidden_state)
        return (logits, output.router_logits) if route else logits


def router_z_loss(router_logits):
    """FP32 mean of squared router log-partitions, over layers and tokens."""
    if not router_logits:
        raise ValueError("Router Z-loss requires logits from every MoE layer")
    return torch.stack([logits.float().logsumexp(dim=-1).square().mean()
                        for logits in router_logits]).mean()


class AccumulatedRouterBalance:
    """Qwen's running-count approximation, with separate counts per layer."""

    def __init__(self, config, device):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.himoe = getattr(config, "hi_moe", None)
        self.counts = torch.zeros(
            config.num_hidden_layers, self.num_experts, device=device, dtype=torch.int64,
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tokens = 0

    def __call__(self, router_logits):
        if len(router_logits) != self.counts.shape[0]:
            raise ValueError("Expected router logits from every MoE layer")
        counts = torch.zeros_like(self.counts)
        probability_means = []
        for layer, logits in enumerate(router_logits):
            probabilities = logits.float().softmax(dim=-1)
            probability_means.append(probabilities.mean(dim=0))
            with torch.no_grad():
                if self.himoe:
                    indices = grouped_topk(probabilities, self.himoe["groups"],
                                           self.himoe["per_group"])[1].flatten()
                else:
                    indices = probabilities.topk(self.top_k, dim=-1).indices.flatten()
                counts[layer].scatter_add_(0, indices, torch.ones_like(indices))
        if self.world_size > 1:
            dist.all_reduce(counts)
        self.counts.add_(counts)
        # Packed batches contain the same number of non-padding positions per rank.
        self.tokens += router_logits[0].shape[0] * self.world_size
        frequency = self.counts.float() / self.tokens
        means = torch.stack(probability_means)
        # Retain the stock HF top-k normalization: uniform routing gives loss K.
        return self.num_experts * (frequency * means).sum(dim=-1).mean()

    def metrics(self):
        relative_load = self.counts.flatten().float() * (
            self.num_experts / (self.tokens * self.top_k)
        )
        percentiles = torch.quantile(relative_load, relative_load.new_tensor([0.1, 0.5]))
        return {
            "train/router_load_p10_ratio": percentiles[0].item(),
            "train/router_load_median_ratio": percentiles[1].item(),
            "train/router_zero_experts": (self.counts == 0).sum().item(),
        }


def himoe_regularizers(router_logits, settings):
    """FP32 inter-group load and intra-group specialization terms."""
    inter, intra = [], []
    for logits in router_logits:
        probabilities = logits.float().softmax(dim=-1)
        selected, _ = grouped_topk(probabilities, settings["groups"], settings["per_group"])
        inter.append(selected.square().sum(dim=-1).mean())
        intra.append(probabilities.square().sum(dim=-1).mean())
    return (settings["lambda_inter"] * torch.stack(inter).mean(),
            -settings["lambda_intra"] * torch.stack(intra).mean())


def synchronize_himoe_ema(lm):
    """Average rank-local router EMAs once per optimizer update."""
    if not getattr(lm.config, "hi_moe", None) or not dist.is_initialized():
        return
    buffers = [layer.mlp.gate.himoe_logits_ema for layer in lm.model.layers]
    joined = torch.cat(buffers)
    dist.all_reduce(joined, op=dist.ReduceOp.SUM)
    joined.div_(dist.get_world_size())
    for buffer, mean in zip(buffers, joined.split([item.numel() for item in buffers])):
        buffer.copy_(mean)

