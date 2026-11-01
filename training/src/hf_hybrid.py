"""HF adapter for the submission's post-norm OLMo Hybrid architecture.

Transformers' stock OLMo Hybrid uses pre-norm GDN blocks. Preserve the native
post-norm blocks, affine rounding, FP32 RoPE and GPU FLA kernels instead.
CPU uses Transformers' PyTorch GDN reference. GPU evaluation uses no cache.
This file is copied into each export by register_for_auto_class().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, OlmoHybridConfig
from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
    OlmoHybridForCausalLM,
    OlmoHybridAttention,
    OlmoHybridGatedDeltaNet,
    OlmoHybridMLP,
    OlmoHybridModel,
    OlmoHybridPreTrainedModel,
    OlmoHybridRMSNorm,
    OlmoHybridRotaryEmbedding,
    apply_mask_to_padding_states,
    eager_attention_forward,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class SubmissionHybridConfig(OlmoHybridConfig):
    model_type = 'scicomp26_olmo_hybrid'


class SubmissionRMSNorm(OlmoHybridRMSNorm):
    def forward(self, hidden_states):
        # Native OLMo RMSNorm applies the affine weight in FP32.
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (x * self.weight.float()).to(dtype)


class SubmissionRotaryEmbedding(OlmoHybridRotaryEmbedding):
    def forward(self, x, position_ids):
        # Native OLMo computes frequencies on the execution device. Reusing
        # HF's CPU-initialized frequencies introduces tiny CPU/CUDA pow errors
        # which later tensor-core GDN operations can amplify.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dim = self.config.hidden_size // self.config.num_attention_heads
            exponent = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
            inverse = 1.0 / (self.config.rope_parameters['rope_theta'] ** exponent)
            phase = position_ids.float().unsqueeze(-1) * inverse
            phase = torch.cat((phase, phase), dim=-1)
            return phase.cos(), phase.sin()


class SubmissionAttention(OlmoHybridAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.q_norm = SubmissionRMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = SubmissionRMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states)).view(head_shape).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states)).view(head_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = (x.unsqueeze(1).float() for x in position_embeddings)
            def rotate(x):
                a, b = x.float().chunk(2, dim=-1)
                return (x.float() * cos + torch.cat((-b, a), dim=-1) * sin).to(x.dtype)
            q, k = rotate(q), rotate(k)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if self.config._attn_implementation == 'sdpa':
            # Match native OLMo's SDPA call, including its implicit scale. Even
            # tiny attention rounding changes can grow through later GDNs.
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None and (past_key_values is None or q.shape[-2] > 1),
            ).transpose(1, 2).contiguous()
            return self.o_proj(output.reshape(*shape, -1)), None
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                        eager_attention_forward)
        output, weights = interface(self, q, k, v, attention_mask,
                                    dropout=self.attention_dropout if self.training else 0.0,
                                    scaling=self.scaling, **kwargs)
        return self.o_proj(output.reshape(*shape, -1).contiguous()), weights


class SubmissionGatedDeltaNet(OlmoHybridGatedDeltaNet):
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if not hidden_states.is_cuda:
            return super().forward(hidden_states, cache_params=cache_params,
                                   attention_mask=attention_mask, **kwargs)
        if cache_params is not None:
            raise ValueError('Native-kernel Hybrid inference requires use_cache=False')
        # Keep the exact native convolution, L2 normalization, delta-rule and
        # gated-norm rounding. The stock HF fallbacks differ even in FP32, and
        # their BF16 errors can grow materially through repeated GDN blocks.
        try:
            from fla.modules.convolution import causal_conv1d
            from fla.modules.fused_norm_gate import rms_norm_gated
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as error:
            raise RuntimeError('GPU Hybrid inference needs the reviewed FLA environment '
                               '(scicomp26:submission-env-hybrid-v3)') from error
        x = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch, length, _ = x.shape
        weights = self.conv1d.weight.split([self.key_dim, self.key_dim, self.value_dim], dim=0)
        q, k, v = [causal_conv1d(x=projection(x), weight=weight.squeeze(1), bias=None,
                                 activation='silu', backend='triton', cu_seqlens=None)[0]
                   for projection, weight in zip((self.q_proj, self.k_proj, self.v_proj), weights)]
        q = q.view(batch, length, self.num_k_heads, self.head_k_dim)
        k = k.view(batch, length, self.num_k_heads, self.head_k_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        beta = self.b_proj(x).sigmoid() * (2.0 if self.allow_neg_eigval else 1.0)
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        output, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                                           use_qk_l2norm_in_kernel=True)
        gate = self.g_proj(x).view(batch, length, self.num_v_heads, self.head_v_dim)
        output = rms_norm_gated(output, gate, self.o_norm.weight, None, 'swish',
                               eps=self.o_norm.variance_epsilon)
        return self.o_proj(output.reshape(batch, length, self.value_dim))


class PostNormGDNLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.linear_attn = SubmissionGatedDeltaNet(config, layer_idx)
        # Stock HF initializes these in FP32; native training stores them in
        # the projection dtype. Preserve checkpoint dtypes during HF reload.
        for name in ('A_log', 'dt_bias'):
            parameter = getattr(self.linear_attn, name)
            setattr(self.linear_attn, name, nn.Parameter(parameter.to(self.linear_attn.a_proj.weight.dtype)))
        self.mlp = OlmoHybridMLP(config)
        self.post_attention_layernorm = SubmissionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = SubmissionRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        # Position IDs/embeddings and use_cache are attention/model arguments,
        # not GDN kernel arguments. The cache object itself controls GDN state.
        for name in ('position_ids', 'position_embeddings', 'use_cache', 'output_attentions'):
            kwargs.pop(name, None)
        mixed = self.linear_attn(
            hidden_states, cache_params=past_key_values, attention_mask=attention_mask, **kwargs,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(mixed)
        return hidden_states + self.post_feedforward_layernorm(self.mlp(hidden_states))


class CompressedKVBranch(nn.Module):
    """Exact 128-token causal compressed branch used by the native checkpoint."""

    def __init__(self, hidden_size, num_heads, block_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.k_compress = nn.Linear(block_size * hidden_size, hidden_size, bias=False)
        self.v_compress = nn.Linear(block_size * hidden_size, hidden_size, bias=False)
        self.k_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.k_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.v_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.v_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        complete_blocks = length // self.block_size
        if complete_blocks == 0:
            return torch.zeros_like(x)
        packed = x[:, :complete_blocks * self.block_size, :].reshape(
            batch, complete_blocks, self.block_size * width,
        )
        key = self.k_compress(packed)
        value = self.v_compress(packed)
        key = key + self.k_down(F.silu(self.k_up(key)))
        value = value + self.v_down(F.silu(self.v_up(value)))
        head_dim = width // self.num_heads
        query = x.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, complete_blocks, self.num_heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, complete_blocks, self.num_heads, head_dim).transpose(1, 2)
        key = torch.cat((torch.zeros_like(key[:, :, :1]), key), dim=2)
        value = torch.cat((torch.zeros_like(value[:, :, :1]), value), dim=2)
        token_end = torch.arange(length, device=x.device)[:, None] + 1
        block_end = torch.arange(complete_blocks + 1, device=x.device)[None, :] * self.block_size
        allowed = block_end <= token_end
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False,
        )
        return output.transpose(1, 2).contiguous().reshape(batch, length, width)


class CompressedAttention(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        spec = config.compressed_attention
        self.base = base
        self.compressed = CompressedKVBranch(
            config.hidden_size, config.num_attention_heads,
            spec['block_size'], spec['intermediate_size'],
        )
        self.scale = spec['scale']

    def forward(self, hidden_states, *args, **kwargs):
        output, weights = self.base(hidden_states, *args, **kwargs)
        return output + self.scale * self.compressed(hidden_states), weights


class SubmissionHybridModel(OlmoHybridModel):
    config_class = SubmissionHybridConfig
    # Native FLA constructs its gated norm weights in FP32, even for BF16 runs.
    _keep_in_fp32_modules_strict = ['o_norm']

    def __init__(self, config):
        super().__init__(config)
        for index, layer_type in enumerate(config.layer_types):
            if layer_type == 'linear_attention':
                self.layers[index] = PostNormGDNLayer(config, index)
            else:
                layer = self.layers[index]
                layer.self_attn = SubmissionAttention(config, index)
                layer.post_attention_layernorm = SubmissionRMSNorm(config.hidden_size, config.rms_norm_eps)
                layer.post_feedforward_layernorm = SubmissionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm = SubmissionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = SubmissionRotaryEmbedding(config)
        spec = getattr(config, 'compressed_attention', None)
        if spec is not None:
            index = int(spec['layer_index'])
            if config.layer_types[index] != 'full_attention':
                raise ValueError('compressed branch must attach to a full-attention layer')
            layer = self.layers[index]
            layer.self_attn = CompressedAttention(layer.self_attn, config)
        self.post_init()


class SubmissionHybridForCausalLM(OlmoHybridForCausalLM):
    config_class = SubmissionHybridConfig
    _keep_in_fp32_modules_strict = ['o_norm']
    _no_split_modules = ['PostNormGDNLayer', 'OlmoHybridAttentionDecoderLayer']

    def __init__(self, config):
        OlmoHybridPreTrainedModel.__init__(self, config)
        self.model = SubmissionHybridModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


SubmissionHybridConfig.register_for_auto_class()
SubmissionHybridForCausalLM.register_for_auto_class('AutoModelForCausalLM')
AutoConfig.register(SubmissionHybridConfig.model_type, SubmissionHybridConfig)
AutoModelForCausalLM.register(SubmissionHybridConfig, SubmissionHybridForCausalLM)
