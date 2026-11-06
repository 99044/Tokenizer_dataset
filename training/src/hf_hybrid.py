"""HF adapter for the submission's post-norm OLMo Hybrid architecture.

Transformers' stock OLMo Hybrid uses pre-norm GDN blocks. Preserve the native
post-norm blocks, affine rounding, FP32 RoPE and GPU FLA kernels instead.
CPU uses Transformers' PyTorch GDN reference. GPU evaluation uses no cache.
This file is copied into each export by register_for_auto_class().
"""

import math
import inspect
from types import MethodType

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
    torch_chunk_gated_delta_rule,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class SubmissionHybridConfig(OlmoHybridConfig):
    model_type = 'scicomp26_olmo_hybrid'


def validate_value_residual_spec(spec):
    """Standalone parser: this file must also load from an offline HF export."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError('model.value_residual must be a mapping')
    unknown = set(spec) - {'enabled', 'first_layer_weight'}
    if unknown:
        raise ValueError(f'unknown value_residual fields: {sorted(unknown)}')
    enabled = spec.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('value_residual.enabled must be boolean')
    weight = spec.get('first_layer_weight', 0.5)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('value_residual.first_layer_weight must be finite and in [0, 1]')
    return {'enabled': True, 'first_layer_weight': float(weight)} if enabled else None


def validate_dual_residual_spec(spec, *, num_layers):
    """Keep exported configuration checks identical to the native extension."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError('model.dual_residual must be a mapping')
    unknown = set(spec) - {'enabled', 'start_layer', 'cross_mix_init'}
    if unknown:
        raise ValueError(f'unknown model.dual_residual fields: {sorted(unknown)}')
    enabled = spec.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('dual_residual.enabled must be a boolean')
    start = spec.get('start_layer', 8)
    if (isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0
            or isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < num_layers):
        raise ValueError('dual_residual.start_layer must be an integer in [0, num_layers)')
    cross = spec.get('cross_mix_init', 0.25)
    if isinstance(cross, bool) or not isinstance(cross, (int, float)) or not math.isfinite(cross) or not 0 < cross < 0.5:
        raise ValueError('dual_residual.cross_mix_init must lie strictly between 0 and 0.5')
    return dict(enabled=True, start_layer=start, cross_mix_init=float(cross)) if enabled else None


class DualResidualMixing(nn.Module):
    def __init__(self, cross_mix_init, *, final):
        super().__init__()
        probabilities = [[1 - cross_mix_init, cross_mix_init],
                         [cross_mix_init, 1 - cross_mix_init]]
        self.post_logits = nn.Parameter(torch.tensor(probabilities, dtype=torch.float32).log())
        if final:
            self.merge_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        else:
            self.register_parameter('merge_logits', None)

    def forward(self, attention_output, ffn_output):
        weights = (2.0 * self.post_logits.float().softmax(dim=-1)).to(attention_output.dtype)
        update0 = weights[0, 0] * attention_output + weights[0, 1] * ffn_output
        update1 = weights[1, 0] * attention_output + weights[1, 1] * ffn_output
        return update0, update1

    def merge(self, lane0, lane1):
        weights = self.merge_logits.float().softmax(dim=-1).to(lane0.dtype)
        return weights[0] * lane0 + weights[1] * lane1


class _DualResidualState:
    def __init__(self, first_layer, final_layer):
        self.first_layer = first_layer
        self.final_layer = final_layer
        self.active = False
        self.next_layer = first_layer
        self.lane1 = None


def _dual_layer_forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
    state = self._dual_residual_state
    if not state.active or state.next_layer != self._dual_layer_index:
        raise RuntimeError('dual-residual layers must run in order inside a full model forward')
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError('dual residual requires [batch, sequence, hidden] tensors')
    lane1 = hidden_states if state.next_layer == state.first_layer else state.lane1
    if (lane1 is None or lane1.shape != hidden_states.shape
            or lane1.dtype != hidden_states.dtype or lane1.device != hidden_states.device):
        raise ValueError('dual-residual streams must have identical shape, dtype and device')
    if hasattr(self, 'linear_attn'):
        for name in ('position_ids', 'position_embeddings', 'use_cache', 'output_attentions'):
            kwargs.pop(name, None)
        attention = self.linear_attn(
            hidden_states, cache_params=past_key_values, attention_mask=attention_mask, **kwargs,
        )
    else:
        attention, _ = self.self_attn(
            hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )
    attention = self.post_attention_layernorm(attention)
    feed_forward = self.post_feedforward_layernorm(self.mlp(lane1))
    update0, update1 = self.dual_residual_mixing(attention, feed_forward)
    lane0 = hidden_states + update0
    lane1 = lane1 + update1
    state.next_layer += 1
    if self._dual_layer_index == state.final_layer:
        state.lane1 = None
        return self.dual_residual_mixing.merge(lane0, lane1)
    state.lane1 = lane1
    return lane0


class _ValueResidualState:
    """Only live during one complete model forward; never part of state_dict."""
    def __init__(self, source_layer, weight):
        self.source_layer = source_layer
        self.weight = weight
        self.active = False
        self.first_value = None

    def mix(self, layer_idx, value):
        if not self.active:
            raise RuntimeError('value residual attention must run inside a full model forward')
        if layer_idx == self.source_layer:
            if self.first_value is not None:
                raise RuntimeError('first value projection was evaluated twice in one forward')
            self.first_value = value
            return value
        first = self.first_value
        if first is None:
            raise RuntimeError('value residual source must precede its consumers')
        if value.shape != first.shape or value.dtype != first.dtype or value.device != first.device:
            raise RuntimeError('value residual projections must have matching shape, dtype and device')
        if self.weight == 0.0:
            return value
        if self.weight == 1.0:
            # Keep the later projection in the graph for DDP's default
            # find_unused_parameters=False, with an exactly zero gradient.
            return first + 0.0 * value
        # Match native projection hooks, including BF16 operation ordering.
        return (1.0 - self.weight) * value + self.weight * first


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
        self._value_residual_state = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states)).view(head_shape).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states)).view(head_shape).transpose(1, 2)
        v = self.v_proj(hidden_states)
        if self._value_residual_state is not None:
            v = self._value_residual_state.mix(self.layer_idx, v)
        v = v.view(head_shape).transpose(1, 2)
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
    def _cpu_forward(self, hidden_states, attention_mask):
        # The public HF functions can dispatch to installed FLA/Triton kernels
        # even on CPU. Invoke the underlying PyTorch implementation locally.
        delta_rule = inspect.unwrap(torch_chunk_gated_delta_rule)
        x = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch, length, _ = x.shape
        weights = self.conv1d.weight.split([self.key_dim, self.key_dim, self.value_dim], dim=0)
        projections = []
        for projection, weight in zip((self.q_proj, self.k_proj, self.v_proj), weights):
            value = projection(x).transpose(1, 2)
            value = F.conv1d(value, weight, padding=self.conv_kernel_size - 1,
                             groups=value.shape[1])[..., :length]
            projections.append(F.silu(value).transpose(1, 2))
        q = projections[0].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        k = projections[1].reshape(batch, length, self.num_k_heads, self.head_k_dim)
        v = projections[2].reshape(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.num_k_heads:
            ratio = self.num_v_heads // self.num_k_heads
            q, k = q.repeat_interleave(ratio, dim=2), k.repeat_interleave(ratio, dim=2)
        beta = self.b_proj(x).sigmoid() * (2.0 if self.allow_neg_eigval else 1.0)
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        output, _ = delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        gate = self.g_proj(x).reshape(batch, length, self.num_v_heads, self.head_v_dim)
        normalized = output.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.o_norm.variance_epsilon,
        )
        output = (normalized * self.o_norm.weight.float() * F.silu(gate.float())).to(x.dtype)
        return self.o_proj(output.reshape(batch, length, self.value_dim))

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if not hidden_states.is_cuda:
            if cache_params is not None:
                return super().forward(hidden_states, cache_params=cache_params,
                                       attention_mask=attention_mask, **kwargs)
            return self._cpu_forward(hidden_states, attention_mask)
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
    """Causal compressed branch using the checkpoint's block size."""

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
    _keep_in_fp32_modules_strict = ['o_norm', 'dual_residual_mixing']

    def __init__(self, config):
        super().__init__(config)
        residual_spec = validate_value_residual_spec(getattr(config, 'value_residual', None))
        attention_layers = [i for i, layer_type in enumerate(config.layer_types)
                            if layer_type == 'full_attention']
        if residual_spec is not None and len(attention_layers) < 2:
            raise ValueError('value residual requires at least two full-attention layers')
        self._value_residual_state = (
            _ValueResidualState(attention_layers[0], residual_spec['first_layer_weight'])
            if residual_spec is not None else None
        )
        for index, layer_type in enumerate(config.layer_types):
            if layer_type == 'linear_attention':
                self.layers[index] = PostNormGDNLayer(config, index)
            else:
                layer = self.layers[index]
                layer.self_attn = SubmissionAttention(config, index)
                layer.self_attn._value_residual_state = self._value_residual_state
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
        dual_spec = validate_dual_residual_spec(
            getattr(config, 'dual_residual', None), num_layers=config.num_hidden_layers,
        )
        self._dual_residual_state = None
        if dual_spec is not None:
            self._dual_residual_state = _DualResidualState(
                dual_spec['start_layer'], config.num_hidden_layers - 1,
            )
            for index in range(dual_spec['start_layer'], config.num_hidden_layers):
                layer = self.layers[index]
                layer.dual_residual_mixing = DualResidualMixing(
                    dual_spec['cross_mix_init'], final=index == config.num_hidden_layers - 1,
                )
                layer._dual_residual_state = self._dual_residual_state
                layer._dual_layer_index = index
                layer.forward = MethodType(_dual_layer_forward, layer)
        self.post_init()

    def forward(self, *args, **kwargs):
        state = self._value_residual_state
        dual = self._dual_residual_state
        if state is None and dual is None:
            return super().forward(*args, **kwargs)
        if len(args) > 1:
            raise ValueError('value residual model accepts only input_ids positionally; use keyword arguments')
        use_cache = kwargs.get('use_cache')
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache or any(kwargs.get(name) is not None
                            for name in ('past_key_values', 'cache_params', 'kv_cache')):
            raise ValueError('value residual requires use_cache=False and no cached states')
        if self.training and self.is_gradient_checkpointing:
            raise ValueError('value residual does not support gradient checkpointing')
        if torch.compiler.is_compiling():
            raise RuntimeError('residual extensions require eager full-model forwards')
        if any(item is not None and item.active for item in (state, dual)):
            raise RuntimeError('nested/reentrant residual model forward is not supported')
        if state is not None:
            state.active = True
            state.first_value = None
        if dual is not None:
            dual.active = True
            dual.next_layer = dual.first_layer
            dual.lane1 = None
        try:
            result = super().forward(*args, **kwargs)
            if dual is not None and (dual.next_layer != dual.final_layer + 1 or dual.lane1 is not None):
                raise RuntimeError('all dual-residual layers must run exactly once and merge')
            return result
        finally:
            # Autograd owns tensors used by outputs; the model retains none.
            if state is not None:
                state.first_value = None
                state.active = False
            if dual is not None:
                dual.lane1 = None
                dual.active = False


class SubmissionHybridForCausalLM(OlmoHybridForCausalLM):
    config_class = SubmissionHybridConfig
    _keep_in_fp32_modules_strict = ['o_norm', 'dual_residual_mixing']
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
