"""Strict weight mapping and a CPU reference for native Hybrid checkpoints."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hybrid_hf_config(settings, vocab_size):
    from hf_hybrid import (
        SubmissionHybridConfig, validate_dual_residual_spec, validate_value_residual_spec,
    )

    model = settings['model']
    # Reuse the source-bound accounting preset's architecture checks without
    # importing OLMo/FLA, whose Triton imports require a GPU driver.
    from hybrid_flops import _validate_preset

    _validate_preset(settings)
    if model['type'] not in {'olmo_hybrid_scaling', 'olmo_hybrid_1b'}:
        raise ValueError('Hybrid export supports the base architecture without signals')
    if model['vocab_size'] != vocab_size or settings['training']['dtype'] != 'bfloat16':
        raise ValueError('Hybrid export requires the checkpoint vocabulary and BF16 training')
    compressed = model.get('compressed_attention')
    if compressed is not None:
        if not isinstance(compressed, dict) or compressed.get('block_size') != 128:
            raise ValueError('compressed_attention must use 128-token blocks')
        index = compressed.get('layer_index')
        if not isinstance(index, int) or not 0 <= index < model['num_layers'] or index % 4 != 3:
            raise ValueError('compressed_attention must attach to a full-attention layer')
        if compressed.get('intermediate_size') != model['intermediate_size']:
            raise ValueError('compressed intermediate size must match model intermediate size')
        scale = compressed.get('scale')
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError('compressed attention scale must be finite and in (0, 1]')
    config = SubmissionHybridConfig(
        vocab_size=vocab_size, hidden_size=model['hidden_size'],
        intermediate_size=model['intermediate_size'], num_hidden_layers=model['num_layers'],
        num_attention_heads=model['num_attention_heads'], num_key_value_heads=model['num_key_value_heads'],
        max_position_embeddings=model['context_length'], rms_norm_eps=model['rms_norm_eps'],
        tie_word_embeddings=model['tie_word_embeddings'], use_cache=False,
        pad_token_id=None, bos_token_id=None, eos_token_id=None,
        rope_parameters={'rope_type': 'default', 'rope_theta': float(model['rope_theta'])},
        layer_types=[('linear_attention' if model['block_pattern'][i % 4] == 'gdn' else 'full_attention')
                     for i in range(model['num_layers'])],
        linear_num_key_heads=model['gdn_num_heads'], linear_num_value_heads=model['gdn_num_value_heads'],
        linear_key_head_dim=model['gdn_head_dim'],
        linear_value_head_dim=int(model['gdn_head_dim'] * model['gdn_value_expand']),
        linear_conv_kernel_dim=model['gdn_conv_size'], linear_allow_neg_eigval=model['gdn_allow_neg_eigval'],
    )
    config.compressed_attention = compressed
    config.value_residual = validate_value_residual_spec(model.get('value_residual'))
    config.dual_residual = validate_dual_residual_spec(
        model.get('dual_residual'), num_layers=model['num_layers'],
    )
    return config


def convert_hybrid_state(config, state):
    """Consume every native tensor; do not silently drop or rename unknown keys."""
    common = {
        'attention_norm.weight': 'post_attention_layernorm.weight',
        'feed_forward_norm.weight': 'post_feedforward_layernorm.weight',
        'feed_forward.w1.weight': 'mlp.gate_proj.weight',
        'feed_forward.w2.weight': 'mlp.down_proj.weight',
        'feed_forward.w3.weight': 'mlp.up_proj.weight',
    }
    shared = {'embeddings.weight': 'model.embed_tokens.weight',
              'lm_head.norm.weight': 'model.norm.weight', 'lm_head.w_out.weight': 'lm_head.weight'}
    converted = {}
    consumed = set()

    def take(source, target):
        converted[target] = state[source].contiguous()
        consumed.add(source)

    for source, target in shared.items():
        take(source, target)
    for index, layer_type in enumerate(config.layer_types):
        source = f'blocks.{index}.'
        target = f'model.layers.{index}.'
        for suffix, destination in common.items():
            take(source + suffix, target + destination)
        dual = getattr(config, 'dual_residual', None)
        if dual is not None and index >= dual['start_layer']:
            take(source + 'dual_residual_mixing.post_logits', target + 'dual_residual_mixing.post_logits')
            if index == config.num_hidden_layers - 1:
                take(source + 'dual_residual_mixing.merge_logits', target + 'dual_residual_mixing.merge_logits')
        compressed = getattr(config, 'compressed_attention', None)
        augmented = compressed is not None and index == compressed['layer_index']
        mixer = 'linear_attn.' if layer_type == 'linear_attention' else 'self_attn.'
        source_mixer = 'attention.base.' if augmented else 'attention.'
        if augmented:
            mixer += 'base.'
        for native, hf in [('w_q', 'q_proj'), ('w_k', 'k_proj'), ('w_v', 'v_proj'), ('w_out', 'o_proj')]:
            take(source + f'{source_mixer}{native}.weight', target + mixer + f'{hf}.weight')
        if layer_type == 'linear_attention':
            for native, hf in [('w_a', 'a_proj'), ('w_b', 'b_proj'), ('w_g', 'g_proj'), ('o_norm', 'o_norm')]:
                take(source + f'attention.{native}.weight', target + mixer + f'{hf}.weight')
            for name in ('A_log', 'dt_bias'):
                take(source + 'attention.' + name, target + mixer + name)
            conv_keys = [source + f'attention.{name}_conv1d.weight' for name in ('q', 'k', 'v')]
            converted[target + mixer + 'conv1d.weight'] = torch.cat([state[key] for key in conv_keys], dim=0)
            consumed.update(conv_keys)
        else:
            for name in ('q_norm', 'k_norm'):
                take(source + f'{source_mixer}{name}.weight', target + mixer + f'{name}.weight')
        if augmented:
            for name in ('k_compress', 'v_compress', 'k_up', 'k_down', 'v_up', 'v_down'):
                take(source + f'attention.compressed.{name}.weight',
                     target + f'self_attn.compressed.{name}.weight')
    if set(state) != consumed:
        raise ValueError(f'unmapped native Hybrid weights: {sorted(set(state) - consumed)}')
    return converted


class NativeHybridCPUReference(nn.Module):
    """Independent, unfused native graph for conversion checks, not training.

    Uses a token-by-token delta-rule recurrence instead of HF's chunk/WY
    algorithm. GPU validation uses the actual native OLMo/FLA model instead.
    """

    def __init__(self, settings, state):
        super().__init__()
        from value_residual import validate_value_residual_spec
        from dual_residual import validate_dual_residual_spec

        self.settings = settings['model']
        self.value_residual = validate_value_residual_spec(self.settings.get('value_residual'))
        self.dual_residual = validate_dual_residual_spec(
            self.settings.get('dual_residual'), num_layers=self.settings['num_layers'],
        )
        for name, tensor in state.items():
            parent = self
            parts = name.split('.')
            for part in parts[:-1]:
                if part not in parent._modules:
                    parent.add_module(part, nn.Module())
                parent = parent._modules[part]
            parent.register_parameter(parts[-1], nn.Parameter(tensor, requires_grad=False))

    def norm(self, x, weight, eps):
        original = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return (x * weight.float()).to(original)

    def delta(self, x, state, prefix):
        cfg = self.settings
        heads, key = cfg['gdn_num_heads'], cfg['gdn_head_dim']
        value = int(key * cfg['gdn_value_expand'])
        projections = []
        for name in ('q', 'k', 'v'):
            y = F.linear(x, state[prefix + f'w_{name}.weight'])
            weight = state[prefix + f'{name}_conv1d.weight']
            y = F.silu(F.conv1d(y.transpose(1, 2), weight, groups=y.shape[-1],
                                padding=cfg['gdn_conv_size'] - 1)[..., :x.shape[1]]).transpose(1, 2)
            projections.append(y.reshape(*x.shape[:2], heads, key if name != 'v' else value))
        q, k, v = [item.float() for item in projections]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        beta = F.linear(x, state[prefix + 'w_b.weight']).sigmoid().float() * 2
        g = -state[prefix + 'A_log'].float().exp() * F.softplus(
            F.linear(x, state[prefix + 'w_a.weight']).float() + state[prefix + 'dt_bias'],
        )
        memory = x.new_zeros((x.shape[0], heads, key, value), dtype=torch.float32)
        output = []
        for t in range(x.shape[1]):
            memory = memory * g[:, t].exp()[..., None, None]
            error = v[:, t] - (memory * k[:, t, :, :, None]).sum(-2)
            memory = memory + k[:, t, :, :, None] * (error * beta[:, t, :, None])[..., None, :]
            output.append((memory * (q[:, t] / math.sqrt(key))[..., None]).sum(-2))
        y = torch.stack(output, dim=1)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + cfg['gdn_norm_eps'])
        gate = F.linear(x, state[prefix + 'w_g.weight']).reshape_as(y)
        y = (y * state[prefix + 'o_norm.weight'].float() * F.silu(gate.float())).to(x.dtype)
        return F.linear(y.flatten(2), state[prefix + 'w_out.weight'])

    def attention(self, x, state, prefix, first_value=None):
        cfg = self.settings
        q, k, v = [F.linear(x, state[prefix + f'w_{name}.weight']) for name in ('q', 'k', 'v')]
        current_value = v
        if self.value_residual is not None and first_value is not None:
            weight = self.value_residual['first_layer_weight']
            if weight == 1.0:
                v = first_value + 0.0 * v
            elif weight != 0.0:
                v = (1.0 - weight) * v + weight * first_value
        q, k = [self.norm(y, state[prefix + f'{name}_norm.weight'], cfg['rms_norm_eps'])
                for name, y in [('q', q), ('k', k)]]
        head_dim = cfg['attention_head_dim']
        q, k, v = [y.reshape(*x.shape[:2], -1, head_dim).transpose(1, 2) for y in (q, k, v)]
        frequency = 1 / (cfg['rope_theta'] ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
        positions = torch.outer(torch.arange(x.shape[1], device=x.device).float(), frequency).repeat(1, 2)
        def rotate(y):
            left, right = y.float().chunk(2, dim=-1)
            return (y.float() * positions.cos() + torch.cat((-right, left), dim=-1) * positions.sin()).to(y.dtype)
        y = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        return F.linear(y.transpose(1, 2).flatten(2), state[prefix + 'w_out.weight']), current_value

    def compressed_attention(self, x, state, prefix, spec):
        """Independent expression of the unchanged causal compressed branch."""
        batch, length, width = x.shape
        blocks = length // spec['block_size']
        if not blocks:
            return torch.zeros_like(x)
        packed = x[:, :blocks * spec['block_size']].reshape(batch, blocks, -1)
        key = F.linear(packed, state[prefix + 'k_compress.weight'])
        value = F.linear(packed, state[prefix + 'v_compress.weight'])
        key = key + F.linear(F.silu(F.linear(key, state[prefix + 'k_up.weight'])),
                             state[prefix + 'k_down.weight'])
        value = value + F.linear(F.silu(F.linear(value, state[prefix + 'v_up.weight'])),
                                 state[prefix + 'v_down.weight'])
        heads = self.settings['num_attention_heads']
        head_dim = width // heads
        query = x.reshape(batch, length, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, blocks, heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, blocks, heads, head_dim).transpose(1, 2)
        key = torch.cat((torch.zeros_like(key[:, :, :1]), key), dim=2)
        value = torch.cat((torch.zeros_like(value[:, :, :1]), value), dim=2)
        token_end = torch.arange(length, device=x.device)[:, None] + 1
        block_end = torch.arange(blocks + 1, device=x.device)[None, :] * spec['block_size']
        y = F.scaled_dot_product_attention(query, key, value, attn_mask=block_end <= token_end,
                                           dropout_p=0.0, is_causal=False)
        return y.transpose(1, 2).contiguous().reshape(batch, length, width)

    def forward(self, inputs):
        state = self.state_dict(keep_vars=True)
        cfg = self.settings
        x = F.embedding(inputs, state['embeddings.weight'])
        first_value = None
        lane1 = None
        compressed = cfg.get('compressed_attention')
        for index in range(cfg['num_layers']):
            prefix = f'blocks.{index}.'
            if cfg['block_pattern'][index % len(cfg['block_pattern'])] == 'gdn':
                mixed = self.delta(x, state, prefix + 'attention.')
            else:
                augmented = compressed is not None and index == compressed['layer_index']
                attention_prefix = prefix + ('attention.base.' if augmented else 'attention.')
                mixed, current_value = self.attention(x, state, attention_prefix, first_value)
                if self.value_residual is not None and first_value is None:
                    first_value = current_value
                if augmented:
                    mixed = mixed + compressed['scale'] * self.compressed_attention(
                        x, state, prefix + 'attention.compressed.', compressed)
            attention = self.norm(mixed, state[prefix + 'attention_norm.weight'], cfg['rms_norm_eps'])
            dual_active = self.dual_residual is not None and index >= self.dual_residual['start_layer']
            if dual_active:
                if index == self.dual_residual['start_layer']:
                    lane1 = x
                ffn_input = lane1
            else:
                x = x + attention
                ffn_input = x
            ffn = F.silu(F.linear(ffn_input, state[prefix + 'feed_forward.w1.weight'])) * F.linear(ffn_input, state[prefix + 'feed_forward.w3.weight'])
            feed_forward = self.norm(F.linear(ffn, state[prefix + 'feed_forward.w2.weight']), state[prefix + 'feed_forward_norm.weight'], cfg['rms_norm_eps'])
            if dual_active:
                weights = (2.0 * state[prefix + 'dual_residual_mixing.post_logits'].float().softmax(-1)).to(x.dtype)
                update0 = weights[0, 0] * attention + weights[0, 1] * feed_forward
                update1 = weights[1, 0] * attention + weights[1, 1] * feed_forward
                x = x + update0
                lane1 = lane1 + update1
                if index == cfg['num_layers'] - 1:
                    merge = state[prefix + 'dual_residual_mixing.merge_logits'].float().softmax(-1).to(x.dtype)
                    x = merge[0] * x + merge[1] * lane1
            else:
                x = x + feed_forward
        return F.linear(self.norm(x, state['lm_head.norm.weight'], cfg['rms_norm_eps']), state['lm_head.w_out.weight'])
