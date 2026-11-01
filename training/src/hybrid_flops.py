"""Analytical operator ledger for the fixed OLMo Hybrid training graphs.

This is a versioned *logical* graph, not a hardware instruction profiler. FLA
chunk-64 forward and backward are decomposed separately. A scalar-op cost must
be supplied explicitly: the competition's arithmetic rules do not specify the
cost of exp, sqrt or division. Reference sources are audit inputs, never imports.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

if __package__:
    from .graph_accounting import file_hash, integer, make_template
else:  # train_ddp.py is a script launched by torchrun, not a package import.
    from graph_accounting import file_hash, integer, make_template


REFERENCE = Path(__file__).with_name('flops_reference')
FLA_COMMIT = '3cf180339b8a1cbad823f553541cd531d18670ea'
CHUNK = 64
SCALAR_BASIS = 'Explicit user-selected cost for exp, sqrt, reciprocal/division and log; not specified by the supplied competition rules.'


class Operations:
    """Keep each named operation visible instead of hiding a total multiplier."""

    def __init__(self, scalar_cost):
        self.scalar_cost = integer(scalar_cost, 'scalar_op_cost', minimum=0)
        self.items = []

    def op(self, name, kind, elements, phase='forward', calls=1, **extra):
        if elements == 0 or calls == 0:
            return
        self.items.append(dict(name=name, kind=kind, elements=elements,
                               phase=phase, calls=calls, **extra))

    def scalar(self, name, elements, phase='forward', calls=1, **extra):
        self.op(name, 'audited_scalar', elements, phase, calls,
                operation=name, flops_per_element=self.scalar_cost,
                counting_basis=SCALAR_BASIS, **extra)

    def mm(self, name, m, n, k, phase='forward', calls=1, **extra):
        self.items.append(dict(name=name, kind='matmul', m=m, n=n, k=k,
                               phase=phase, calls=calls, **extra))

    def sum(self, name, groups, width, phase='forward', calls=1):
        self.items.append(dict(name=name, kind='reduce_sum', groups=groups,
                               width=width, phase=phase, calls=calls))

    def sigmoid(self, name, elements, phase='forward', calls=1, **extra):
        # sigmoid(x) = 1 / (1 + exp(-x)); saved forward values are reused.
        self.op(name + '.negate', 'multiply', elements, phase, calls, **extra)
        self.scalar(name + '.exp', elements, phase, calls, **extra)
        self.op(name + '.add_one', 'add', elements, phase, calls, **extra)
        self.scalar(name + '.reciprocal', elements, phase, calls, **extra)

    def silu(self, name, elements, *, backward_calls=1):
        self.sigmoid(name + '.sigmoid', elements)
        self.op(name + '.value', 'multiply', elements)
        # dy * s * (1 + x * (1 - s)): 3 multiplies, 2 additions.
        self.op(name + '.gradient_product', 'multiply', elements, 'backward', 3 * backward_calls)
        self.op(name + '.gradient_add', 'add', elements, 'backward', 2 * backward_calls)

    def rms(self, name, rows, width, *, copies=1, torch_autograd=False):
        elements = rows * width
        self.op(name + '.square', 'multiply', elements, calls=copies)
        self.sum(name + '.sum_square', rows, width, calls=copies)
        self.scalar(name + '.mean', rows, calls=copies)
        self.op(name + '.epsilon', 'add', rows, calls=copies)
        self.scalar(name + '.sqrt', rows, calls=copies)
        self.scalar(name + '.reciprocal', rows, calls=copies)
        self.op(name + '.normalize_affine', 'multiply', elements, calls=2 * copies)
        # Torch's pow/mean/rsqrt DAG differs from FLA's custom RMS derivative.
        self.op(name + '.gradient_product', 'multiply', elements, 'backward', (6 if torch_autograd else 5) * copies)
        self.op(name + '.gradient_combine', 'add' if torch_autograd else 'subtract', elements, 'backward', copies)
        self.sum(name + '.gradient_inner_product', rows, width, 'backward', copies)
        self.scalar(name + '.gradient_mean', rows, 'backward', copies)
        if torch_autograd:
            # d(rsqrt(v))/dv = -0.5 * rstd^3: two products for the
            # fixed power three, scale -0.5, and multiplication by upstream.
            self.op(name + '.rsqrt_derivative', 'multiply', rows, 'backward', 4 * copies)
        self.sum(name + '.affine_weight_gradient', width, rows, 'backward', copies)


def _reference_sources():
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    sources = {}
    root = Path(__file__).resolve().parents[2]
    for name, record in manifest['files'].items():
        path = REFERENCE / name
        if file_hash(path) != record['sha256']:
            raise ValueError(f'FLOPs reference source changed: {name}')
        sources[name] = dict(path=str(path.relative_to(root)), sha256=record['sha256'],
                             role='kernel', **{'package': record['package']})
    path = REFERENCE / 'manifest.json'
    sources['reference_manifest'] = dict(path=str(path.relative_to(root)), sha256=file_hash(path))
    for name in ('training/src/hybrid_flops.py', 'training/count_flops.py'):
        sources[name] = dict(path=name, sha256=file_hash(root / name), role='accounting_code')
    return sources


def _validate_preset(settings):
    model = settings['model']
    expected = dict(
        gdn_head_dim=96, gdn_value_expand=2.0, gdn_conv_size=4,
        gdn_allow_neg_eigval=True, gdn_norm_eps=1.0e-5,
        activation='swiglu', normalization='rmsnorm',
        normalization_style='reordered_norm', rms_norm_eps=1.0e-6, qk_norm=True,
        position_encoding='rope', rope_theta=500000, context_length=8192,
        tie_word_embeddings=True, attention_bias=False, mlp_bias=False, dropout=0.0,
        attention_window_pattern=[4096, 4096, 4096, -1],
        force_full_attention_on_first_layer=False, force_full_attention_on_last_layer=True,
        block_pattern=['gdn', 'gdn', 'gdn', 'attn'],
    )
    if model['type'] in {'olmo_hybrid_1b', 'olmo_hybrid_1b_with_signals'}:
        expected.update(
            num_layers=16, hidden_size=1792, intermediate_size=7168,
            num_attention_heads=14, num_key_value_heads=14, attention_head_dim=128,
            gdn_num_heads=14, gdn_num_value_heads=14,
        )
    elif model['type'] == 'olmo_hybrid_scaling':
        if model.get('num_layers', 0) % 4:
            raise ValueError('scaling num_layers must contain complete four-layer Hybrid groups')
        if model.get('hidden_size') != model.get('num_attention_heads', 0) * model.get('attention_head_dim', 0):
            raise ValueError('scaling hidden_size must equal attention heads * head dimension')
        if not model.get('compressed_attention') and model.get('intermediate_size') != 4 * model.get('hidden_size', 0):
            raise ValueError('scaling intermediate_size must equal 4 * hidden_size')
        if model.get('num_key_value_heads') != model.get('num_attention_heads'):
            raise ValueError('scaling family requires equal attention and KV head counts')
        if model.get('gdn_num_heads') != model.get('num_attention_heads'):
            raise ValueError('scaling family requires equal GDN and attention head counts')
        if model.get('gdn_num_value_heads') != model.get('gdn_num_heads'):
            raise ValueError('scaling family requires equal GDN key and value head counts')
    else:
        raise ValueError(f'unsupported Hybrid preset {model["type"]!r}')
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(f'{name} differs from the reviewed Hybrid preset; use a reviewed custom ledger')
    if model['type'].endswith('_with_signals'):
        signal = settings.get('signal', {})
        for name, value in dict(source='gdn_prediction_error', layers_per_group=3,
                                aggregation='mean', normalization='relative_l2',
                                detach=True, gate_granularity='headwise').items():
            if signal.get(name) != value:
                raise ValueError(f'signal.{name} differs from the reviewed graph')


def expected_parameter_count(shape, vocab_size, *, signals=False):
    """Exact trainable parameter count for the reviewed tied-embedding graph."""

    layers = integer(shape.get('num_layers'), 'model_shape.num_layers')
    pattern = shape.get('block_pattern')
    if pattern != ['gdn', 'gdn', 'gdn', 'attn'] or layers % len(pattern):
        raise ValueError('parameter count requires complete Hybrid block groups')
    gdn_layers = sum(pattern[layer % len(pattern)] == 'gdn' for layer in range(layers))
    attention_layers = layers - gdn_layers
    hidden = integer(shape.get('hidden_size'), 'model_shape.hidden_size')
    intermediate = integer(shape.get('intermediate_size'), 'model_shape.intermediate_size')
    attention_heads = integer(shape.get('num_attention_heads'), 'model_shape.num_attention_heads')
    gdn_heads = integer(shape.get('gdn_num_heads'), 'model_shape.gdn_num_heads')
    value_heads = integer(shape.get('gdn_num_value_heads'), 'model_shape.gdn_num_value_heads')
    key = integer(shape.get('gdn_head_dim'), 'model_shape.gdn_head_dim')
    value = integer(shape.get('gdn_value_dim'), 'model_shape.gdn_value_dim')
    vocab = integer(vocab_size, 'padded_vocab_size')
    return (
        hidden * vocab
        + layers * 3 * hidden * intermediate + (2 * layers + 1) * hidden
        + gdn_layers * (
            hidden * (2 * gdn_heads * key + 3 * value_heads * value + 2 * value_heads)
            + 4 * (2 * gdn_heads * key + value_heads * value)
            + 2 * value_heads + value
        )
        + attention_layers * (4 * hidden * hidden + 2 * hidden)
        + (attention_layers * (hidden * attention_heads + 2 * attention_heads)
           if signals else 0)
    )


def chunk_delta_operations(batch, sequence, heads, key, value, scalar_cost):
    """FLA chunk-64 DAG; K and V remain distinct, including in backward.

    GEMMs use their logical operand shapes. Warp tiling/padding and duplicated
    tile-local scalar work are not model-graph operations. Chunk-local causal
    GEMMs are dense and masked afterwards; unlike global FlashAttention, these
    are NOT divided by two. solve_tril's actual 16x16 block algorithm is counted.
    """
    if sequence % CHUNK:
        raise ValueError('review partial-chunk semantics before changing sequence length')
    p = CHUNK
    chunks = batch * heads * (sequence // p)
    rows = batch * heads * sequence
    ops = Operations(scalar_cost)

    # q/k L2 normalization, saved normalized tensors and reciprocal std.
    ops.op('qk_l2.square', 'multiply', rows * key, calls=2)
    ops.sum('qk_l2.sum', rows, key, calls=2)
    ops.op('qk_l2.epsilon', 'add', rows, calls=2)
    ops.scalar('qk_l2.sqrt', rows, calls=2)
    ops.scalar('qk_l2.reciprocal', rows, calls=2)
    ops.op('qk_l2.scale', 'multiply', rows * key, calls=2)
    ops.op('g.local_prefix_scan', 'add', p - 1, calls=chunks)

    # A_raw = beta * K K^T * exp(g_i-g_j), then inverse(I + A_raw).
    ops.mm('kkt.score', p, key, p, calls=chunks)
    ops.op('kkt.g_difference', 'subtract', p * p, calls=chunks)
    ops.scalar('kkt.exp', p * p, calls=chunks)
    ops.op('kkt.g_and_beta', 'multiply', p * p, calls=2 * chunks)
    ops.op('inverse.diagonal_negate', 'multiply', 4 * 16 * 16, calls=chunks)
    ops.op('inverse.row_negate', 'multiply', 16, calls=4 * 14 * chunks)
    # This is an explicit product/reduction/add, not a rounded 2mnk GEMM.
    ops.op('inverse.row_product', 'multiply', 16 * 16, calls=4 * 14 * chunks)
    ops.sum('inverse.row_sum', 16, 16, calls=4 * 14 * chunks)
    ops.op('inverse.row_add', 'add', 16, calls=4 * 14 * chunks)
    ops.op('inverse.identity_add', 'add', 4 * 16 * 16, calls=chunks)
    ops.mm('inverse.off_diagonal_products', 16, 16, 16, calls=16 * chunks)
    ops.op('inverse.off_diagonal_sums', 'add', 16 * 16, calls=4 * chunks)
    ops.op('inverse.off_diagonal_negate', 'multiply', 16 * 16, calls=6 * chunks)

    # WY representation. These first forward calls are real algorithm work.
    wy_start = len(ops.items)
    ops.op('wy.beta_v', 'multiply', p * value, calls=chunks)
    ops.op('wy.beta_k_and_gate', 'multiply', p * key, calls=2 * chunks)
    ops.scalar('wy.exp_g', p, calls=chunks)
    ops.mm('wy.u', p, p, value, calls=chunks)
    ops.mm('wy.w', p, p, key, calls=chunks)
    wy_forward = ops.items[wy_start:]

    h_start = len(ops.items)
    ops.mm('state.predict', p, key, value, calls=chunks)
    ops.op('state.value_residual', 'subtract', p * value, calls=chunks)
    ops.op('state.g_difference', 'subtract', p, calls=chunks)
    ops.scalar('state.exp_difference', p, calls=chunks)
    ops.scalar('state.exp_last', 1, calls=chunks)
    ops.op('state.value_gate', 'multiply', p * value, calls=chunks)
    ops.op('state.decay', 'multiply', key * value, calls=chunks)
    ops.mm('state.update', key, p, value, calls=chunks)
    ops.op('state.add_update', 'add', key * value, calls=chunks)
    h_forward = ops.items[h_start:]

    ops.mm('output.inter_chunk', p, key, value, calls=chunks)
    ops.mm('output.qkt', p, key, p, calls=chunks)
    ops.op('output.g_difference', 'subtract', p * p, calls=chunks)
    ops.scalar('output.exp_difference', p * p, calls=chunks)
    ops.scalar('output.exp_g', p, calls=chunks)
    ops.op('output.score_gate', 'multiply', p * p, calls=chunks)
    ops.op('output.inter_gate_and_two_scales', 'multiply', p * value, calls=3 * chunks)
    ops.mm('output.local_readout', p, p, value, calls=chunks)
    ops.op('output.add_readout', 'add', p * value, calls=chunks)

    # FLA explicitly recreates w/u and h/v_new during backward to save memory.
    for operation in wy_forward + h_forward:
        ops.items.append(dict(operation, name='recompute.' + operation['name'], phase='backward',
                              exclude='activation_recomputation',
                              exclusion_basis='chunk_gated_delta_rule_bwd recreates unsaved forward intermediates w/u/h/v_new.'))
    recompute = dict(exclude='activation_recomputation',
                     exclusion_basis='Reconstruction of unsaved forward score/decay intermediates inside the backward kernel.')
    ops.mm('recompute.dv_local_qkt', p, key, p, 'backward', chunks, **recompute)
    ops.mm('recompute.dqkwg_qkt', p, key, p, 'backward', chunks, **recompute)
    ops.mm('recompute.wy_kkt', p, key, p, 'backward', chunks, **recompute)

    # Gradient of local readout; reverse recurrence propagates state gradients.
    ops.mm('dv_local.gradient', p, p, value, 'backward', chunks)
    ops.mm('reverse.dv', p, key, value, 'backward', chunks)
    ops.op('reverse.dv_gate', 'multiply', p * value, 'backward', chunks)
    ops.op('reverse.dv_local_add', 'add', p * value, 'backward', chunks)
    ops.op('reverse.q_gate', 'multiply', p * key, 'backward', chunks)
    ops.mm('reverse.dh_q', key, p, value, 'backward', chunks)
    ops.mm('reverse.dh_w', key, p, value, 'backward', chunks)
    ops.op('reverse.state_decay_and_scale', 'multiply', key * value, 'backward', 2 * chunks)
    ops.op('reverse.state_subtract_add', 'add', key * value, 'backward', 2 * chunks)

    # dq, dk, dw and the gate derivative; no blanket factor three for GDN.
    ops.mm('dqkwg.ds', p, value, p, 'backward', chunks)
    for name in ('dq', 'dk', 'dw'):
        ops.mm('dqkwg.' + name, p, value, key, 'backward', chunks)
    ops.op('dqkwg.dw_negate', 'multiply', p * key, 'backward', chunks)
    ops.op('dqkwg.state_inner_product', 'multiply', key * value, 'backward', chunks)
    ops.sum('dqkwg.state_sum', 1, key * value, 'backward', chunks)
    ops.op('dqkwg.last_gate_gradient', 'multiply', 1, 'backward', chunks)
    ops.op('dqkwg.dq_gate_scale', 'multiply', p * key, 'backward', 2 * chunks)
    ops.op('dqkwg.dk_gate', 'multiply', p * key, 'backward', chunks)
    ops.op('dqkwg.dg_qk_products', 'multiply', p * key, 'backward', 3 * chunks)
    ops.sum('dqkwg.dg_token_sums', p, key, 'backward', 2 * chunks)
    ops.sum('dqkwg.dg_last_sum', 1, p * key, 'backward', chunks)
    ops.op('dqkwg.dg_subtract', 'subtract', p, 'backward', chunks)
    ops.op('dqkwg.dg_last_add', 'add', 1, 'backward', chunks)
    ops.op('dqkwg.ds_gate_scale', 'multiply', p * p, 'backward', 2 * chunks)
    ops.op('dqkwg.dg_score_product', 'multiply', p * p, 'backward', chunks)
    ops.sum('dqkwg.dg_score_row_column', p, p, 'backward', 2 * chunks)
    ops.op('dqkwg.dg_score_accumulate', 'add', p, 'backward', 2 * chunks)
    ops.mm('dqkwg.dq_local', p, p, key, 'backward', chunks)
    ops.mm('dqkwg.dk_local', p, p, key, 'backward', chunks)
    ops.op('dqkwg.dq_dk_accumulate', 'add', p * key, 'backward', 2 * chunks)
    ops.op('dqkwg.last_token_insert', 'add', p, 'backward', chunks)

    # WY derivatives and inverse derivative -A^T dA A^T (both GEMMs).
    ops.mm('wy_backward.dA_w', p, key, p, 'backward', chunks)
    ops.mm('wy_backward.dkbg', p, p, key, 'backward', chunks)
    ops.mm('wy_backward.dA_u', p, value, p, 'backward', chunks)
    ops.mm('wy_backward.dvb', p, p, value, 'backward', chunks)
    ops.op('wy_backward.dA_add', 'add', p * p, 'backward', chunks)
    ops.op('wy_backward.dk_beta_gate_coefficient', 'multiply', p, 'backward', chunks)
    ops.op('wy_backward.dk_beta_gate', 'multiply', p * key, 'backward', chunks)
    ops.op('wy_backward.db_from_w', 'multiply', p * key, 'backward', 2 * chunks)
    ops.sum('wy_backward.db_from_w_sum', p, key, 'backward', chunks)
    ops.op('wy_backward.dg_from_w', 'multiply', p * key, 'backward', chunks)
    ops.sum('wy_backward.dg_from_w_sum', p, key, 'backward', chunks)
    ops.op('wy_backward.dv_beta', 'multiply', p * value, 'backward', chunks)
    ops.op('wy_backward.db_from_u', 'multiply', p * value, 'backward', chunks)
    ops.sum('wy_backward.db_from_u_sum', p, value, 'backward', chunks)
    ops.op('wy_backward.db_accumulate', 'add', p, 'backward', 2 * chunks)
    ops.mm('inverse_backward.right', p, p, p, 'backward', chunks)
    ops.mm('inverse_backward.left', p, p, p, 'backward', chunks)
    ops.op('inverse_backward.gate_and_negate', 'multiply', p * p, 'backward', 2 * chunks)
    ops.mm('kkt_backward.left', p, p, key, 'backward', chunks)
    ops.mm('kkt_backward.right', key, p, p, 'backward', chunks)
    ops.op('kkt_backward.key_beta', 'multiply', p * key, 'backward', chunks)
    ops.op('kkt_backward.db_product', 'multiply', p * key, 'backward', chunks)
    ops.sum('kkt_backward.db_sum', p, key, 'backward', chunks)
    ops.op('kkt_backward.db_add', 'add', p, 'backward', chunks)
    ops.op('kkt_backward.dk_beta', 'multiply', p * key, 'backward', chunks)
    ops.op('kkt_backward.dk_additions', 'add', p * key, 'backward', 2 * chunks)
    ops.op('kkt_backward.dg_product', 'multiply', p * p, 'backward', chunks)
    ops.sum('kkt_backward.dg_row_column', p, p, 'backward', 2 * chunks)
    ops.op('kkt_backward.dg_difference_add', 'add', p, 'backward', 2 * chunks)
    ops.op('combine.dk', 'add', rows * key, 'backward')
    ops.op('combine.dg', 'add', rows, 'backward')
    # Scalar reverse scan is implemented as prefix + sum - prefix + input.
    ops.op('g.reverse_prefix', 'add', p - 1, 'backward', chunks)
    ops.sum('g.reverse_sum', 1, p, 'backward', chunks)
    ops.op('g.reverse_negate', 'multiply', p, 'backward', chunks)
    ops.op('g.reverse_adds', 'add', p, 'backward', 2 * chunks)
    ops.op('qk_l2.gradient_product', 'multiply', rows * key, 'backward', 8)
    ops.op('qk_l2.gradient_subtract', 'subtract', rows * key, 'backward', 2)
    ops.sum('qk_l2.gradient_inner_product', rows, key, 'backward', 2)
    return ops.items


def convolution_operations(tokens, batch, sequence, channels, kernel, scalar_cost):
    ops = Operations(scalar_cost)
    ops.op('depthwise.forward', 'fma', tokens * channels, calls=kernel)
    # Distinct input and weight derivatives, with shared-weight reduction.
    ops.op('depthwise.input_gradient', 'fma', tokens * channels, 'backward', kernel)
    ops.op('depthwise.weight_gradient_product', 'multiply', tokens * channels, 'backward', kernel)
    ops.sum('depthwise.weight_gradient_sum', channels * kernel, tokens, 'backward')
    ops.silu('depthwise.silu', tokens * channels, backward_calls=kernel)
    # The Triton backward applies the SiLU derivative separately for each tap.
    # Its sigmoid is a recreated forward activation; preactivation conv also
    # gets recreated explicitly by causal_conv1d_bwd, and both are excluded.
    recompute = dict(exclude='activation_recomputation',
                     exclusion_basis='causal_conv1d_bwd recreates preactivation convolution and sigmoid values, not their derivatives.')
    ops.op('recompute.convolution', 'fma', tokens * channels, 'backward', kernel, **recompute)
    ops.sigmoid('recompute.silu_sigmoid', tokens * channels, 'backward', kernel, **recompute)
    return ops.items


def gdn_elementwise(tokens, heads, value, scalar_cost):
    ops = Operations(scalar_cost)
    h = tokens * heads
    ops.sigmoid('beta.sigmoid', h)
    ops.op('beta.allow_negative_eigenvalue', 'multiply', h)
    ops.op('beta.gradient_scale', 'multiply', h, 'backward')
    ops.op('beta.sigmoid_gradient_products', 'multiply', h, 'backward', 2)
    ops.op('beta.sigmoid_gradient_subtract', 'subtract', h, 'backward')
    # -exp(A_log) * softplus(w_a(x) + dt_bias). softplus has a
    # value-dependent threshold: retain its unbranched logical definition.
    ops.scalar('decay.exp_A', heads)
    ops.op('decay.negate_A', 'multiply', heads)
    ops.op('decay.dt_bias', 'add', h)
    ops.scalar('decay.softplus_exp', h)
    ops.op('decay.softplus_add_one', 'add', h)
    ops.scalar('decay.softplus_log', h)
    ops.op('decay.product', 'multiply', h)
    ops.op('decay.da_gradient_product', 'multiply', h, 'backward')
    ops.sigmoid('decay.softplus_gradient_sigmoid', h, 'backward')
    ops.op('decay.softplus_gradient', 'multiply', h, 'backward')
    ops.op('decay.dA_gradient_product', 'multiply', h, 'backward')
    ops.sum('decay.dA_token_sum', heads, tokens, 'backward')
    ops.op('decay.dA_exp_gradient', 'multiply', heads, 'backward')
    ops.op('decay.dA_negate_gradient', 'multiply', heads, 'backward')
    ops.sum('decay.dt_bias_gradient', heads, tokens, 'backward')
    ops.rms('output_rms', h, value)
    ops.sigmoid('output_gate.sigmoid', h * value)
    ops.op('output_gate.value', 'multiply', h * value, calls=2)
    ops.op('output_gate.gradient_products', 'multiply', h * value, 'backward', 6)
    ops.op('output_gate.gradient_additions', 'add', h * value, 'backward', 2)
    return ops.items


def signal_scan_other(tokens, heads, key, value, scalar_cost):
    ops = Operations(scalar_cost)
    rows = tokens * heads
    # F.normalize uses sqrt(sum(k^2)), clamp_min(eps), then division.
    ops.op('signal.key_square', 'multiply', rows * key)
    ops.sum('signal.key_sum', rows, key)
    ops.scalar('signal.key_sqrt', rows)
    ops.scalar('signal.key_divide', rows * key)
    ops.scalar('signal.decay_exp', rows)
    ops.op('signal.error_and_value_square', 'multiply', rows * value, calls=2)
    ops.sum('signal.error_and_value_sum', rows, value, calls=2)
    ops.scalar('signal.error_and_value_sqrt', rows, calls=2)
    ops.scalar('signal.relative_error_divide', rows)
    ops.sum('signal.head_sum', tokens, heads)
    ops.scalar('signal.head_mean', tokens)
    return ops.items


def complete_hybrid_ledger(model_config, settings, micro_batch_size, vocab_size, scalar_cost, *, base_only=False):
    """Generate all components automatically; no pending template is returned."""
    if settings['model'].get('compressed_attention') and not base_only:
        raise ValueError('compressed attention requires an expanded operator ledger')
    _validate_preset(settings)
    ledger = make_template(model_config, settings, micro_batch_size, vocab_size)
    ledger['source_files'].update(_reference_sources())
    model = settings['model']
    batch, sequence = micro_batch_size, model['context_length']
    tokens, hidden, intermediate = batch * sequence, model['hidden_size'], model['intermediate_size']
    heads, key, value = model['gdn_num_heads'], model['gdn_head_dim'], int(model['gdn_head_dim'] * model['gdn_value_expand'])
    signals = model['type'].endswith('_with_signals')
    by_id = {component['id']: component for component in ledger['components']}
    evidence = list(ledger['source_files'])

    def set_ops(identity, operations, basis, *, forward_only=False):
        component = by_id[identity]
        component.update(status='reviewed', operations=operations, evidence=evidence,
                         counting_basis=basis, forward_only=forward_only)
        if not operations:
            component['zero_cost_justification'] = basis

    embedding = Operations(scalar_cost)
    embedding.op('embedding.gradient_scatter_add', 'add', tokens * hidden, 'backward')
    embedding.op('tied_embedding_output_gradient_add', 'add', vocab_size * hidden, 'backward')
    set_ops('embedding', embedding.items, 'Embedding lookup/copies are zero FLOPs; count backward scatter and tied vocabulary-weight gradient addition.')
    output = Operations(scalar_cost)
    output.rms('lm_head.norm', tokens, hidden, torch_autograd=True)
    set_ops('output.elementwise', output.items, 'Affine output RMSNorm; softmax cross-entropy is a training loss rather than model prediction and is outside the model graph.')
    for component in ledger['components']:
        if component['id'].endswith('.gemm'):
            component.update(status='reviewed', evidence=evidence,
                             counting_basis='Bias-free projections from the fixed preset. Forward, input gradient and trainable weight gradient are separate 2mnk operations; weights may be shared but calls are not deduplicated.')

    for layer in range(model['num_layers']):
        prefix = f'layer.{layer}'
        ffn = Operations(scalar_cost)
        ffn.silu('swiglu.silu', tokens * intermediate)
        ffn.op('swiglu.product', 'multiply', tokens * intermediate)
        ffn.op('swiglu.product_gradients', 'multiply', tokens * intermediate, 'backward', 2)
        ffn.op('swiglu.input_gradient_add', 'add', tokens * hidden, 'backward')
        set_ops(prefix + '.ffn.elementwise', ffn.items, 'SwiGLU and gradient products; add the two input-projection gradients once.')
        residual = Operations(scalar_cost)
        residual.rms('reordered_norm', tokens, hidden, copies=2, torch_autograd=True)
        residual.op('residual.forward', 'add', tokens * hidden, calls=2)
        residual.op('residual.backward_branch_accumulate', 'add', tokens * hidden, 'backward', 2)
        set_ops(prefix + '.norm_residual', residual.items, 'Two post-mixer/post-FFN RMSNorms and two residual connections, with separate backward branch sums.')
        if model['block_pattern'][layer % 4] == 'gdn':
            component = by_id[prefix + '.gdn.gemm']
            component['operations'].append(dict(name='projection.input_gradient_sum', kind='add',
                                                phase='backward', elements=tokens * hidden, calls=5))
            # Six input projections (q,k,v,a,b,g) each receive an input grad.
            conv = convolution_operations(tokens, batch, sequence, heads * (2 * key + value), model['gdn_conv_size'], scalar_cost)
            set_ops(prefix + '.gdn.convolution', conv, 'Three independent depthwise causal convolutions; every tap and its SiLU derivative are counted. Bias/state/CP are absent; shared convolution weights are reduced across tokens.')
            set_ops(prefix + '.gdn.delta_rule', chunk_delta_operations(batch, sequence, heads, key, value, scalar_cost), 'Pinned FLA chunk-64 WY/state/output DAG and separately derived backward; logical K and V dimensions remain distinct. Memory-saving reconstruction is explicitly excluded.')
            for suffix, entry in [('convolution', 'causal_conv1d_fwd/causal_conv1d_bwd'), ('delta_rule', 'chunk_gated_delta_rule_fwd/chunk_gated_delta_rule_bwd')]:
                by_id[prefix + '.gdn.' + suffix]['implementation'] = dict(package_version_or_commit=FLA_COMMIT, entrypoint=entry, source_role='accounting reference; runtime match checked separately')
            set_ops(prefix + '.gdn.elementwise', gdn_elementwise(tokens, heads, value, scalar_cost), 'Beta sigmoid/scale, decay and gated RMSNorm, with logical derivatives and shared parameter reductions. Softplus is counted by its unbranched exp/add/log definition.')
            if signals:
                component = by_id[prefix + '.signal.scan_arithmetic']
                set_ops(component['id'], component['operations'], 'A second algorithmic recurrence, once per head/token: decay, prediction, residual and delta update. detach=True gives no backward; this scan is never a recomputation exclusion.', forward_only=True)
                set_ops(prefix + '.signal.scan_other', signal_scan_other(tokens, heads, key, value, scalar_cost), 'Independent key normalization, exp decay, relative L2 error and head mean. Clamp/comparison/indexing perform no floating-point arithmetic.', forward_only=True)
        else:
            attn_heads, dimension = model['num_attention_heads'], model['attention_head_dim']
            core = Operations(scalar_cost)
            # Megatron's causal S^2/2 convention; do not apply it to GDN chunks.
            for phase, count in [('forward', 2), ('backward', 4)]:
                core.mm('causal_qkt_and_av.' + phase, sequence, dimension, sequence, phase,
                        calls=count * batch * attn_heads, causal_reference_half=True)
            set_ops(prefix + '.attn.core', core.items, 'Megatron causal-attention convention: each SxS GEMM contributes half its dense 2mnk cost; all four attention layers are global. QK/AV gradients are separate calls.')
            elem = Operations(scalar_cost)
            elem.rms('qk_norm', tokens, hidden, copies=2, torch_autograd=True)  # use_head_qk_norm=False.
            elem.op('rope.qk_products', 'multiply', tokens * hidden, calls=4)
            elem.op('rope.qk_sums', 'add', tokens * hidden, calls=2)
            elem.op('rope.rotate_negate', 'multiply', tokens * hidden // 2, calls=2)
            elem.op('rope.gradient_products', 'multiply', tokens * hidden, 'backward', 4)
            elem.op('rope.gradient_sums', 'add', tokens * hidden, 'backward', 2)
            elem.op('rope.gradient_rotate_negate', 'multiply', tokens * hidden // 2, 'backward', 2)
            rows = batch * attn_heads * sequence
            pairs = batch * attn_heads * sequence * sequence // 2
            elem.op('attention.scale', 'multiply', pairs)
            elem.op('softmax.subtract_max', 'subtract', pairs)
            elem.scalar('softmax.exp', pairs)
            elem.op('softmax.sum', 'add', pairs - rows)
            elem.scalar('softmax.divide', pairs)
            elem.op('softmax.gradient_products', 'multiply', pairs, 'backward', 2)
            elem.op('softmax.gradient_sum', 'add', pairs - rows, 'backward')
            elem.op('softmax.gradient_subtract', 'subtract', pairs, 'backward')
            elem.op('attention.gradient_scale', 'multiply', pairs, 'backward')
            elem.op('projection.input_gradient_add', 'add', tokens * hidden, 'backward', 3 if signals else 2)
            if signals:
                gates = tokens * attn_heads
                elem.sigmoid('head_gate.sigmoid', gates)
                elem.op('head_gate.apply', 'multiply', tokens * hidden)
                elem.op('head_gate.input_and_gate_gradients', 'multiply', tokens * hidden, 'backward', 2)
                elem.sum('head_gate.reduce_feature_gradient', gates, dimension, 'backward')
                elem.op('head_gate.sigmoid_gradient_products', 'multiply', gates, 'backward', 2)
                elem.op('head_gate.sigmoid_gradient_subtract', 'subtract', gates, 'backward')
                gate = Operations(scalar_cost)
                gate.op('signal_scale.gradient_product', 'multiply', gates, 'backward')
                gate.sum('signal_scale.gradient_sum', attn_heads, tokens, 'backward')
                gate.sum('signal_bias.gradient_sum', attn_heads, tokens, 'backward')
                set_ops(prefix + '.signal.gate_backward', gate.items, 'Detached rho has no gradient; signal_scale and signal_bias remain trainable with token-wise reductions.')
                mean = Operations(scalar_cost)
                mean.op('signal_accumulator.two_adds', 'add', tokens, calls=2)
                mean.scalar('signal_accumulator.divide_three', tokens)
                set_ops(prefix + '.signal.group_mean', mean.items, 'Three GDN scalar signals are summed using two additions and divided by three; no backward.', forward_only=True)
                component = by_id[prefix + '.signal.gate_forward']
                set_ops(component['id'], component['operations'], 'Wx + rho * signal_scale + signal_bias. Projection and sigmoid are accounted for separately.', forward_only=True)
            set_ops(prefix + '.attn.elementwise', elem.items, 'QK RMSNorm, RoPE, causal softmax, scaling and optional head gates. Softmax uses the same S^2/2 convention as the Megatron attention GEMMs, without hardware tile-padding/recomputation overhead.')
    ledger.update(status='complete', calculation_kind='versioned_logical_reference_graph',
                  runtime_kernel_verified=False, scalar_op_cost=scalar_cost,
                  conventions=dict(chunk_size=CHUNK, fla_commit=FLA_COMMIT,
                                   scalar_cost_basis=SCALAR_BASIS,
                                   causal_attention='Megatron S^2/2; GDN local GEMMs stay dense',
                                   gradient_scope='all model parameters trainable; no frozen projections, CP or initial/final state',
                                   loss_and_diagnostic_statistics='outside the prediction graph',
                                   hardware_tile_padding='excluded from logical tensor dimensions'),
                  notes=['Complete analytical reference, not an official runtime FLOPs certificate.',
                         'Verify runtime kernel source hashes and model metadata before treating this as actual training accounting.',
                         'Repeated model calls are counted. Only memory-saving backward reconstruction, optimizer and communication are excluded.',
                         'Saved forward activations may be reused in logical derivatives; hardware-local duplicates are not new model operators.'])
    return ledger


def runtime_graph_metadata(model):
    """Read-only metadata; called after construction, outside the training loop."""
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    roots = {}
    for package in ('fla', 'olmo_core'):
        spec = importlib.util.find_spec(package)
        roots[package] = Path(spec.origin).parent if spec is not None and spec.origin else None
    files = {}
    for name, record in manifest['files'].items():
        package_root = roots[record['package']]
        path = package_root / record['relative_path'] if package_root else None
        files[name] = file_hash(path) if path is not None and path.is_file() else None
    convolutions = [dict(name=name, backend=getattr(module, 'backend', None),
                         activation=module.activation, bias=module.bias is not None,
                         channels=module.in_channels, kernel_size=list(module.kernel_size),
                         cp_enabled=module.cp_enabled)
                    for name, module in model.named_modules()
                    if type(module).__name__ == 'CausalConv1d']
    linears = {}
    for module in model.modules():
        if any(base.__name__ == 'Linear' for base in type(module).__mro__):
            key = f'{module.in_features}:{module.out_features}'
            linears[key] = linears.get(key, 0) + 1
    native = ['training/src/model_config.py', 'training/src/train_ddp.py']
    if (Path(__file__).with_name('signal_hybrid.py')).is_file():
        native.append('training/src/signal_hybrid.py')
    return dict(schema='scicomp26-runtime-graph-v1',
                all_parameters_trainable=all(parameter.requires_grad for parameter in model.parameters()),
                parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                linear_shapes=linears,
                source_sha256=files, reference_fla_commit=FLA_COMMIT,
                convolution_modules=convolutions,
                environment=dict(FLA_CONV_BACKEND=os.environ.get('FLA_CONV_BACKEND', 'triton')),
                project_graph_sha256={name: file_hash(Path(__file__).resolve().parents[2] / name)
                                      for name in native})


def verify_runtime_metadata(path, ledger):
    metadata = json.loads(Path(path).read_text())
    if metadata.get('schema') != 'scicomp26-runtime-graph-v1' or metadata.get('all_parameters_trainable') is not True:
        raise ValueError('runtime evidence must confirm the reviewed schema and all-trainable graph')
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        if metadata.get('source_sha256', {}).get(name) != record['sha256']:
            raise ValueError(f'runtime kernel differs from the counting reference: {name}; use a reviewed custom ledger')
    project_sources = metadata.get('project_graph_sha256', {})
    required_project = {'training/src/model_config.py', 'training/src/train_ddp.py'}
    if ledger['model_type'].endswith('_with_signals'):
        required_project.add('training/src/signal_hybrid.py')
    if not required_project <= project_sources.keys():
        raise ValueError('runtime model source evidence is missing')
    for name, expected in project_sources.items():
        if name not in ledger['source_files'] or ledger['source_files'][name]['sha256'] != expected:
            # Base Hybrid also records the unused signal module; only require it
            # for the signals architecture, where it participates in prediction.
            if name.endswith('signal_hybrid.py') and ledger['model_type'] == 'olmo_hybrid_1b':
                continue
            raise ValueError(f'runtime model/runner source differs: {name}')
    shape = ledger.get('model_shape', {})
    layers = integer(shape.get('num_layers'), 'model_shape.num_layers')
    pattern = shape.get('block_pattern')
    if pattern != ['gdn', 'gdn', 'gdn', 'attn'] or layers % len(pattern):
        raise ValueError('runtime Hybrid block pattern differs from the reviewed graph')
    gdn_layers = sum(pattern[layer % len(pattern)] == 'gdn' for layer in range(layers))
    attention_layers = layers - gdn_layers
    hidden = integer(shape.get('hidden_size'), 'model_shape.hidden_size')
    intermediate = integer(shape.get('intermediate_size'), 'model_shape.intermediate_size')
    heads = integer(shape.get('gdn_num_heads'), 'model_shape.gdn_num_heads')
    value_heads = integer(shape.get('gdn_num_value_heads'), 'model_shape.gdn_num_value_heads')
    key = integer(shape.get('gdn_head_dim'), 'model_shape.gdn_head_dim')
    value = integer(shape.get('gdn_value_dim'), 'model_shape.gdn_value_dim')
    conv = metadata.get('convolution_modules', [])
    if len(conv) != 3 * gdn_layers or any(module.get('backend') != 'triton' or module.get('activation') not in {'silu', 'swish'} or module.get('bias') is not False or module.get('cp_enabled') is not False or module.get('kernel_size') != [4] for module in conv):
        raise ValueError('runtime convolution graph differs from the reviewed bias-free Triton convolutions')
    expected_linears = {}
    def linear(input_width, output_width, copies):
        key = f'{input_width}:{output_width}'
        expected_linears[key] = expected_linears.get(key, 0) + copies
    linear(hidden, intermediate, 2 * layers)
    linear(intermediate, hidden, layers)
    linear(hidden, heads * key, 2 * gdn_layers)
    linear(hidden, value_heads * value, 2 * gdn_layers)
    linear(value_heads * value, hidden, gdn_layers)
    linear(hidden, value_heads, 2 * gdn_layers)
    linear(hidden, hidden, 4 * attention_layers)
    linear(hidden, ledger['batch_shape']['padded_vocab_size'], 1)
    signals = ledger['model_type'].endswith('_with_signals')
    if signals:
        linear(hidden, shape['num_attention_heads'], attention_layers)
    if metadata.get('linear_shapes') != expected_linears:
        raise ValueError('runtime linear projections differ from the reviewed graph')
    expected_parameters = expected_parameter_count(
        shape,
        ledger['batch_shape']['padded_vocab_size'],
        signals=signals,
    )
    if metadata.get('parameter_count') != expected_parameters:
        raise ValueError('runtime parameter coverage differs from the reviewed all-trainable graph')
    from collections import Counter
    expected_channels = Counter()
    expected_channels[heads * key] += 2 * gdn_layers
    expected_channels[value_heads * value] += gdn_layers
    if Counter(module.get('channels') for module in conv) != expected_channels:
        raise ValueError('runtime convolution channel dimensions differ from the reviewed graph')
    return dict(path=str(path), sha256=file_hash(path), kernel_sources_match=True,
                all_parameters_trainable=True, scalar_convention_is_officially_specified=False,
                first_optimizer_step=integer(metadata.get('first_optimizer_step'), 'runtime.first_optimizer_step'),
                planned_stop_step=integer(metadata.get('planned_stop_step'), 'runtime.planned_stop_step'))


def verify_runtime_history(paths, ledger, checkpoint_step):
    """Require source evidence for the initial run AND every resumed segment."""
    records = sorted((verify_runtime_metadata(path, ledger) for path in paths),
                     key=lambda record: record['first_optimizer_step'])
    starts = [record['first_optimizer_step'] for record in records]
    if not starts or starts[0] != 1 or len(set(starts)) != len(starts):
        raise ValueError('runtime evidence must begin at step 1 and identify distinct resumed segments')
    if starts[-1] > checkpoint_step:
        raise ValueError('runtime evidence contains a segment after the selected checkpoint')
    for index, record in enumerate(records):
        last = starts[index + 1] - 1 if index + 1 < len(records) else checkpoint_step
        if last > record['planned_stop_step']:
            raise ValueError('runtime evidence does not cover the selected checkpoint history')
        record['accounted_through_step'] = last
    return records
