"""A serial prefix followed by two persistent, learned residual streams.

The sequence mixer and FFN retain OLMo's *output* normalization. Each tail
block reads the attention lane and FFN lane independently, then writes both
normalized branch outputs into both lanes. Two row-softmax mixtures, each
with total gain 2, are learned per block. The last block merges the lanes
with a learned convex combination before the unchanged LM head.

This is a full-sequence eager implementation. A context-local secondary lane
keeps the original Transformer loop and all existing parameter names intact.
It is cleared after every top-level forward, including failures; it is not a
GDN recurrent state or an inference cache. Install before DDP/optimizer and
before loading this architecture's checkpoint.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import math

import torch
import torch.nn as nn


def validate_dual_residual_spec(spec, *, num_layers):
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
    if isinstance(cross, bool) or not isinstance(cross, (int, float)):
        raise ValueError('dual_residual.cross_mix_init must be numeric')
    cross = float(cross)
    if not math.isfinite(cross) or not 0 < cross < 0.5:
        raise ValueError('dual_residual.cross_mix_init must lie strictly between 0 and 0.5')
    return dict(enabled=True, start_layer=start, cross_mix_init=cross) if enabled else None


class DualResidualMixing(nn.Module):
    """Four trainable logits per block, plus two final-merge logits once."""

    def __init__(self, cross_mix_init, *, final, device):
        super().__init__()
        probabilities = [[1 - cross_mix_init, cross_mix_init],
                         [cross_mix_init, 1 - cross_mix_init]]
        self.post_logits = nn.Parameter(torch.tensor(probabilities, device=device,
                                                    dtype=torch.float32).log())
        if final:
            self.merge_logits = nn.Parameter(torch.zeros(2, device=device, dtype=torch.float32))
        else:
            self.register_parameter('merge_logits', None)

    def forward(self, attention_output, ffn_output):
        weights = (2.0 * self.post_logits.float().softmax(dim=-1)).to(attention_output.dtype)
        update0 = weights[0, 0] * attention_output + weights[0, 1] * ffn_output
        update1 = weights[1, 0] * attention_output + weights[1, 1] * ffn_output
        return update0, update1

    def merge(self, lane0, lane1):
        if self.merge_logits is None:
            raise RuntimeError('only the final dual-residual block can merge the streams')
        weights = self.merge_logits.float().softmax(dim=-1).to(lane0.dtype)
        return weights[0] * lane0 + weights[1] * lane1


@dataclass
class _ForwardState:
    next_layer: int
    lane1: torch.Tensor | None = None


class DualResidualController:
    def __init__(self, model, layer_indices):
        self.layer_indices = tuple(layer_indices)
        self._state = ContextVar(f'dual_residual_{id(self)}', default=None)
        original_forward = model.forward

        @wraps(original_forward)
        def forward_with_dual_residual(*args, **kwargs):
            if torch.compiler.is_compiling():
                raise RuntimeError('dual residual currently requires eager full-model forwards; torch.compile is unsupported')
            for name in ('past_key_values', 'cache_params', 'kv_cache'):
                if kwargs.get(name) is not None:
                    raise ValueError('dual residual requires cache-free full-sequence forwards')
            if kwargs.get('use_cache', False):
                raise ValueError('dual residual requires use_cache=False')
            with self.forward_context():
                return original_forward(*args, **kwargs)

        for index in self.layer_indices:
            block = model.blocks[str(index)]
            block.forward = self._block_forward(block, index)
        model.forward = forward_with_dual_residual

    @property
    def active(self):
        return self._state.get() is not None

    @contextmanager
    def forward_context(self):
        if self.active:
            raise RuntimeError('nested forwards of the same dual-residual model are unsupported')
        state = _ForwardState(next_layer=self.layer_indices[0])
        token = self._state.set(state)
        try:
            yield
            if state.next_layer != self.layer_indices[-1] + 1 or state.lane1 is not None:
                raise RuntimeError('all dual-residual blocks must run exactly once and merge at the final block')
        finally:
            state.lane1 = None
            self._state.reset(token)

    def _block_forward(self, block, index):
        def forward(x, *, loss_div_factor=None, **kwargs):
            del loss_div_factor  # ReorderedNormTransformerBlock has the same behavior.
            state = self._state.get()
            if state is None:
                raise RuntimeError('dual-residual block called outside a full-model forward')
            if state.next_layer != index:
                raise RuntimeError('dual-residual blocks ran in an unexpected order')
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                raise ValueError('dual residual requires [batch, sequence, hidden] tensors')
            lane1 = x if index == self.layer_indices[0] else state.lane1
            if lane1 is None or lane1.shape != x.shape or lane1.dtype != x.dtype or lane1.device != x.device:
                raise ValueError('dual-residual streams must have identical shape, dtype and device')
            # The compressed attention branch stays inside block.attention and
            # sees the same lane as its base attention. V hooks remain intact.
            attention = block.attention_norm(block.attention(x, **kwargs))
            feed_forward = block.feed_forward_norm(block.feed_forward(lane1))
            update0, update1 = block.dual_residual_mixing(attention, feed_forward)
            lane0 = block.attention_residual_stream(x, update0)
            lane1 = block.feed_forward_residual_stream(lane1, update1)
            state.next_layer += 1
            if index == self.layer_indices[-1]:
                state.lane1 = None
                return block.dual_residual_mixing.merge(lane0, lane1)
            state.lane1 = lane1
            return lane0

        return forward


def attach_dual_residual(model, settings):
    """Attach after compressed attention, before value residual, DDP and load."""
    model_spec = settings['model']
    spec = validate_dual_residual_spec(model_spec.get('dual_residual'),
                                      num_layers=model_spec['num_layers'])
    if spec is None:
        return None
    if getattr(model, '_dual_residual_controller', None) is not None:
        raise RuntimeError('dual residual is already attached')
    if model_spec.get('normalization_style') != 'reordered_norm':
        raise ValueError('dual residual requires reordered_norm output normalization')
    if model_spec.get('dropout', 0.0) != 0.0:
        raise ValueError('dual residual currently requires dropout=0')
    training = settings.get('training', {})
    if training.get('gradient_checkpointing', False):
        raise ValueError('dual residual does not support gradient_checkpointing')
    if training.get('compile', False) or training.get('torch_compile', False):
        raise ValueError('dual residual currently requires eager execution, without torch.compile')
    indices = tuple(range(spec['start_layer'], model_spec['num_layers']))
    for index in indices:
        block = model.blocks[str(index)]
        for name in ('attention', 'attention_norm', 'feed_forward', 'feed_forward_norm',
                     'attention_residual_stream', 'feed_forward_residual_stream'):
            if not isinstance(getattr(block, name, None), nn.Module):
                raise TypeError(f'dual-residual block {index} is missing module {name}')
        if hasattr(block, 'dual_residual_mixing'):
            raise RuntimeError(f'dual-residual mixing is already registered at layer {index}')
    for index in indices:
        block = model.blocks[str(index)]
        block.dual_residual_mixing = DualResidualMixing(
            spec['cross_mix_init'], final=index == indices[-1],
            device=next(block.parameters()).device,
        )
    controller = DualResidualController(model, indices)
    model._dual_residual_controller = controller
    return controller
