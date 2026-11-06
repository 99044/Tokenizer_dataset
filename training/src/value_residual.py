"""Fixed first-attention value residual for native OLMo-core training.

The first attention's projected V stays unchanged. Later attention layers use
``(1 - alpha) * V_current + alpha * V_first`` before head reshaping. GDN and the
compressed branch keep their implementations and parameters unchanged.

Hooks preserve the original parameter/state-dict names. A context local to one
top-level forward owns V_first and releases it in ``finally``. Autograd retains
only the graph edges needed by that forward, never a tensor from another batch.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import math

import torch
import torch.nn as nn


def validate_value_residual_spec(spec):
    """Return a normalized enabled spec, or None for the unchanged baseline."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError('model.value_residual must be a mapping')
    unknown = set(spec) - {'enabled', 'first_layer_weight'}
    if unknown:
        raise ValueError(f'unknown model.value_residual fields: {sorted(unknown)}')
    enabled = spec.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('value_residual.enabled must be a boolean')
    alpha = spec.get('first_layer_weight', 0.5)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError('value_residual.first_layer_weight must be numeric')
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError('value_residual.first_layer_weight must lie in [0, 1]')
    return {'enabled': True, 'first_layer_weight': alpha} if enabled else None


@dataclass
class _ForwardState:
    first_value: torch.Tensor | None = None
    next_attention: int = 0


class ValueResidualController:
    """Ephemeral forward state and projection hooks; this adds no parameters."""

    def __init__(self, model, projections, layer_indices, alpha):
        self.alpha = alpha
        self.layer_indices = tuple(layer_indices)
        self._state = ContextVar(f'value_residual_{id(self)}', default=None)
        self._handles = []
        original_forward = model.forward

        @wraps(original_forward)
        def forward_with_value_residual(*args, **kwargs):
            for name in ('past_key_values', 'cache_params', 'kv_cache'):
                if kwargs.get(name) is not None:
                    raise ValueError('value residual requires cache-free full-sequence forwards')
            if kwargs.get('use_cache', False):
                raise ValueError('value residual requires use_cache=False')
            with self.forward_context():
                return original_forward(*args, **kwargs)

        for ordinal, projection in enumerate(projections):
            self._handles.append(projection.register_forward_hook(self._projection_hook(ordinal)))
        # Assigning a callable does not register or wrap submodules, so all
        # checkpoint keys and optimizer parameter identities remain unchanged.
        model.forward = forward_with_value_residual

    @property
    def active(self):
        return self._state.get() is not None

    @contextmanager
    def forward_context(self):
        if self.active:
            raise RuntimeError('nested forwards of the same value-residual model are unsupported')
        state = _ForwardState()
        token = self._state.set(state)
        try:
            yield
            if state.next_attention != len(self.layer_indices):
                raise RuntimeError('not all configured attention value projections ran exactly once')
        finally:
            # This is deliberately not detached: backwards through the output
            # must still reach the first attention's V projection.
            state.first_value = None
            self._state.reset(token)

    def _projection_hook(self, ordinal):
        def hook(module, inputs, output):
            del module, inputs
            state = self._state.get()
            if state is None:
                raise RuntimeError(
                    'attention V called outside a full-model forward; '
                    'activation checkpointing and direct block calls are unsupported'
                )
            if state.next_attention != ordinal:
                raise RuntimeError('attention V projections ran in an unexpected order')
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError('attention V must have shape [batch, sequence, value_width]')
            state.next_attention += 1
            if ordinal == 0:
                state.first_value = output
                return output
            first = state.first_value
            if first is None or first.shape != output.shape:
                raise ValueError('first/current attention V shapes differ')
            if first.dtype != output.dtype or first.device != output.device:
                raise ValueError('first/current attention V dtype or device differs')
            if self.alpha == 0.0:
                return output
            if self.alpha == 1.0:
                # DDP uses find_unused_parameters=False. Keep a zero-gradient
                # edge to the current V projection at the endpoint as well.
                return first + 0.0 * output
            # Keep the same BF16 operation order in native, HF and CPU export
            # references. The first V is not detached from its autograd graph.
            return (1.0 - self.alpha) * output + self.alpha * first

        return hook


def attach_value_residual(model, settings):
    """Attach after compressed attention and before DDP/optimizer construction.

    Baseline configs without ``model.value_residual`` do not modify the model.
    This initial experiment supports cache-free, non-checkpointed full forwards.
    """
    spec = validate_value_residual_spec(settings['model'].get('value_residual'))
    if spec is None:
        return None
    if getattr(model, '_value_residual_controller', None) is not None:
        raise RuntimeError('value residual is already attached')
    if settings.get('training', {}).get('gradient_checkpointing', False):
        raise ValueError('value residual does not support gradient_checkpointing in this experiment')
    model_spec = settings['model']
    pattern = model_spec['block_pattern']
    layers = model_spec['num_layers']
    if not pattern:
        raise ValueError('value residual needs a nonempty block_pattern')
    layer_indices = [index for index in range(layers) if pattern[index % len(pattern)] == 'attn']
    if len(layer_indices) < 2:
        raise ValueError('value residual needs at least two attention layers')
    projections = []
    for index in layer_indices:
        mixer = model.blocks[str(index)].attention
        # The existing compressed branch augments the base attention; intercept
        # only base.w_v and leave compressed K/V projections untouched.
        if hasattr(mixer, 'compressed') and hasattr(mixer, 'base'):
            mixer = mixer.base
        projection = getattr(mixer, 'w_v', None)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f'attention layer {index} does not expose a linear w_v projection')
        projections.append(projection)
    if len({projection.out_features for projection in projections}) != 1:
        raise ValueError('all value-residual attention layers need equal V widths')
    if len({id(projection) for projection in projections}) != len(projections):
        raise ValueError('value residual does not support shared V projection modules')
    controller = ValueResidualController(model, projections, layer_indices, spec['first_layer_weight'])
    model._value_residual_controller = controller
    return controller
