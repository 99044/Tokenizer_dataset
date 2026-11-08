"""Token-based FP32 EMA, maintained on rank zero without changing training weights."""

import math

import torch


def resolve_ema_config(config, max_steps, tokens_per_step):
    if not config or not config.get('enabled', False):
        return None
    fraction = config.get('start_fraction', 0.7)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
        raise ValueError('ema.start_fraction must be in [0, 1)')
    lives = config.get('half_life_tokens', [20_000_000, 80_000_000])
    if not isinstance(lives, list) or not lives or any(
        isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in lives
    ) or len(set(lives)) != len(lives):
        raise ValueError('ema.half_life_tokens must contain unique positive integers')
    if max_steps < 1 or tokens_per_step < 1:
        raise ValueError('EMA needs positive step and token counts')
    return dict(start_step=math.floor(fraction * max_steps) + 1,
                tokens_per_step=tokens_per_step,
                variants={f'ema_{life}': dict(half_life_tokens=life,
                          decay=math.exp(-math.log(2) * tokens_per_step / life)) for life in lives})


class WeightEMA:
    def __init__(self, model, specification, master_weights=None):
        self.specification = specification
        self.source = 'fp32_master' if master_weights is not None else 'model'
        self.parameters = dict(model.named_parameters())
        canonical = {id(p): name for name, p in self.parameters.items()}
        self.aliases = {name: canonical[id(p)] for name, p in model.named_parameters(remove_duplicate=False)}
        masters = {} if master_weights is None else master_weights.parameter_map
        self.sources = {name: masters.get(id(p), p) for name, p in self.parameters.items()}
        self.shadows = {}
        self.last_step = 0
        self.updates = 0

    @torch.no_grad()
    def update(self, completed_step):
        if completed_step != self.last_step + 1:
            raise ValueError('EMA steps must be consecutive; restore EMA when resuming')
        self.last_step = completed_step
        if completed_step < self.specification['start_step']:
            return
        if not self.shadows:
            self.shadows = {variant: {name: p.detach().float().clone() for name, p in self.sources.items()}
                            for variant in self.specification['variants']}
        else:
            for variant, shadow in self.shadows.items():
                decay = self.specification['variants'][variant]['decay']
                for name, p in self.sources.items():
                    shadow[name].lerp_(p.detach().float(), 1 - decay)
        self.updates += 1

    def state_dict(self):
        return dict(version=1, specification=self.specification, source=self.source,
                    aliases=self.aliases, last_step=self.last_step, updates=self.updates,
                    shadows={v: {n: p.detach().cpu().clone() for n, p in s.items()}
                             for v, s in self.shadows.items()})

    def load_state_dict(self, state, checkpoint_step):
        for key, expected in [('version', 1), ('specification', self.specification),
                              ('source', self.source), ('aliases', self.aliases),
                              ('last_step', checkpoint_step)]:
            if state.get(key) != expected:
                raise ValueError(f'EMA checkpoint {key} differs from the current run')
        updates = max(0, checkpoint_step - self.specification['start_step'] + 1)
        if state.get('updates') != updates:
            raise ValueError('EMA update count does not match the checkpoint step')
        expected_variants = set(self.specification['variants']) if updates else set()
        if set(state.get('shadows', {})) != expected_variants:
            raise ValueError('EMA checkpoint variants are incomplete')
        shadows = {}
        for variant, values in state['shadows'].items():
            if set(values) != set(self.sources):
                raise ValueError('EMA checkpoint parameters are incomplete')
            shadows[variant] = {}
            for name, value in values.items():
                target = self.sources[name]
                if value.shape != target.shape or value.dtype != torch.float32:
                    raise ValueError(f'invalid EMA tensor: {name}')
                shadows[variant][name] = value.to(device=target.device, copy=True)
        self.shadows, self.last_step, self.updates = shadows, checkpoint_step, updates


def select_ema_weights(checkpoint, variant):
    """Return an export-only state dict; preserve tied aliases and native dtypes."""
    state = checkpoint.get('ema')
    if not state or state.get('version') != 1 or state.get('updates', 0) < 1:
        raise ValueError('checkpoint contains no initialized EMA weights')
    if state.get('last_step') != checkpoint['step']:
        raise ValueError('EMA is stale relative to the checkpoint')
    if variant not in state['shadows']:
        raise ValueError(f'unknown EMA variant {variant}; available: {list(state["shadows"])}')
    weights = dict(checkpoint['model'])
    values = state['shadows'][variant]
    aliases = state['aliases']
    if set(values) != set(aliases.values()) or not set(aliases).issubset(weights):
        raise ValueError('EMA aliases do not match the checkpoint model')
    converted = {}
    for alias, canonical in aliases.items():
        reference, value = weights[alias], values[canonical]
        if value.shape != reference.shape or value.dtype != torch.float32:
            raise ValueError(f'invalid EMA export tensor: {alias}')
        if canonical not in converted:
            converted[canonical] = value.to(dtype=reference.dtype)
        if converted[canonical].dtype != reference.dtype:
            raise ValueError('tied parameters have different dtypes')
        weights[alias] = converted[canonical]
    return weights
