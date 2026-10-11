"""Resolve DDP training options: explicit CLI, then YAML, then legacy defaults."""

import math
from pathlib import Path

import yaml


def resolve_training_options(args):
    settings = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(settings, dict):
        raise ValueError('训练 YAML 必须包含配置映射')
    fields = {
        'max_steps': ('training', 'max_steps', 20),
        'micro_batch_size': ('training', 'micro_batch_size', 1),
        'grad_accum_steps': ('training', 'gradient_accumulation_steps', 1),
        'learning_rate': ('optimizer', 'learning_rate', 2e-4),
        'warmup_steps': ('scheduler', 'warmup_steps', 0),
        'eval_every': ('logging', 'eval_every_steps', 10),
        'eval_batches': ('logging', 'eval_batches', 2),
        'save_every': ('logging', 'checkpoint_every_steps', 10),
    }
    for name, (section, key, fallback) in fields.items():
        value = getattr(args, name)
        if value is None:
            values = settings.get(section) or {}
            if not isinstance(values, dict):
                raise ValueError(f'YAML 的 {section} 必须为配置映射')
            configured = values.get(key)
            value = configured if configured is not None else fallback
        if name == 'learning_rate':
            if isinstance(value, bool):
                raise ValueError('learning_rate 必须为有限正数')
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError('learning_rate 必须为有限正数') from None
            if not math.isfinite(value) or value <= 0:
                raise ValueError('learning_rate 必须为有限正数')
        else:
            minimum = 0 if name == 'warmup_steps' else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} 必须为大于等于 {minimum} 的整数')
        setattr(args, name, value)
    if args.warmup_steps >= args.max_steps:
        raise ValueError('warmup_steps 必须小于 max_steps')
    return args
