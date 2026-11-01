#!/usr/bin/env python3
"""Generate the four 24K wide Hybrid training configurations."""

from __future__ import annotations

import copy
import math
import sys
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from graph_accounting import audit_ledger
from hybrid_flops import complete_hybrid_ledger, expected_parameter_count

BASE = ROOT / 'training' / 'configs' / 'FLOPs_3e17.yaml'
TIERS = (
    ('3e17', 300_000_000_000_000_000, 256, 1984, 2, 4, 'soap', 0.003),
    ('1e18', 1_000_000_000_000_000_000, 640, 2048, 5, 6, 'soap', 0.003),
    ('3e18', 3_000_000_000_000_000_000, 640, 2048, 5, 8, 'soap', 0.003),
    ('1e19', 10_000_000_000_000_000_000, 1536, 6144, 12, 16, 'soap', 0.003),
)
NOMINAL_SIZES = {'3e17': '50m', '1e18': '200m', '3e18': '200m', '1e19': '1.2b'}


def make_config(tier):
    label, budget, hidden, intermediate, heads, accumulation, optimizer_name, lr = tier
    settings = copy.deepcopy(yaml.safe_load(BASE.read_text()))
    model = settings['model']
    model.update(
        num_layers=12, hidden_size=hidden, intermediate_size=intermediate,
        num_attention_heads=heads, num_key_value_heads=heads,
        gdn_num_heads=heads, gdn_num_value_heads=heads,
        vocab_size=24576,
        compressed_attention=dict(
            layer_index=11, block_size=128,
            intermediate_size=intermediate, scale=0.001,
        ),
    )
    shape = dict(model, gdn_value_dim=192)
    base_parameters = expected_parameter_count(shape, 24576)
    branch_parameters = 2 * 128 * hidden * hidden + 4 * hidden * intermediate
    total_parameters = base_parameters + branch_parameters
    ledger = complete_hybrid_ledger(BASE, settings, 1, 24576, 1, base_only=True)
    base_micro_flops = audit_ledger(
        ledger, BASE, settings, 1, 24576, ROOT, base_only=True,
    )['totals']['total']
    compressed_blocks = model['context_length'] // 128
    branch_forward = Fraction(
        4 * compressed_blocks * 128 * hidden * hidden
        + 8 * compressed_blocks * hidden * intermediate,
        model['context_length'],
    ) + 4 * hidden * compressed_blocks
    branch_per_token = math.ceil(3 * branch_forward)
    per_micro = base_micro_flops + model['context_length'] * branch_per_token
    tokens_per_update = 4 * accumulation * model['context_length']
    flops_per_update = 4 * accumulation * per_micro
    max_steps = (budget * 99 // 100) // flops_per_update
    if max_steps < 10:
        raise ValueError('budget is too small for checkpoint schedule')
    checkpoints = sorted({round(max_steps * part / 10) for part in range(1, 10)})
    settings['run'] = dict(name=f'wide_compressed_{hidden}_{label}_{optimizer_name}', seed=42)
    settings['scaling'] = dict(
        family='wide_compressed_24k', nominal_size=NOMINAL_SIZES[label],
        expected_total_parameters=total_parameters,
        expected_branch_parameters=branch_parameters,
        parameter_count_vocab_size=24576,
        tokens_per_update=tokens_per_update,
        flops_accounting='Hybrid ledger plus compressed-branch upper estimate',
    )
    settings['budget'] = dict(
        total_training_flops=budget,
        planned_training_flops=max_steps * flops_per_update,
        planned_training_tokens=max_steps * tokens_per_update,
        flops_per_token_numerator=per_micro,
        flops_per_token_denominator=model['context_length'],
        flops_per_update=flops_per_update,
    )
    settings['training'].update(
        max_steps=max_steps, micro_batch_size=1,
        gradient_accumulation_steps=accumulation,
    )
    settings['training'].pop('fp32_master_weights', None)
    optimizer = dict(
        name=optimizer_name, learning_rate=lr, betas=[0.95, 0.95],
        eps=1e-8, weight_decay=0.1,
    )
    optimizer.update(precondition_frequency=10, max_precond_dim=1024)
    settings['optimizer'] = optimizer
    settings['initialization'] = dict(projection_scale=0.8, repair_gdn_a_log=True)
    settings['scheduler'] = dict(
        name='linear_cooldown', warmup_steps=20,
        cooldown_steps=max(1, round(max_steps * 0.167)), final_lr_ratio=0.0,
    )
    settings['logging'] = dict(
        backend='wandb', entity='echo-lab', project='scicomp26',
        log_every_steps=5, eval_every_steps=max_steps,
        eval_batches=8, final_eval_batches=64,
        checkpoint_every_steps=max_steps, checkpoint_steps=checkpoints,
        checkpoint_keep_last=2,
    )
    settings['output'] = dict(directory=f'runs/submission_20261103/{label}')
    settings['submission'] = dict(target_date='2026-11-03', budget_tier=label,
                                  data_parallel_size=4)
    return settings


def main():
    for tier in TIERS:
        label = tier[0]
        path = ROOT / 'training' / 'configs' / f'FLOPs_{label}.yaml'
        config = make_config(tier)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(label, config['scaling']['expected_total_parameters'],
              config['training']['max_steps'],
              config['budget']['planned_training_flops'])


if __name__ == '__main__':
    main()
