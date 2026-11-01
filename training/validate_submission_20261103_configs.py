#!/usr/bin/env python3
"""Check the generated training plans and evaluation defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from generate_submission_20261103_configs import TIERS, make_config


def main():
    for tier in TIERS:
        label, budget, hidden, intermediate, heads, accumulation, optimizer, _ = tier
        path = ROOT / 'training' / 'configs' / f'FLOPs_{label}.yaml'
        actual = yaml.safe_load(path.read_text())
        expected = make_config(tier)
        if actual != expected:
            raise ValueError(f'{path} differs from the generated plan')
        model = actual['model']
        training = actual['training']
        budget_record = actual['budget']
        assert model['hidden_size'] == hidden
        assert model['intermediate_size'] == intermediate
        assert model['num_attention_heads'] == heads
        assert model['vocab_size'] == 24576
        assert model['compressed_attention']['block_size'] == 128
        assert model['compressed_attention']['scale'] == 0.001
        assert training['gradient_accumulation_steps'] == accumulation
        assert actual['optimizer']['name'] == optimizer
        assert actual['optimizer']['learning_rate'] == 0.003
        if label == '3e17':
            assert actual['scheduler']['warmup_steps'] == 505
            assert actual['scheduler']['cooldown_steps'] == 3027
            assert actual['scheduler']['final_lr_ratio'] == 0.01
        else:
            assert actual['scheduler']['warmup_steps'] == 20
            assert actual['scheduler']['final_lr_ratio'] == 0.0
        assert actual['initialization']['projection_scale'] == 0.8
        assert actual['scaling']['expected_branch_parameters'] > 0
        assert budget_record['planned_training_flops'] <= budget * 99 // 100
        assert budget_record['planned_training_tokens'] == (
            training['max_steps'] * actual['scaling']['tokens_per_update']
        )
        print(label, actual['scaling']['expected_total_parameters'], training['max_steps'])
    evaluation = (ROOT / 'evaluation' / 'eval.py').read_text()
    launcher = (ROOT / 'evaluation' / 'eval.sh').read_text()
    if 'default=32' not in evaluation or 'default="ffn_rmsnorm"' not in evaluation:
        raise ValueError('HF evaluation defaults differ from batch 32 and operator fusion')
    if 'BATCH_SIZE:-32' not in launcher or 'FUSION_MODE:-ffn_rmsnorm' not in launcher:
        raise ValueError('HF evaluation launcher defaults differ from the evaluator')


if __name__ == '__main__':
    main()
