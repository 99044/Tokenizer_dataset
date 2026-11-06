#!/usr/bin/env python3
"""Check the generated training plans and evaluation defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from generate_submission_20261103_configs import DATA_PARALLEL_SIZE, TIERS, make_config
from context_curriculum import validate_curriculum


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
        assert model['compressed_attention']['scale'] == (0.0001 if label == '3e17' else 0.001)
        assert training['gradient_accumulation_steps'] == accumulation
        assert training['micro_batch_size'] == 1
        assert actual['submission']['data_parallel_size'] == DATA_PARALLEL_SIZE == 8
        assert actual['scaling']['tokens_per_update'] == (
            DATA_PARALLEL_SIZE * accumulation * model['context_length']
        )
        assert actual['optimizer']['name'] == optimizer
        assert actual['optimizer']['compressed_branch'] == 'adamw'
        assert actual['optimizer']['learning_rate'] == 0.003
        if optimizer == 'muon':
            assert actual['optimizer']['scalar_learning_rate'] == 0.000265
            assert actual['optimizer']['adjust_lr'] == 'rms_norm'
            assert 'precondition_frequency' not in actual['optimizer']
        if label == '3e17':
            reference = training['length_curriculum']['reference']
            assert reference['max_steps'] == 5045
            assert reference['warmup_steps'] == 253
            assert reference['cooldown_steps'] == 1514
            assert reference['planned_training_flops'] == 296984324404926080
            assert training['max_steps'] == 5502
            assert actual['scaling']['tokens_per_update'] == 262144
            assert actual['scheduler']['final_lr_ratio'] == 0.01
            assert model['dual_residual'] == dict(enabled=True, start_layer=8, cross_mix_init=0.25)
            assert model['value_residual'] == dict(enabled=True, first_layer_weight=0.5)
            assert actual['scaling']['additional_residual_parameters'] == 18
            assert actual['scaling']['expected_total_parameters'] == 47756790
            assert training['length_curriculum']['accounting']['residual_extra_flops_per_token'] == 75264
        elif label == '1e19':
            assert actual['scheduler']['warmup_steps'] == 20
            assert actual['scheduler']['final_lr_ratio'] == 0.0
        if label != '3e17':
            assert 'dual_residual' not in model and 'value_residual' not in model
        if label in {'3e17', '1e18', '3e18'}:
            assert training['fp32_master_weights'] is True
            stages = training['length_curriculum']['stages']
            assert [stage['length'] for stage in stages] == [2048, 4096, 8192]
            assert [stage['micro_batch_size'] for stage in stages] == [4, 2, 1]
            assert all(stage['grad_accum_steps'] == accumulation for stage in stages)
            assert actual['scheduler']['final_lr_ratio'] == (0.01 if label == '3e17' else 0.0)
            validate_curriculum(
                actual, world_size=DATA_PARALLEL_SIZE, micro_batch_size=1,
                grad_accum_steps=accumulation, max_steps=training['max_steps'],
            )
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
