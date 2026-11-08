#!/usr/bin/env python3
"""Check the three selected training plans and evaluation defaults."""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/src'))
from context_curriculum import validate_curriculum

EXPECTED = {
    '3e17': (24576, 8, 4, 5502, 47756790),
    '3e18': (24576, 8, 4, 16902, 200197018),
    '1e19': (32768, 8, 8, 8143, 1171647936),
}


def main():
    files = {path.name for path in (ROOT / 'training/configs').glob('FLOPs_*.yaml')}
    expected_files = {f'FLOPs_{tier}.yaml' for tier in EXPECTED}
    if files != expected_files:
        raise ValueError(f'Unexpected tier configurations: {files ^ expected_files}')

    for tier, (vocab, world, accum, steps, parameters) in EXPECTED.items():
        path = ROOT / 'training/configs' / f'FLOPs_{tier}.yaml'
        settings = yaml.safe_load(path.read_text())
        model, training = settings['model'], settings['training']
        tokens_per_update = world * model['context_length'] * training['micro_batch_size'] * accum
        assert model['vocab_size'] == settings['scaling']['parameter_count_vocab_size'] == vocab
        assert settings['submission']['data_parallel_size'] == world
        assert training['gradient_accumulation_steps'] == accum
        assert training['max_steps'] == steps
        assert settings['scaling']['expected_total_parameters'] == parameters
        assert settings['scaling']['tokens_per_update'] == tokens_per_update
        assert settings['budget']['planned_training_tokens'] == steps * tokens_per_update
        assert settings['budget']['planned_training_flops'] < settings['budget']['total_training_flops']
        assert settings['logging']['checkpoint_keep_last'] == 2
        assert model['compressed_attention']['layer_index'] == model['num_layers'] - 1
        if 'length_curriculum' in training:
            validate_curriculum(settings, world, training['micro_batch_size'], accum, steps)
        print(tier, parameters, steps, tokens_per_update)

    three = yaml.safe_load((ROOT / 'training/configs/FLOPs_3e18.yaml').read_text())
    assert three['optimizer']['name'] == 'soap'
    assert 'length_curriculum' not in three['training']
    assert 'dual_residual' not in three['model']
    assert 'value_residual' not in three['model']
    assert three['scheduler']['warmup_steps'] == 845
    assert three['scheduler']['cooldown_steps'] == 5071
    assert three['scheduler']['final_lr_ratio'] == 0.01

    large = yaml.safe_load((ROOT / 'training/configs/FLOPs_1e19.yaml').read_text())
    assert large['optimizer']['name'] == 'muon'
    assert large['optimizer']['legacy_compressed_muon_split'] is True
    assert large['model']['compressed_attention']['block_size'] == 388
    assert large['model']['num_layers'] == 16
    assert large['training']['dtype'] == 'bfloat16'
    assert 'fp32_master_weights' not in large['training']

    evaluation = (ROOT / 'evaluation/eval.py').read_text()
    launcher = (ROOT / 'evaluation/eval.sh').read_text()
    assert 'default=32' in evaluation
    assert 'default="ffn_rmsnorm"' in evaluation
    assert 'BATCH_SIZE:-32' in launcher
    assert 'FUSION_MODE:-ffn_rmsnorm' in launcher


if __name__ == '__main__':
    main()
