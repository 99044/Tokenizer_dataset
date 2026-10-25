"""Check tokenizer identity and reject corrupted conversion validation inputs."""

import json
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_hf import export_checkpoint, load_offline_model, logits_metrics, preserve_tokenizer, read_tokenizer, repack_hf, sha256_file, validate_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_export import NativeHybridCPUReference, convert_hybrid_state, hybrid_hf_config


def hybrid_fixture(vocab_size, heads=1):
    """A small genuine member of the submission family, with native key names."""
    import yaml

    config = Path(__file__).resolve().parents[2] / 'configs' / 'FLOPs_3e17.yaml'
    settings = yaml.safe_load(config.read_text())
    model = settings['model']
    hidden, intermediate, key, value = 128 * heads, 512 * heads, 96 * heads, 192 * heads
    model.update(num_layers=4, hidden_size=hidden, intermediate_size=intermediate,
                 num_attention_heads=heads, num_key_value_heads=heads,
                 gdn_num_heads=heads, gdn_num_value_heads=heads, vocab_size=vocab_size)
    generator = torch.Generator().manual_seed(19)
    state = {}

    def weight(name, shape, norm=False):
        value = torch.randn(shape, generator=generator) * (0.03 if not norm else 0.1)
        if norm:
            value += 1
        # These FP32 FLA norm weights deliberately cannot round-trip via BF16.
        state[name] = value if name.endswith('.o_norm.weight') else value.bfloat16()

    weight('embeddings.weight', (vocab_size, hidden))
    state['lm_head.w_out.weight'] = state['embeddings.weight']
    weight('lm_head.norm.weight', (hidden,), norm=True)
    for index in range(4):
        prefix = f'blocks.{index}.'
        for suffix in ('attention_norm.weight', 'feed_forward_norm.weight'):
            weight(prefix + suffix, (hidden,), norm=True)
        for name, shape in [('w1', (intermediate, hidden)), ('w2', (hidden, intermediate)),
                            ('w3', (intermediate, hidden))]:
            weight(prefix + f'feed_forward.{name}.weight', shape)
        if index == 3:
            for name in ('w_q', 'w_k', 'w_v', 'w_out'):
                weight(prefix + f'attention.{name}.weight', (hidden, hidden))
            for name in ('q_norm', 'k_norm'):
                weight(prefix + f'attention.{name}.weight', (hidden,), norm=True)
        else:
            for name, shape in [('w_q', (key, hidden)), ('w_k', (key, hidden)),
                                ('w_v', (value, hidden)), ('w_a', (heads, hidden)), ('w_b', (heads, hidden)),
                                ('w_g', (value, hidden)), ('w_out', (hidden, value))]:
                weight(prefix + f'attention.{name}.weight', shape)
            for name, width in [('q', key), ('k', key), ('v', value)]:
                weight(prefix + f'attention.{name}_conv1d.weight', (width, 1, 4))
            weight(prefix + 'attention.o_norm.weight', (192,), norm=True)
            state[prefix + 'attention.A_log'] = torch.full((heads,), -1.0, dtype=torch.bfloat16)
            state[prefix + 'attention.dt_bias'] = torch.full((heads,), -2.0, dtype=torch.bfloat16)
    return settings, state


class ExportHFTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / 'source'
        self.output = Path(temporary.name) / 'export'
        self.source.mkdir()
        self.output.mkdir()
        special = [f'<|scicomp26_{name}_v1|>' for name in ('unk', 'bos', 'eos', 'pad')]
        backend = Tokenizer(models.BPE(unk_token=special[0]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(
            ['科学计算 Hello, world! café\n spaces and 🙂'],
            trainers.BpeTrainer(vocab_size=300, special_tokens=special,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token=special[0], bos_token=special[1],
            eos_token=special[2], pad_token=special[3], model_max_length=8192,
        )
        tokenizer.save_pretrained(self.source)
        (self.source / 'special_tokens_map.json').write_text(json.dumps(tokenizer.special_tokens_map))
        (self.source / 'training_metadata.json').write_text(json.dumps({
            'revision': 'fixture_v1', 'special_token_ids': {'unk': 0, 'bos': 1, 'eos': 2, 'pad': 3},
        }))
        self.data = {
            'vocab_size': len(tokenizer), 'eos_token_id': 2, 'tokenizer': 'fixture_v1',
            'tokenizer_details': {'tokenizer_json_sha256': sha256_file(self.source / 'tokenizer.json')},
        }

    def test_original_tokenizer_bytes_and_offline_behavior_are_preserved(self):
        tokenizer, _ = read_tokenizer(self.source, self.data)
        hashes = preserve_tokenizer(self.source, self.output, tokenizer)
        for name, expected in hashes.items():
            self.assertEqual(sha256_file(self.output / name), expected)
        self.assertTrue((self.output / 'vocab.json').stat().st_size > 0)
        self.assertTrue((self.output / 'merges.txt').stat().st_size > 0)
        self.assertEqual(validate_tokenizer(tokenizer, self.output)['status'], 'passed')

    def test_wrong_tokenizer_hash_revision_vocab_or_eos_is_rejected(self):
        for field, value in (
            ('tokenizer_details', {'tokenizer_json_sha256': 'wrong'}),
            ('tokenizer', 'wrong_revision'), ('vocab_size', self.data['vocab_size'] + 1),
            ('eos_token_id', 1),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_tokenizer(self.source, dict(self.data, **{field: value}))

    def test_exported_special_token_change_is_detected(self):
        tokenizer, _ = read_tokenizer(self.source, self.data)
        preserve_tokenizer(self.source, self.output, tokenizer)
        config_path = self.output / 'tokenizer_config.json'
        config = json.loads(config_path.read_text())
        config['eos_token'] = tokenizer.bos_token
        config_path.write_text(json.dumps(config))
        map_path = self.output / 'special_tokens_map.json'
        mapping = json.loads(map_path.read_text())
        mapping['eos_token'] = tokenizer.bos_token
        map_path.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, 'special token mapping changed'):
            validate_tokenizer(tokenizer, self.output)

    def test_logit_corruption_after_chunk_boundary_is_detected(self):
        logits = torch.randn(1, 257, 7)
        self.assertTrue(logits_metrics(logits, logits.clone(), 1e-4, 1e-4)['passed'])
        corrupted = logits.clone()
        corrupted[0, 256, 6] += 1.0
        metrics = logits_metrics(logits, corrupted, 1e-4, 1e-4)
        self.assertFalse(metrics['passed'])
        self.assertEqual(metrics['mismatched_logits'], 1)
        self.assertEqual(metrics['total_logits'], logits.numel())

    def test_nonfinite_or_wrong_shape_logits_are_rejected(self):
        logits = torch.zeros(1, 2, 3)
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite'):
                logits_metrics(logits, torch.full_like(logits, value), 1e-4, 1e-4)
        with self.assertRaisesRegex(ValueError, 'shapes differ'):
            logits_metrics(logits, torch.zeros(1, 2, 4), 1e-4, 1e-4)

    def test_four_submission_hf_shapes_match_native_parameter_counts(self):
        import yaml
        from transformers import AutoModelForCausalLM

        configs = Path(__file__).resolve().parents[2] / 'configs'
        for path in sorted(configs.glob('FLOPs_*.yaml')):
            with self.subTest(config=path.name), torch.device('meta'):
                settings = yaml.safe_load(path.read_text())
                config = hybrid_hf_config(settings, 32768)
                hf = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa')
                self.assertEqual(sum(p.numel() for p in hf.parameters()),
                                 settings['scaling']['expected_total_parameters'])

    def test_hybrid_export_round_trips_shards_fp32_norms_and_offline_logits(self):
        settings, state = hybrid_fixture(self.data['vocab_size'])
        checkpoint = self.source.parent / 'step-7.pt'
        torch.save({'step': 7, 'model': state,
                    'plan': {'settings': settings, 'data': self.data}}, checkpoint)
        args = Namespace(checkpoint=checkpoint, tokenizer_dir=self.source, output_dir=self.output,
                         max_sequence_length=None, max_shard_size='100KB', validation_device='cpu',
                         validation_lengths=[1, 65, 129], logits_atol=1e-4, logits_rtol=1e-4)
        with contextlib.redirect_stdout(io.StringIO()):
            export_checkpoint(args)
        record = json.loads((self.output / 'export_record.json').read_text())
        self.assertIn(record['status'], ('passed', 'passed_with_bf16_differences'))
        self.assertEqual(record['validation_reference'], 'unfused_native_cpu_recurrence')
        self.assertTrue((self.output / 'model.safetensors.index.json').is_file())
        self.assertTrue((self.output / 'hf_hybrid.py').is_file())
        model, _ = load_offline_model(self.output, trust_remote_code=True)
        converted = convert_hybrid_state(model.config, state)
        loaded = model.state_dict()
        for name, expected in converted.items():
            self.assertEqual(expected.dtype, loaded[name].dtype, name)
            self.assertTrue(torch.equal(expected, loaded[name]), name)
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        self.assertTrue(all(item['passed'] for item in record['logits_validation']
                            if item['dtype'] == 'torch.float32'))
        # A new interpreter has neither the repository imports nor Auto mappings.
        script = '''
import sys, torch
from transformers import AutoConfig, AutoModelForCausalLM
path = sys.argv[1]
try:
    AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
except ValueError:
    pass
else:
    raise AssertionError('custom adapter loaded without explicit trust')
config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, config=config, local_files_only=True,
    trust_remote_code=True, dtype=torch.bfloat16, attn_implementation='sdpa').eval()
with torch.inference_mode():
    a = model(torch.tensor([[1, 4, 7]]), use_cache=False).logits
    b = model(torch.tensor([[1, 4, 7]]), use_cache=False).logits
assert a.shape == (1, 3, config.vocab_size)
assert torch.equal(a, b) and torch.isfinite(a).all()
'''
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable, '-B', '-c', script, str(self.output)], cwd='/tmp',
                       env=env, check=True, timeout=120, capture_output=True, text=True)
        repacked = self.source.parent / 'repacked'
        with contextlib.redirect_stdout(io.StringIO()):
            repack_hf(Namespace(source_hf=self.output, output_dir=repacked, max_shard_size='100KB'))
        repacked_model, _ = load_offline_model(repacked, trust_remote_code=True)
        for name, expected in converted.items():
            actual = repacked_model.state_dict()[name]
            self.assertEqual(expected.dtype, actual.dtype, name)
            self.assertTrue(torch.equal(expected, actual), name)

    def test_batched_multihead_hybrid_logits_match_independent_recurrence(self):
        from transformers import AutoModelForCausalLM

        settings, state = hybrid_fixture(self.data['vocab_size'], heads=2)
        config = hybrid_hf_config(settings, self.data['vocab_size'])
        hf = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa')
        hf.load_state_dict(convert_hybrid_state(config, state), strict=True, assign=True)
        hf.tie_weights()
        hf = hf.float().eval()
        reference = NativeHybridCPUReference(settings, state).float().eval()
        inputs = torch.randint(self.data['vocab_size'], (2, 65), generator=torch.Generator().manual_seed(5))
        with torch.inference_mode():
            expected = reference(inputs)
            actual = hf(inputs, use_cache=False).logits
        self.assertTrue(logits_metrics(expected, actual, 1e-4, 1e-4)['passed'])

    def test_hybrid_unknown_weights_and_different_tied_weights_are_rejected(self):
        settings, state = hybrid_fixture(self.data['vocab_size'])
        config = hybrid_hf_config(settings, self.data['vocab_size'])
        with self.assertRaisesRegex(ValueError, 'unmapped native Hybrid weights'):
            convert_hybrid_state(config, dict(state, unexpected=torch.zeros(1)))
        state['lm_head.w_out.weight'] = state['embeddings.weight'].clone() + 1
        checkpoint = self.source.parent / 'bad.pt'
        torch.save({'step': 1, 'model': state,
                    'plan': {'settings': settings, 'data': self.data}}, checkpoint)
        args = Namespace(checkpoint=checkpoint, tokenizer_dir=self.source, output_dir=self.output,
                         max_sequence_length=None, validation_device='cpu')
        with self.assertRaisesRegex(ValueError, 'tied embeddings but stores different weights'):
            export_checkpoint(args)


if __name__ == '__main__':
    unittest.main()
