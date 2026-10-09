"""Check tokenizer identity and reject corrupted conversion validation inputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_hf import logits_metrics, preserve_tokenizer, read_tokenizer, sha256_file, validate_tokenizer


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


if __name__ == '__main__':
    unittest.main()
