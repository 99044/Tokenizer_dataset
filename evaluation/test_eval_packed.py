"""Check identical labels and BPB denominators for JSONL and the archive layout."""

from contextlib import redirect_stderr
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import load_evaluation_cache, pack_jsonl, packed_batch, parse_args


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        return {'input_ids': [[ord(char) % 250 + 3 for char in text] for text in texts]}


class PackedEvaluationTests(unittest.TestCase):
    def test_cached_uint32_labels_and_bpb_match_jsonl_without_retokenization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            texts = ['中文 hello ' * 16, 'another document ' * 12]
            source = root / 'dev.jsonl'
            source.write_text('\n'.join(json.dumps({'text': text}) for text in texts))
            packed = root / 'original'
            packed.mkdir()
            expected = pack_jsonl(source, CharacterTokenizer(), packed)
            tokens = np.fromfile(packed / 'tokens.bin', dtype=np.int32)
            mask = np.fromfile(packed / 'is_text.bin', dtype=np.bool_)
            cache = root / ('a' * 64)
            cache.mkdir()
            np.save(cache / 'validation.npy', tokens.astype(np.uint32))
            np.save(cache / 'validation_is_text.npy', mask.astype(np.uint8))
            model = root / 'hf-model'
            model.mkdir()
            tokenizer_bytes = b'original training tokenizer bytes'
            (model / 'tokenizer.json').write_bytes(tokenizer_bytes)
            metadata = dict(
                format='flat_npy_token_stream_with_is_text_sidecar',
                tokenizer=dict(vocab_size=256, eos_token_id=1, pad_token_id=0,
                               tokenizer_json_sha256=hashlib.sha256(tokenizer_bytes).hexdigest()),
                splits=dict(validation=dict(stream_tokens=len(tokens),
                                            scored_text_tokens=expected['total_tokens'],
                                            utf8_bytes=expected['total_bytes'])),
            )
            (cache / 'metadata.json').write_text(json.dumps(metadata))
            config = SimpleNamespace(vocab_size=256, eos_token_id=1)
            cached_tokens, cached_mask, actual, identity = load_evaluation_cache(cache, model, config)
            count = math.ceil((len(tokens) - 1) / 17)
            original_inputs, original_labels = packed_batch(tokens, mask, 0, count, 17, 0)
            inputs, labels = packed_batch(cached_tokens, cached_mask, 0, count, 17, 0)
            np.testing.assert_array_equal(inputs, original_inputs)
            np.testing.assert_array_equal(labels, original_labels)
            self.assertEqual(int(np.count_nonzero(labels != -100)), actual['total_tokens'])
            self.assertEqual(actual['total_bytes'], sum(len(text.encode('utf-8')) for text in texts))
            logits = -np.arange(256, dtype=np.float64) / 100
            normalizer = np.log(np.exp(logits).sum())
            nll = lambda values: (normalizer - logits[values[values != -100]]).sum()
            self.assertEqual(nll(labels) / actual['total_bytes'], nll(original_labels) / expected['total_bytes'])
            self.assertFalse(cached_tokens.flags.writeable)
            self.assertFalse(cached_mask.flags.writeable)
            self.assertEqual(identity['identity']['checkpoint_tokenizer_hash_status'], 'validated')
            (model / 'tokenizer.json').write_bytes(b'different tokenizer')
            with self.assertRaises(ValueError):
                load_evaluation_cache(cache, model, config)

    def test_uint32_preserves_negative_ignore_labels(self):
        tokens = np.asarray([1, 3, 1, 4], dtype=np.uint32)
        mask = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        _, labels = packed_batch(tokens, mask, 0, 1, 4, 0)
        self.assertEqual(labels.tolist(), [[3, -100, 4, -100]])

    def test_cli_supports_cache_and_keeps_jsonl_modes_exclusive(self):
        args = parse_args(['--model-path', 'model', '--packed-cache-dir', 'cache'])
        self.assertIsNone(args.dataset_path)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.fusion, 'ffn_rmsnorm')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--model-path', 'model', '--packed-cache-dir', 'cache', '--dataset-path', 'data.jsonl'])


if __name__ == '__main__':
    unittest.main()
