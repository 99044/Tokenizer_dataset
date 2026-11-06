"""CPU fixtures for token-only validation; no tokenizer or training files needed."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import packed_cache
except ImportError:
    import packed_cache


class PackedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hash = "a" * 64
        self.tokens = np.arange(130, dtype=np.uint32) % 31
        self.tokens[0] = 2
        self.mask = np.ones(130, dtype=np.uint8)
        self.mask[[0, 32, 129]] = 0
        self.scored = int(self.mask.sum())

    def fixture(self, name="npy", *, raw=False, metadata_update=None, tokens=None, mask=None):
        directory = self.root / name
        directory.mkdir()
        tokens = self.tokens if tokens is None else tokens
        mask = self.mask if mask is None else mask
        if raw:
            tokens.astype(np.int32).tofile(directory / "tokens.bin")
            mask.tofile(directory / "is_text.bin")
            metadata = dict(total_tokens=self.scored, total_bytes=257,
                            tokenizer_sha256=self.hash, pad_token_id=2)
        else:
            np.save(directory / "validation.npy", tokens)
            np.save(directory / "validation_is_text.npy", mask)
            metadata = dict(
                format="flat_npy_token_stream_with_is_text_sidecar",
                tokenizer=dict(tokenizer_json_sha256=self.hash, vocab_size=32,
                               eos_token_id=2, pad_token_id=None,
                               path="/unavailable/tokenizer"),
                arrays=dict(validation="validation.npy", validation_is_text="validation_is_text.npy",
                            train="/never/read/train.npy", train_is_text="/never/read/train_is_text.npy",
                            token_dtype=str(tokens.dtype), is_text_dtype=str(mask.dtype)),
                splits=dict(validation=dict(stream_tokens=len(tokens), scored_text_tokens=self.scored,
                                             utf8_bytes=257)),
            )
        if metadata_update:
            metadata_update(metadata)
        self.write_metadata(directory, metadata)
        return directory, metadata

    def write_metadata(self, directory, metadata):
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_formats_keep_identical_scoring_and_raw_legacy_metadata(self):
        npy, _ = self.fixture()
        raw, _ = self.fixture("raw", raw=True)
        a, ma, ia = packed_cache.load_packed_cache(npy)
        b, mb, ib = packed_cache.load_packed_cache(raw)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ma, mb)
        self.assertEqual(a.dtype, np.dtype(np.uint32))
        self.assertEqual(b.dtype, np.dtype(np.int32))
        self.assertEqual(mb.dtype, np.dtype(np.bool_))
        for key in ("total_tokens", "total_bytes", "tokenizer_sha256", "pad_token_id", "stream_tokens"):
            self.assertEqual(ia[key], ib[key])
        self.assertEqual(ia["format"], "npy")
        self.assertEqual(ib["format"], "raw_int32")
        self.assertEqual(ia["eos_token_id"], 2)
        self.assertEqual(ia["vocab_size"], 32)
        self.assertNotIn("eos_token_id", ib)
        self.assertNotIn("vocab_size", ib)

    def test_files_only_select_validation_and_reject_partial_npy(self):
        directory, _ = self.fixture()
        self.assertEqual([path.name for path in packed_cache.packed_cache_files(directory)],
                         ["metadata.json", "validation.npy", "validation_is_text.npy"])
        # Nothing reads the invalid train paths declared in metadata.
        packed_cache.load_packed_cache(directory)
        raw, _ = self.fixture("raw", raw=True)
        self.assertEqual([path.name for path in packed_cache.packed_cache_files(raw)],
                         ["metadata.json", "tokens.bin", "is_text.bin"])
        (raw / "validation.npy").touch()
        with self.assertRaisesRegex(ValueError, "missing required validation files"):
            packed_cache.packed_cache_files(raw)

    def test_metadata_cannot_redirect_validation_paths(self):
        directory, metadata = self.fixture()
        metadata["arrays"]["validation"] = "../../another.npy"
        self.write_metadata(directory, metadata)
        with self.assertRaisesRegex(ValueError, "fixed name"):
            packed_cache.load_packed_cache(directory)

    def test_required_metadata_and_integer_byte_denominator(self):
        for raw in (False, True):
            for key, bad_value in (("hash", None), ("hash", "fixture"),
                                   ("bytes", None), ("bytes", 0), ("bytes", -1),
                                   ("bytes", 257.0), ("bytes", True), ("tokens", 126)):
                with self.subTest(raw=raw, key=key, bad_value=bad_value):
                    directory, metadata = self.fixture(f"fixture-{len(list(self.root.iterdir()))}", raw=raw)
                    if key == "hash":
                        owner, field = (metadata, "tokenizer_sha256") if raw else (
                            metadata["tokenizer"], "tokenizer_json_sha256")
                    else:
                        owner = metadata if raw else metadata["splits"]["validation"]
                        field = ({"bytes": "total_bytes", "tokens": "total_tokens"} if raw else
                                 {"bytes": "utf8_bytes", "tokens": "scored_text_tokens"})[key]
                    if bad_value is None:
                        del owner[field]
                    else:
                        owner[field] = bad_value
                    self.write_metadata(directory, metadata)
                    with self.assertRaises(ValueError):
                        packed_cache.load_packed_cache(directory)

    def test_raw_mask_rejects_nonbinary_bytes_and_initial_score(self):
        for index, value in ((40, 2), (40, 255), (0, 1)):
            with self.subTest(index=index, value=value):
                mask = self.mask.copy()
                mask[index] = value
                directory, _ = self.fixture(f"mask-{index}-{value}", raw=True, mask=mask)
                with self.assertRaises(ValueError):
                    packed_cache.load_packed_cache(directory)

    def test_npy_mask_rejects_bad_values_and_shapes(self):
        masks = [self.mask.astype(np.float32), self.mask.reshape(2, 65), self.mask[:-1],
                 np.where(np.arange(130) == 70, 2, self.mask).astype(np.int32)]
        for index, mask in enumerate(masks):
            with self.subTest(index=index):
                directory, _ = self.fixture(f"mask-{index}", mask=mask)
                with self.assertRaises(ValueError):
                    packed_cache.load_packed_cache(directory)

    def test_raw_dtype_is_never_guessed_and_partial_values_fail(self):
        for declared in ("uint32", "int64", "mystery"):
            with self.subTest(dtype=declared):
                directory, _ = self.fixture(declared, raw=True,
                                            metadata_update=lambda metadata: metadata.update(token_dtype=declared))
                with self.assertRaisesRegex(ValueError, "dtype"):
                    packed_cache.load_packed_cache(directory)
        directory, _ = self.fixture("truncated", raw=True)
        with (directory / "tokens.bin").open("ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "complete int32"):
            packed_cache.load_packed_cache(directory)

    def test_npy_header_dtype_and_metadata_must_agree(self):
        directory, metadata = self.fixture()
        metadata["arrays"]["token_dtype"] = "int32"
        self.write_metadata(directory, metadata)
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            packed_cache.load_packed_cache(directory)
        directory, _ = self.fixture("floating", tokens=self.tokens.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "integer tokens"):
            packed_cache.load_packed_cache(directory)

    def test_npy_trailing_bytes_and_inconsistent_stream_metadata_fail(self):
        directory, _ = self.fixture()
        with (directory / "validation.npy").open("ab") as stream:
            stream.write(b"trailing")
        with self.assertRaisesRegex(ValueError, "trailing NPY"):
            packed_cache.load_packed_cache(directory)
        directory, metadata = self.fixture("length")
        metadata["splits"]["validation"]["stream_tokens"] += 1
        self.write_metadata(directory, metadata)
        with self.assertRaisesRegex(ValueError, "stream length"):
            packed_cache.load_packed_cache(directory)

    def test_token_ranges_scan_all_chunks_and_all_positions(self):
        # Invalid IDs in the final, unscored position must still be rejected:
        # masked targets can still become model inputs.
        for raw, value in ((False, 32), (True, -1)):
            tokens = self.tokens.astype(np.int64)
            tokens[-1] = value
            directory, _ = self.fixture(f"range-{raw}", raw=raw, tokens=tokens)
            with patch.object(packed_cache, "_CHUNK_TOKENS", 17), self.assertRaisesRegex(ValueError, "Token IDs"):
                packed_cache.load_packed_cache(directory)

    def test_identity_matches_without_opening_tokenizer(self):
        directory, _ = self.fixture()
        tokens, _, metadata = packed_cache.load_packed_cache(directory)
        checkpoint = dict(vocab_size=32, eos_token_id=2, tokenizer="/missing",
                          tokenizer_details=dict(tokenizer_json_sha256=self.hash, eos_token_id=2))
        with patch.object(packed_cache, "_CHUNK_TOKENS", 17):
            result = packed_cache.validate_packed_identity(metadata, checkpoint, 32, tokens)
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["checkpoint_tokenizer_hash_status"], "validated")
        self.assertEqual(result["eos_match_status"], "validated")
        self.assertEqual(result["token_id_range"]["count"], 130)
        self.assertEqual(result["token_id_range"]["maximum"], 30)
        self.assertFalse(result["tokenizer_files_required"])

    def test_missing_checkpoint_hash_is_not_claimed_as_verified_identity(self):
        directory, _ = self.fixture("raw", raw=True)
        tokens, _, metadata = packed_cache.load_packed_cache(directory)
        result = packed_cache.validate_packed_identity(metadata, dict(vocab_size=32), 32, tokens)
        self.assertEqual(result["status"], "validated_without_checkpoint_tokenizer_hash")
        self.assertEqual(result["checkpoint_tokenizer_hash_status"], "not_available")
        self.assertFalse(result["hash_present_in_checkpoint"])
        self.assertEqual(result["eos_match_status"], "not_available")
        self.assertEqual(result["token_id_range"]["status"], "validated")

    def test_identity_rejects_hash_eos_vocab_pad_and_conflicting_fields(self):
        directory, _ = self.fixture()
        tokens, _, metadata = packed_cache.load_packed_cache(directory)
        for checkpoint in (
            dict(tokenizer_json_sha256="b" * 64),
            dict(tokenizer_json_sha256="invalid"),
            dict(tokenizer_json_sha256=self.hash, tokenizer_details=dict(tokenizer_json_sha256="b" * 64)),
            dict(eos_token_id=3), dict(eos_token_id=32), dict(eos_token_id=[2]),
            dict(eos_token_id=2, tokenizer_details=dict(eos_token_id=3)), dict(vocab_size=33),
        ):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                packed_cache.validate_packed_identity(metadata, checkpoint, 32, tokens)
        for field, value in (("pad_token_id", -1), ("pad_token_id", 32), ("pad_token_id", None),
                             ("total_bytes", 0), ("tokenizer_sha256", None), ("vocab_size", 33),
                             ("stream_tokens", 130.0), ("stream_tokens", 127)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                packed_cache.validate_packed_identity(dict(metadata, **{field: value}), {}, 32, tokens)

    def test_identity_checks_ids_even_when_legacy_cache_has_no_vocab(self):
        directory, _ = self.fixture("raw", raw=True)
        tokens, _, metadata = packed_cache.load_packed_cache(directory)
        invalid = np.array(tokens)
        invalid[-1] = 32
        with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
            packed_cache.validate_packed_identity(metadata, {}, 32, invalid)
        result = packed_cache.validate_packed_identity(metadata, {}, 32)
        self.assertEqual(result["token_id_range"]["status"], "not_checked")

    def test_npy_rejects_conflicting_saved_identity(self):
        directory, metadata = self.fixture()
        metadata["tokenizer_sha256"] = "b" * 64
        self.write_metadata(directory, metadata)
        with self.assertRaisesRegex(ValueError, "Conflicting tokenizer"):
            packed_cache.load_packed_cache(directory)


if __name__ == "__main__":
    unittest.main()
