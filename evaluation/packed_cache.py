"""Read either the HF evaluation cache or the training cache's dev split."""

import hashlib
import json
from pathlib import Path

import numpy as np


_CHUNK_TOKENS = 1_000_000


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"{name} must contain a valid tokenizer SHA-256; a tokenizer file is not required")
    return value.lower()


def _identity_sources(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Tokenizer/data metadata must be a JSON object")
    result = [("data", metadata)]
    for name in ("tokenizer", "tokenizer_details"):
        value = metadata.get(name)
        if isinstance(value, dict):
            result.append((name, value))
    return result


def _identity_field(sources, keys, label, validator):
    values = []
    for name, source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                field = f"{name}.{key}"
                values.append((field, validator(source[key], field)))
    if values and any(value != values[0][1] for _, value in values[1:]):
        raise ValueError(f"Conflicting {label} values in metadata")
    return (values[0][1] if values else None), [name for name, _ in values]


def packed_cache_files(directory) -> list[Path]:
    """Return only metadata and the two fixed validation filenames, never train data.

    Paths declared inside metadata are not followed. A partial NPY pair is an
    error, even when an unrelated raw cache also exists in the directory.
    """
    directory = Path(directory)
    metadata = directory / "metadata.json"
    npy = [directory / "validation.npy", directory / "validation_is_text.npy"]
    streams = npy if any(path.exists() for path in npy) else [
        directory / "tokens.bin", directory / "is_text.bin"
    ]
    files = [metadata, *streams]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"Packed cache is missing required validation files: {', '.join(missing)}")
    return files


def _check_declared_dtype(source, keys, actual, label):
    for container in (source, source.get("arrays", {})):
        if not isinstance(container, dict):
            raise ValueError("Cache arrays metadata must be a JSON object")
        for key in keys:
            if key not in container:
                continue
            try:
                declared = np.dtype(container[key])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Unknown {label} dtype in cache metadata: {container[key]!r}") from error
            if declared != actual:
                raise ValueError(f"{label} dtype differs from the supported file dtype: {declared} != {actual}")


def _token_range(tokens, vocab_size=None):
    if not isinstance(tokens, np.ndarray) or tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError("Cache requires a flat integer token stream")
    minimum = maximum = None
    for offset in range(0, len(tokens), _CHUNK_TOKENS):
        chunk = tokens[offset:offset + _CHUNK_TOKENS]
        low, high = int(chunk.min()), int(chunk.max())
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
        if low < 0 or high > np.iinfo(np.int64).max:
            raise ValueError("Token IDs must be nonnegative and representable as int64")
        if vocab_size is not None and high >= vocab_size:
            raise ValueError(f"Token IDs are outside model vocabulary [0, {vocab_size})")
    return dict(status="validated", count=len(tokens), minimum=minimum, maximum=maximum)


def model_tokenizer_sha256(model_path):
    """Verify bundled tokenizer bytes, or checkpoint metadata for token-only HF exports."""
    model_path = Path(model_path)
    tokenizer = model_path / 'tokenizer.json'
    if tokenizer.is_file():
        return hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    record = json.loads((model_path / 'export_record.json').read_text())
    metadata = json.loads((model_path / 'tokenizer_metadata.json').read_text())
    if record.get('status') not in ('passed', 'passed_with_bf16_differences'):
        raise ValueError('HF export has not passed weight/logits validation')
    if metadata != record.get('tokenizer') or metadata.get('files_included') is not False:
        raise ValueError('HF tokenizer metadata differs from its export record')
    if metadata.get('source') != 'checkpoint_data_metadata':
        raise ValueError('HF tokenizer identity must come from checkpoint data metadata')
    config = json.loads((model_path / 'config.json').read_text())
    if config.get('vocab_size') != metadata['vocab_size']:
        raise ValueError('HF vocabulary size differs from checkpoint metadata')
    for name in ('bos', 'eos', 'pad'):
        if config.get(f'{name}_token_id') != metadata['special_token_ids'][name]:
            raise ValueError(f'HF {name} token ID differs from checkpoint metadata')
    digest = metadata['tokenizer_json_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid tokenizer identity in checkpoint metadata')
    return digest


def load_packed_cache(directory):
    """Load and validate the complete validation stream without a tokenizer.

    ``total_bytes`` must be the saved UTF-8 byte count of the original scored
    text. It cannot be recovered faithfully by decoding token IDs, so a missing
    denominator is an error. Legacy raw caches retain their int32/byte-mask
    convention; NPY files carry their own integer dtype in the file header.
    """
    metadata_path, token_path, mask_path = packed_cache_files(directory)
    try:
        source = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Packed cache metadata.json must be valid JSON") from error
    sources = _identity_sources(source)
    digest, _ = _identity_field(sources, ("tokenizer_sha256", "tokenizer_json_sha256"),
                                "tokenizer SHA-256", _sha256)
    if digest is None:
        raise ValueError("Packed cache metadata is missing tokenizer SHA-256 identity")
    eos, _ = _identity_field(sources, ("eos_token_id",), "EOS", _integer)
    vocab, _ = _identity_field(sources, ("vocab_size",), "vocabulary size",
                               lambda value, name: _integer(value, name, 1))
    pad, _ = _identity_field(sources, ("pad_token_id",), "PAD", _integer)
    is_npy = token_path.name == "validation.npy"
    if pad is None and is_npy:
        pad = eos
    if pad is None:
        raise ValueError("Packed cache metadata requires a PAD ID (or EOS fallback for NPY)")
    if vocab is not None and (pad >= vocab or (eos is not None and eos >= vocab)):
        raise ValueError("Cache PAD/EOS token ID is outside the vocabulary")
    if is_npy:
        try:
            split = source["splits"]["validation"]
            total_tokens = _integer(split["scored_text_tokens"], "scored_text_tokens", 1)
            total_bytes = _integer(split["utf8_bytes"], "UTF-8 bytes", 1)
            expected_stream = _integer(split["stream_tokens"], "stream_tokens", 1)
        except (KeyError, TypeError) as error:
            raise ValueError("NPY cache requires validation stream_tokens, scored_text_tokens and UTF-8 bytes") from error
        arrays = source.get("arrays", {})
        if not isinstance(arrays, dict):
            raise ValueError("Cache arrays metadata must be a JSON object")
        for key, expected in (("validation", token_path.name), ("validation_is_text", mask_path.name)):
            if key in arrays and arrays[key] != expected:
                raise ValueError(f"Cache {key} filename must be the fixed name {expected}")
        try:
            tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        except (ValueError, OSError) as error:
            raise ValueError("Cannot read integer NPY token/mask streams") from error
        for path, array in ((token_path, tokens), (mask_path, mask)):
            if not isinstance(array, np.memmap):
                raise ValueError(f"{path.name} must be a single NPY array")
            if path.stat().st_size != array.offset + array.nbytes:
                raise ValueError(f"Incomplete or trailing NPY data: {path.name}")
        _check_declared_dtype(source, ("token_dtype", "tokens_dtype", "dtype"), tokens.dtype, "Token")
        _check_declared_dtype(source, ("is_text_dtype", "mask_dtype"), mask.dtype, "Mask")
    else:
        try:
            total_tokens = _integer(source["total_tokens"], "total_tokens", 1)
            total_bytes = _integer(source["total_bytes"], "UTF-8 bytes", 1)
        except KeyError as error:
            raise ValueError("Raw cache requires total_tokens and original-text UTF-8 bytes (total_bytes)") from error
        expected_stream = source.get("stream_tokens")
        if expected_stream is not None:
            expected_stream = _integer(expected_stream, "stream_tokens", 1)
        _check_declared_dtype(source, ("token_dtype", "tokens_dtype", "dtype"), np.dtype(np.int32), "Token")
        # Read raw mask bytes as uint8, otherwise bool decoding would silently
        # turn malformed bytes such as 2 or 255 into a valid-looking True.
        for container in (source, source.get("arrays", {})):
            if not isinstance(container, dict):
                raise ValueError("Cache arrays metadata must be a JSON object")
            for key in ("is_text_dtype", "mask_dtype"):
                if key in container:
                    try:
                        dtype = np.dtype(container[key])
                    except (ValueError, TypeError) as error:
                        raise ValueError("Unknown raw mask dtype") from error
                    if dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
                        raise ValueError("Raw cache mask dtype must be bool or uint8")
        if token_path.stat().st_size == 0 or token_path.stat().st_size % np.dtype(np.int32).itemsize:
            raise ValueError("Raw token stream must contain complete int32 values")
        if mask_path.stat().st_size == 0:
            raise ValueError("Raw cache text mask is empty")
        tokens = np.memmap(token_path, dtype=np.int32, mode="r")
        mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    metadata = dict(total_tokens=total_tokens, total_bytes=total_bytes,
                    tokenizer_sha256=digest, pad_token_id=pad,
                    format="npy" if is_npy else "raw_int32", stream_tokens=int(tokens.size))
    if eos is not None:
        metadata["eos_token_id"] = eos
    if vocab is not None:
        metadata["vocab_size"] = vocab
    if tokens.ndim != 1 or mask.ndim != 1 or len(tokens) != len(mask) or len(tokens) < 64:
        raise ValueError("Cache requires aligned flat token/mask streams with at least 64 tokens")
    if expected_stream is not None and len(tokens) != expected_stream:
        raise ValueError("Validation stream length differs from metadata")
    if tokens.dtype.kind not in "iu" or mask.dtype.kind not in "biu":
        raise ValueError("Cache requires integer tokens and a boolean/integer text mask")
    if mask[0] != 0:
        raise ValueError("The first stream token must be an unscored document separator")
    scored = 0
    for offset in range(0, len(mask), _CHUNK_TOKENS):
        chunk = mask[offset:offset + _CHUNK_TOKENS]
        if np.any((chunk != 0) & (chunk != 1)):
            raise ValueError("Text mask contains values other than 0 and 1")
        scored += int(np.count_nonzero(chunk))
    if scored != metadata["total_tokens"]:
        raise ValueError("Scored tokens or UTF-8 bytes disagree with cache metadata")
    _token_range(tokens, vocab)
    if not is_npy:
        mask = mask.view(np.bool_)
    return tokens, mask, metadata


def validate_packed_identity(metadata, checkpoint_data, vocab_size, tokens=None) -> dict:
    """Check saved cache/checkpoint identities, without opening tokenizer files.

    A valid cache hash is mandatory. A missing checkpoint hash is explicitly
    reported as unavailable; vocabulary/range checks alone do not establish
    that the checkpoint and the data used the same tokenizer. Saved byte counts
    and hashes are provenance, not a reconstruction of the original text.
    """
    vocab_size = _integer(vocab_size, "Model vocabulary size", 1)
    cache_sources = _identity_sources(metadata)
    checkpoint_sources = _identity_sources(checkpoint_data)
    cache_hash, _ = _identity_field(cache_sources, ("tokenizer_sha256", "tokenizer_json_sha256"),
                                    "cache tokenizer SHA-256", _sha256)
    if cache_hash is None:
        raise ValueError("Packed cache metadata is missing tokenizer SHA-256 identity")
    expected_hash, hash_sources = _identity_field(
        checkpoint_sources, ("tokenizer_sha256", "tokenizer_json_sha256"),
        "checkpoint tokenizer SHA-256", _sha256)
    if expected_hash is not None and cache_hash != expected_hash:
        raise ValueError("Cache tokenizer SHA-256 differs from checkpoint tokenizer identity")
    vocab_sources = []
    eos_values = []
    for label, sources in (("cache", cache_sources), ("checkpoint", checkpoint_sources)):
        vocab, fields = _identity_field(sources, ("vocab_size",), f"{label} vocabulary size",
                                        lambda value, name: _integer(value, name, 1))
        if vocab is not None and vocab != vocab_size:
            raise ValueError(f"{label} vocabulary size differs from the model")
        vocab_sources.extend(f"{label}.{field}" for field in fields)
        eos, fields = _identity_field(sources, ("eos_token_id",), f"{label} EOS", _integer)
        if eos is not None:
            if eos >= vocab_size:
                raise ValueError(f"{label} EOS token ID is outside the vocabulary")
            eos_values.append((label, eos, fields))
    if len(eos_values) == 2 and eos_values[0][1] != eos_values[1][1]:
        raise ValueError("Cache EOS token ID differs from checkpoint EOS")
    pad, _ = _identity_field(cache_sources, ("pad_token_id",), "cache PAD", _integer)
    if pad is None or pad >= vocab_size:
        raise ValueError("Cache PAD token ID must be inside the model vocabulary")
    for key in ("total_tokens", "total_bytes"):
        if key not in metadata:
            raise ValueError(f"Packed cache metadata is missing required {key}")
        _integer(metadata[key], key, 1)
    stream_tokens = metadata.get("stream_tokens")
    if stream_tokens is not None:
        stream_tokens = _integer(stream_tokens, "stream_tokens", 64)
        if metadata["total_tokens"] >= stream_tokens:
            raise ValueError("Scored tokens must fit the stream after its unscored initial separator")
    token_check = dict(status="not_checked", reason="tokens were not supplied")
    if tokens is not None:
        token_check = _token_range(tokens, vocab_size)
        if len(tokens) < 64 or metadata["total_tokens"] >= len(tokens):
            raise ValueError("Scored tokens must fit the stream after its unscored initial separator")
        if stream_tokens is not None and stream_tokens != len(tokens):
            raise ValueError("Validation stream length differs from metadata")
    return dict(
        status="validated" if expected_hash is not None else "validated_without_checkpoint_tokenizer_hash",
        cache_tokenizer_sha256=cache_hash,
        cache_tokenizer_hash_status="present_valid_sha256",
        checkpoint_tokenizer_hash_status="validated" if expected_hash is not None else "not_available",
        hash_present_in_checkpoint=expected_hash is not None,
        hash_metadata_checked=hash_sources,
        vocab_size=vocab_size, vocab_metadata_checked=vocab_sources,
        eos_metadata_checked=[f"{label}.{field}" for label, _, fields in eos_values for field in fields],
        eos_token_id=eos_values[0][1] if eos_values else None,
        eos_match_status=("validated" if len(eos_values) == 2 else "not_available"),
        pad_token_id=pad, token_id_range=token_check,
        utf8_bytes_source="saved_cache_metadata; original text not reconstructed",
        tokenizer_files_required=False,
    )
