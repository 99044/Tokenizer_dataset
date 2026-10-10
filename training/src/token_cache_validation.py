"""Persist full array validation; reuse it only for unchanged files and metadata."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

from training_data import PACKED_FORMAT, data_vocab_size, normalize_data_metadata

# Tokenization and packing are unchanged from this reviewed implementation.
# Keep its recipe identity stable when validation/reporting code changes. This
# value identifies the recipe baseline; it does not check the executing script.
TOKENIZATION_RECIPE_SHA256 = '283afc880a3270447a9ac8706d5dff5321c7cc20afeade2da74f90e1cc07c881'
SCHEMA = 1
BLOCK_BYTES = 8 * 1024 * 1024


def file_signature(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def metadata_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_spec(path, metadata, vocab_size):
    if vocab_size <= 0:
        raise ValueError('Vocabulary size must be positive')
    if metadata and data_vocab_size(metadata) != vocab_size:
        raise ValueError('Token cache vocabulary does not match the model')
    prefix = 'raw-jsonl-token-cache-v1-'
    if str(metadata.get('data_version', '')).startswith(prefix):
        request = metadata.get('preprocessing', {})
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if metadata['data_version'] != prefix + fingerprint:
            raise ValueError('Token cache request fingerprint mismatch')
    is_mask = path.stem.endswith('_is_text')
    split = path.stem.removesuffix('_is_text')
    count = metadata.get('splits', {}).get(split, {}).get('stream_tokens', metadata.get(f'{split}_tokens'))
    native = metadata.get('format') == PACKED_FORMAT
    if native:
        tokenizer = metadata.get('tokenizer', {})
        if metadata.get('status') != 'generated_and_verified':
            raise ValueError('Token cache was not completely generated')
        if tokenizer.get('vocab_size') != vocab_size:
            raise ValueError('Token cache tokenizer vocabulary does not match the model')
        if not 0 <= tokenizer.get('eos_token_id', -1) < vocab_size:
            raise ValueError('Token cache EOS ID is outside the vocabulary')
    dtype = np.dtype('u1' if is_mask else '<u4') if native else None
    scored = metadata.get('splits', {}).get(split, {}).get('scored_text_tokens') if is_mask else None
    return dict(kind='is_text' if is_mask else 'tokens', count=count, dtype=dtype, scored=scored)


def check_header(path, array, spec):
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'{path} must contain a one-dimensional integer array')
    if spec['dtype'] is not None and array.dtype != spec['dtype']:
        raise ValueError(f'Invalid cache dtype: {path}')
    if spec['count'] is not None and array.size != spec['count']:
        raise ValueError(f'Invalid cache token count: {path}')
    if path.stat().st_size != array.offset + array.nbytes:
        raise ValueError(f'Incomplete or trailing cache data: {path}')


def check_values(path, values, spec, vocab_size):
    minimum, maximum = values['min'], values['max']
    limit = 2 if spec['kind'] == 'is_text' else vocab_size
    if minimum is not None and (minimum < 0 or maximum >= limit):
        raise ValueError(f'{path} contains a value outside [0, {limit})')
    if spec['scored'] is not None and values.get('ones') != spec['scored']:
        raise ValueError(f'Cache is_text count does not match metadata: {path}')


def record_for_array(path, metadata, vocab_size, values):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    spec = array_spec(path, metadata, vocab_size)
    check_header(path, array, spec)
    check_values(path, values, spec, vocab_size)
    return dict(schema=SCHEMA, status='verified', metadata_sha256=metadata_digest(metadata),
                file=file_signature(path), filename=path.name, kind=spec['kind'],
                vocab_size=vocab_size, dtype=array.dtype.str, count=array.size,
                validated_at_utc=datetime.now(timezone.utc).isoformat(), **values)


def record_directory(path):
    return Path(os.environ['TOKEN_CACHE_VALIDATION_DIR']) if os.environ.get('TOKEN_CACHE_VALIDATION_DIR') else path.parent / '.validation'


def atomic_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def matching_record(path, array, metadata, vocab_size, spec):
    # Generated records travel with the cache; a writable external directory
    # holds refreshed records when the dataset itself is mounted read-only.
    directories = dict.fromkeys((record_directory(path), path.parent / '.validation'))
    signature = file_signature(path)
    for directory in directories:
        try:
            record = json.loads((directory / (path.name + '.json')).read_text())
            if (record['schema'] == SCHEMA and record['status'] == 'verified'
                    and record['metadata_sha256'] == metadata_digest(metadata)
                    and record['file'] == signature and record['filename'] == path.name
                    and record['kind'] == spec['kind'] and record['vocab_size'] == vocab_size
                    and record['dtype'] == array.dtype.str and record['count'] == array.size):
                check_values(path, record, spec, vocab_size)
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def scan_array(path, metadata, vocab_size):
    """Read every value once to validate token ranges and mask counts."""
    before = file_signature(path)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    spec = array_spec(path, metadata, vocab_size)
    check_header(path, array, spec)
    minimum = maximum = None
    ones = 0
    consumed = 0
    started = last_report = time.monotonic()
    print(f'Full cache validation: {path.name} ({array.nbytes / 2**30:.2f} GiB)', file=sys.stderr, flush=True)
    with path.open('rb') as stream:
        stream.seek(array.offset)
        for block in iter(lambda: stream.read(BLOCK_BYTES), b''):
            values = np.frombuffer(block, dtype=array.dtype)
            if values.size:
                low, high = int(values.min()), int(values.max())
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                if spec['kind'] == 'is_text':
                    ones += int(np.count_nonzero(values == 1))
                check_values(path, dict(min=minimum, max=maximum), dict(spec, scored=None), vocab_size)
            consumed += len(block)
            now = time.monotonic()
            if now - last_report >= 10:
                rate = consumed / (now - started)
                eta = (array.nbytes - consumed) / rate
                print(f'Validating {path.name}: {consumed / max(1, array.nbytes):.1%}; '
                      f'{rate / 2**20:.0f} MiB/s; remaining ~{eta:.0f}s', file=sys.stderr, flush=True)
                last_report = now
    if file_signature(path) != before:
        raise ValueError(f'Cache file changed during validation: {path}')
    result = dict(min=minimum, max=maximum, method='full_file_scan')
    if spec['kind'] == 'is_text':
        result['ones'] = ones
    record = record_for_array(path, metadata, vocab_size, result)
    if record['file'] != before:
        raise ValueError(f'Cache file changed during validation: {path}')
    return record


def ensure_array(path, metadata, vocab_size):
    """Return True on reuse; otherwise validate and atomically publish a record."""
    path = Path(path)
    metadata = metadata or {}
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    spec = array_spec(path, metadata, vocab_size)
    check_header(path, array, spec)
    if matching_record(path, array, metadata, vocab_size, spec):
        return True
    directory = record_directory(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / (path.name + '.lock')).open('a')
    except OSError as error:
        raise RuntimeError('Cache needs full validation but its record directory is not writable. '
                           'Run training/train.sh for CPU preflight, or set '
                           'TOKEN_CACHE_VALIDATION_DIR to a persistent writable directory.') from error
    with lock:
        # Concurrent DDP ranks/launches share one scan, then reread the record.
        fcntl.flock(lock, fcntl.LOCK_EX)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        check_header(path, array, spec)
        if matching_record(path, array, metadata, vocab_size, spec):
            return True
        record = scan_array(path, metadata, vocab_size)
        atomic_record(directory / (path.name + '.json'), record)
    return False


def ensure_cache(directory, metadata, vocab_size):
    filenames = ['train.npy', 'validation.npy']
    if metadata.get('format') == PACKED_FORMAT:
        filenames += ['train_is_text.npy', 'validation_is_text.npy']
    result = {name: ensure_array(directory / name, metadata, vocab_size) for name in filenames}
    print(f'Cache validation complete: reused={sum(result.values())}, scanned={len(result) - sum(result.values())}',
          file=sys.stderr, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--model-config', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.data_dir / 'metadata.json').read_text())
    if args.model_config:
        import yaml
        context = yaml.safe_load(args.model_config.read_text())['model']['context_length']
        normalize_data_metadata(metadata, context)
    ensure_cache(args.data_dir, metadata, data_vocab_size(metadata))


if __name__ == '__main__':
    main()
