#!/usr/bin/env python3
"""Tokenize raw JSONL once; stdout contains the reusable cache directory name."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from training_data import MASK_RULE, PACKED_FORMAT, normalize_data_metadata
from token_cache_validation import (TOKENIZATION_RECIPE_SHA256, atomic_record,
                                    ensure_cache, record_for_array)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path: Path) -> dict:
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def validation_document(record: dict, permyriad: int) -> bool:
    value = str(record['id']) if record.get('id') is not None else record['text']
    digest = hashlib.sha256(value.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % 10000 < permyriad


def load_tokenizer(directory: Path) -> tuple:
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config = json.loads((directory / 'tokenizer_config.json').read_text(encoding='utf-8'))
    ids = {}
    for name in ['unk', 'bos', 'eos', 'pad']:
        value = config.get(f'{name}_token')
        if isinstance(value, dict):
            value = value.get('content')
        ids[f'{name}_token_id'] = tokenizer.token_to_id(value) if isinstance(value, str) else None
    if ids['eos_token_id'] is None:
        raise ValueError('tokenizer_config.json must identify a valid EOS token')
    metadata_path = directory / 'training_metadata.json'
    provenance = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    info = dict(ids, revision=provenance.get('revision', directory.name),
                vocab_size=tokenizer.get_vocab_size(), encode_add_special_tokens=False,
                tokenizer_json_sha256=sha256_file(directory / 'tokenizer.json'))
    return tokenizer, info


def write_npy(raw: Path, output: Path, dtype: str, count: int) -> dict:
    """Validate every value while copying, without another disk scan."""
    minimum = maximum = None
    ones = 0
    copied = 0
    with output.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream, {
            'descr': np.lib.format.dtype_to_descr(np.dtype(dtype)),
            'fortran_order': False, 'shape': (count,),
        })
    with output.open('ab') as stream, raw.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            values = np.frombuffer(block, dtype=dtype)
            if values.size:
                low, high = int(values.min()), int(values.max())
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                if dtype == 'u1':
                    ones += int(np.count_nonzero(values == 1))
            stream.write(block)
            copied += len(block)
    if copied != count * np.dtype(dtype).itemsize:
        raise ValueError(f'Raw cache length does not match token count: {raw}')
    raw.unlink()
    result = dict(min=minimum, max=maximum, method='validated_during_npy_write')
    if dtype == 'u1':
        result['ones'] = ones
    return result


def validate_cache(directory: Path, request: dict) -> dict:
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('preprocessing') != request:
        raise ValueError(f'Cache provenance mismatch: {directory}')
    normalize_data_metadata(metadata, request['context_length'])
    for split in ['train', 'validation']:
        count = metadata['splits'][split]['stream_tokens']
        for filename, dtype in [(f'{split}.npy', np.dtype('<u4')),
                                (f'{split}_is_text.npy', np.dtype('u1'))]:
            path = directory / filename
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (count,) or array.dtype != dtype:
                raise ValueError(f'Invalid cache array: {path}')
            del array
    ensure_cache(directory, metadata, metadata['tokenizer']['vocab_size'])
    return metadata


def build_cache(source_root: Path, files: list, tokenizer: Tokenizer, tokenizer_info: dict,
                directory: Path, request: dict, batch_size: int) -> None:
    stats = {split: {'documents_seen': 0, 'empty_documents': 0, 'nonempty_documents': 0,
                     'raw_text_tokens': 0, 'scored_text_tokens': 0, 'inserted_eos_tokens': 0,
                     'stream_tokens': 0, 'utf8_bytes': 0}
             for split in ['train', 'validation']}
    handles = {}
    source_records = []
    started = time.monotonic()
    documents = 0
    next_report = 100000
    log_path = directory / 'packing.log'

    def log(message):
        line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
        print(line, file=sys.stderr, flush=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    def emit(batch):
        encodings = tokenizer.encode_batch([text for _, text in batch], add_special_tokens=False)
        for (split, text), encoding in zip(batch, encodings):
            ids = np.asarray(encoding.ids, dtype='<u4')
            if text and not len(ids):
                raise ValueError('Tokenizer returned no tokens for nonempty text')
            if len(ids) and int(ids.max()) >= tokenizer_info['vocab_size']:
                raise ValueError('Tokenizer emitted an ID outside its vocabulary')
            # Match evaluation.pack_jsonl(): every original text token is a
            # target, even if its ID equals EOS/PAD. Only inserted EOS is masked.
            mask = np.ones(len(ids), dtype=np.uint8)
            token_file, mask_file = handles[split]
            np.asarray([tokenizer_info['eos_token_id']], dtype='<u4').tofile(token_file)
            np.asarray([0], dtype=np.uint8).tofile(mask_file)
            ids.tofile(token_file)
            mask.tofile(mask_file)
            stats[split]['nonempty_documents' if text else 'empty_documents'] += 1
            stats[split]['raw_text_tokens'] += len(ids)
            stats[split]['scored_text_tokens'] += int(mask.sum())
            stats[split]['inserted_eos_tokens'] += 1
            stats[split]['stream_tokens'] += len(ids) + 1
            stats[split]['utf8_bytes'] += len(text.encode('utf-8'))

    try:
        for split in stats:
            handles[split] = ((directory / f'{split}.tokens.raw').open('wb'),
                              (directory / f'{split}.mask.raw').open('wb'))
        log('Starting raw JSONL tokenization; no text normalization, truncation or padding')
        for path in files:
            batch = []
            batch_characters = 0
            with path.open('rb') as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict) or not isinstance(record.get('text'), str):
                            raise ValueError('every record must have a string text field')
                    except (ValueError, UnicodeDecodeError) as error:
                        raise ValueError(f'{path}:{number}: {error}') from error
                    split = 'validation' if validation_document(record, request['validation_permyriad']) else 'train'
                    stats[split]['documents_seen'] += 1
                    documents += 1
                    batch.append((split, record['text']))
                    batch_characters += len(record['text'])
                    # A character bound also prevents a batch of long documents
                    # from consuming unbounded memory; single documents stay whole.
                    if len(batch) >= batch_size or batch_characters >= 4 * 1024 * 1024:
                        emit(batch)
                        batch.clear()
                        batch_characters = 0
                    if documents >= next_report:
                        log(f'documents={documents:,}; rate={documents / (time.monotonic() - started):.0f} docs/s')
                        next_report += 100000
            if batch:
                emit(batch)
            source_records.append(dict(name=path.name, **file_stat(path)))
            log(f'Finished {path.name}; documents={documents:,}')
    finally:
        for pair in handles.values():
            for stream in pair:
                stream.close()

    validation_values = {}
    for split in stats:
        count = stats[split]['stream_tokens']
        if count < request['context_length'] + 1 or not stats[split]['scored_text_tokens']:
            raise ValueError(f'{split} has insufficient text tokens for context_length+1; '
                             'provide more documents or change validation-permyriad')
        validation_values.update({
            f'{split}.npy': write_npy(directory / f'{split}.tokens.raw', directory / f'{split}.npy', '<u4', count),
            f'{split}_is_text.npy': write_npy(directory / f'{split}.mask.raw', directory / f'{split}_is_text.npy', 'u1', count),
        })
    if [dict(name=path.name, **file_stat(path)) for path in sorted(source_root.glob('*.jsonl'))] != request['source_files']:
        raise ValueError('Source files changed during preprocessing')
    if any(sha256_file(Path(request['tokenizer_root']) / name) != digest
           for name, digest in request['tokenizer_files'].items()):
        raise ValueError('Tokenizer files changed during preprocessing')
    filenames = ['train.npy', 'validation.npy', 'train_is_text.npy', 'validation_is_text.npy']
    metadata = {
        'data_version': 'raw-jsonl-token-cache-v1-' + directory.name.removeprefix('.building-'),
        'format': PACKED_FORMAT, 'status': 'generated_and_verified',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'preprocessing': request,
        'source': {'root': request['source_label'], 'files': source_records,
                   'text_policy': 'original text unchanged; no cleaning or deduplication',
                   'split_rule': 'SHA-256(str(id)), fallback text; first 64 bits big-endian % 10000 < validation_permyriad'},
        'tokenizer': dict(tokenizer_info, path=request['tokenizer_label']),
        'sequence': {'context_length': request['context_length'],
                     'padding_policy': 'no_padding_in_flat_npy',
                     'document_eos_policy': 'prepend_exactly_one_eos_before_each_document_including_empty_text',
                     'long_document_policy': 'fully_tokenize; concatenate without truncation'},
        'arrays': {'train': 'train.npy', 'validation': 'validation.npy',
                   'train_is_text': 'train_is_text.npy', 'validation_is_text': 'validation_is_text.npy',
                   'token_dtype': 'uint32', 'is_text_dtype': 'uint8', 'label_mask_rule': MASK_RULE,
                   'is_text_definition': '1 for every original text token; 0 only for inserted EOS'},
        'splits': stats, 'cache_files': {name: file_stat(directory / name) for name in filenames},
        'reproducibility': {'file_order': 'lexical filename order, original line order within each split',
                            'generation_script': 'prepare_token_cache.py', 'external_data': False,
                            'cache_reuse_checks': 'validation records, metadata digest, array headers, size, mtime_ns and ctime_ns'},
    }
    metadata.update(vocab_size=tokenizer_info['vocab_size'],
                    context_length=request['context_length'],
                    eos_token_id=tokenizer_info['eos_token_id'],
                    train_tokens=stats['train']['stream_tokens'],
                    validation_tokens=stats['validation']['stream_tokens'])
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for name, values in validation_values.items():
        record = record_for_array(directory / name, metadata, tokenizer_info['vocab_size'], values)
        atomic_record(directory / '.validation' / (name + '.json'), record)
    validate_cache(directory, request)
    log(f'Complete; train tokens={stats["train"]["stream_tokens"]:,}; validation tokens={stats["validation"]["stream_tokens"]:,}')


def prepare_cache(source_root: Path, tokenizer_dir: Path, output: Path, context_length: int = 8192,
                  validation_permyriad: int = 100, batch_size: int = 128,
                  source_label: str | None = None, tokenizer_label: str | None = None) -> Path:
    if context_length <= 0 or batch_size <= 0 or not 0 < validation_permyriad < 10000:
        raise ValueError('context-length and encode-batch-size must be positive; validation-permyriad must be in 1..9999')
    source_root, tokenizer_dir = source_root.resolve(), tokenizer_dir.resolve()
    files = sorted(source_root.glob('*.jsonl'))
    if not files:
        raise ValueError(f'No .jsonl files under {source_root}')
    tokenizer, tokenizer_info = load_tokenizer(tokenizer_dir)
    tokenizer_files = {name: sha256_file(tokenizer_dir / name)
                       for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'training_metadata.json']
                       if (tokenizer_dir / name).is_file()}
    if tokenizer_info['tokenizer_json_sha256'] != tokenizer_files['tokenizer.json']:
        raise ValueError('Tokenizer changed while it was being loaded')
    request = {'source_root': str(source_root), 'source_label': source_label or str(source_root),
               'source_files': [dict(name=path.name, **file_stat(path)) for path in files],
               'tokenizer_root': str(tokenizer_dir),
               'tokenizer_label': tokenizer_label or str(tokenizer_dir), 'tokenizer_files': tokenizer_files,
               'context_length': context_length, 'validation_permyriad': validation_permyriad,
               'script_sha256': TOKENIZATION_RECIPE_SHA256, 'label_mask_rule': MASK_RULE}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    directory = output / key
    # Concurrent launches share the cache; only one process may build this key.
    with (output / f'.{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            validate_cache(directory, request)
            print(f'Reusing token cache: {directory}', file=sys.stderr, flush=True)
            return directory
        building = output / f'.building-{key}'
        if building.exists():
            shutil.rmtree(building)  # Incomplete cache from an interrupted build.
        building.mkdir()
        try:
            build_cache(source_root, files, tokenizer, tokenizer_info, building, request, batch_size)
            building.rename(directory)  # Publish only after all files are complete.
        except BaseException:
            shutil.rmtree(building)
            raise
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Parent directory for fingerprinted caches')
    parser.add_argument('--context-length', type=int)
    parser.add_argument('--model-config', type=Path, help='Read context_length from the training YAML')
    parser.add_argument('--validation-permyriad', type=int, default=100, help='100 = 1%% held out by document ID hash')
    parser.add_argument('--encode-batch-size', type=int, default=128)
    parser.add_argument('--source-label', help='Original host source path for provenance and cache identity')
    parser.add_argument('--tokenizer-label', help='Original host tokenizer path')
    args = parser.parse_args()
    context = args.context_length
    if args.model_config:
        import yaml
        configured = yaml.safe_load(args.model_config.read_text())['model']['context_length']
        if context is not None and context != configured:
            parser.error('--context-length differs from --model-config')
        context = configured
    directory = prepare_cache(args.source_root, args.tokenizer, args.output,
                              context if context is not None else 8192,
                              args.validation_permyriad, args.encode_batch_size,
                              args.source_label, args.tokenizer_label)
    print(directory.name)


if __name__ == '__main__':
    main()
