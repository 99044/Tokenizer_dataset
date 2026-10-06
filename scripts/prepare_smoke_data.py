#!/usr/bin/env python3
"""CPU preprocessing; no GPU or tokenizer training is required."""

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser(description="Encode the published smoke samples with the existing tokenizer.")
parser.add_argument('--source-dir', type=Path, default=Path('/home/pretrain10/dolma3_official/github_staging/scicomp26_data_v0.1/data/splits_v2'))
parser.add_argument('--tokenizer-dir', type=Path, default=Path('/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k'))
parser.add_argument('--output-dir', type=Path, default=Path('/home/zhangrui/comp/data/splits_v2-tok32k-smoke-ctx1024-v1'))
parser.add_argument('--reuse', action='store_true', help='Validate and reuse a completed output directory')
args = parser.parse_args()
samples, tokenizer_path, out = args.source_dir, args.tokenizer_dir, args.output_dir

if out.exists():
    if not args.reuse:
        raise FileExistsError(f'{out} already exists; use --reuse to validate it')
    metadata = json.loads((out / 'metadata.json').read_text())
    for key, expected in [('vocab_size', 32768), ('eos_token_id', 2), ('context_length', 1024)]:
        if metadata[key] != expected:
            raise ValueError(f'{out}: incompatible {key}')
    for split, records in [('train', 10000), ('validation', 1000)]:
        tokens = np.load(out / f'{split}.npy', mmap_mode='r')
        if tokens.ndim != 1 or tokens.dtype != np.uint32 or tokens.size < 1025:
            raise ValueError(f'{out}: invalid {split} array')
        if tokens.size != metadata[f'{split}_tokens'] or int(tokens.max()) >= 32768:
            raise ValueError(f'{out}: inconsistent {split} token IDs/count')
        if metadata['split_stats'][split]['records'] != records:
            raise ValueError(f'{out}: not the expected smoke dataset')
    if not (out / 'tokenizer' / 'tokenizer.json').is_file():
        raise FileNotFoundError(out / 'tokenizer' / 'tokenizer.json')
    print(f'Reusing validated smoke data: {out}', flush=True)
    raise SystemExit(0)

sources = {
    'train': [samples / 'sample_train_10000.jsonl.zst'],
    'validation': [samples / 'sample_dev_1000.jsonl.zst'],
}

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    str(tokenizer_path), local_files_only=True,
)
assert len(tokenizer) == 32768
assert tokenizer.eos_token_id == 2
for paths in sources.values():
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
out.mkdir(parents=True, exist_ok=False)
shutil.copytree(tokenizer_path, out / 'tokenizer')

def convert(split, paths):
    raw_path = out / f'{split}.raw.tmp'
    token_count = record_count = empty_count = 0
    with raw_path.open('wb') as raw:
        for path in paths:
            # 逐条解压和编码，不把整份语料装进内存。
            with subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE) as proc:
                try:
                    for line in proc.stdout:
                        record = json.loads(line)
                        text = record['text']
                        if not isinstance(text, str):
                            raise TypeError(f'{path}: text must be a string')
                        record_count += 1
                        if text == '':
                            empty_count += 1
                            continue
                        ids = tokenizer(
                            text, add_special_tokens=False,
                            truncation=False, padding=False,
                            # Preserve the full document; PackedTokenSampler limits model inputs.
                            verbose=False,
                        )['input_ids']
                        if any(token < 0 or token >= 32768 for token in ids):
                            raise ValueError(f'{path}: invalid token ID')
                        tokens = np.asarray([2] + ids, dtype=np.uint32)
                        raw.write(tokens.tobytes())
                        token_count += tokens.size
                except BaseException:
                    proc.terminate()
                    raise
                finally:
                    proc.stdout.close()
                if proc.wait() != 0:
                    raise RuntimeError(f'zstd failed: {path}')
    if token_count < 1025:
        raise ValueError(f'{split}: not enough tokens for context length 1024')

    # np.load 要求 .npy 头；不能将原始二进制文件直接改名为 .npy。
    source = np.memmap(raw_path, dtype=np.uint32, mode='r', shape=(token_count,))
    target = np.lib.format.open_memmap(
        out / f'{split}.npy', mode='w+', dtype=np.uint32, shape=(token_count,),
    )
    chunk_size = 1_000_000
    for start in range(0, token_count, chunk_size):
        target[start:start + chunk_size] = source[start:start + chunk_size]
    target.flush()
    del target, source
    raw_path.unlink()
    print(f'{split}: records={record_count}, empty={empty_count}, tokens={token_count}')
    return {'records': record_count, 'empty_records': empty_count, 'tokens': token_count}

stats = {split: convert(split, paths) for split, paths in sources.items()}
assert stats['train']['records'] == 10000
assert stats['validation']['records'] == 1000
metadata = {
    'data_version': out.name,
    'source': 'splits_v2_cleaned release smoke samples',
    'source_files': {split: [str(path) for path in paths] for split, paths in sources.items()},
    'tokenizer': 'tok_bpe32k_v1',
    'vocab_size': 32768,
    'eos_token_id': 2,
    'train_tokens': stats['train']['tokens'],
    'validation_tokens': stats['validation']['tokens'],
    'context_length': 1024,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'split_stats': stats,
    'notes': 'Unmodified text; one EOS before each nonempty document; no BOS, PAD or truncation; smoke data only.',
}
(out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
print(f'Prepared: {out}')
