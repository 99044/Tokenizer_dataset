#!/usr/bin/env python3
"""Run one configured training job, then export and evaluate its final checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'training/configs/FLOPs_3e17.yaml'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'eval-cache-dir', 'tokenizer-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--max-restarts', type=int, default=20)
    parser.add_argument('--max-stalls', type=int, default=2)
    parser.add_argument('--retry-delay', type=int, default=30)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and args.evaluate_only:
        parser.error('--resume and --evaluate-only cannot be combined')
    args.gpus = args.gpus.split(',')
    if not args.gpus or any(not re.fullmatch(r'\d+', gpu) for gpu in args.gpus):
        parser.error('--gpus must contain distinct GPU IDs')
    args.gpus = [str(int(gpu)) for gpu in args.gpus]
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('--gpus must contain distinct GPU IDs')
    if (args.eval_batch_size < 1 or args.max_restarts < 0 or args.max_stalls < 1
            or args.retry_delay < 0):
        parser.error('eval batch and restart limits must be positive')
    for name in ('config', 'data_dir', 'eval_cache_dir', 'tokenizer_dir', 'output_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def read_config(args):
    settings = yaml.safe_load(args.config.read_text())
    if not isinstance(settings, dict):
        raise ValueError('Training configuration must be a mapping')
    if len(args.gpus) != settings['submission']['data_parallel_size']:
        raise ValueError('GPU count differs from the fixed training and curriculum plan')
    model, training, budget = settings['model'], settings['training'], settings['budget']
    tokens = len(args.gpus) * model['context_length'] * training['micro_batch_size'] * training['gradient_accumulation_steps']
    if tokens != settings['scaling']['tokens_per_update']:
        raise ValueError('Global tokens/update disagree with the YAML')
    if training['max_steps'] * tokens != budget['planned_training_tokens']:
        raise ValueError('Training steps and token budget disagree')
    if budget['planned_training_flops'] > budget['total_training_flops']:
        raise ValueError('Training exceeds the configured FLOPs budget')
    if 'length_curriculum' in training:
        sys.path.insert(0, str(ROOT / 'training/src'))
        from context_curriculum import validate_curriculum
        validate_curriculum(settings, len(args.gpus), training['micro_batch_size'],
                            training['gradient_accumulation_steps'], training['max_steps'])
    return settings


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_identity(args):
    from evaluation.packed_cache import packed_cache_files

    manifests = (args.config, args.data_dir / 'metadata.json',
                 args.eval_cache_dir / 'metadata.json', args.tokenizer_dir / 'tokenizer.json')
    arrays = (args.data_dir / 'train.npy', args.data_dir / 'validation.npy',
              *packed_cache_files(args.eval_cache_dir)[1:])
    source = ('training/run_jd.py', 'training/src/train_ddp.py', 'training/src/gdn_stability.py',
              'training/src/optimizer_factory.py', 'training/src/context_curriculum.py',
              'training/src/fp32_master.py', 'training/src/model_config.py',
              'training/src/compressed_attention.py', 'training/src/dual_residual.py',
              'training/src/value_residual.py', 'training/src/hybrid_export.py',
              'training/export_hf.py', 'evaluation/eval.py',
              'evaluation/eval.sh', 'evaluation/packed_cache.py')
    return dict(
        paths=dict(config=str(args.config), data=str(args.data_dir),
                   evaluation=str(args.eval_cache_dir), tokenizer=str(args.tokenizer_dir)),
        metadata_sha256={str(path): file_hash(path) for path in manifests},
        array_signatures={str(path): dict(size=path.stat().st_size,
                                         mtime_ns=path.stat().st_mtime_ns)
                          for path in arrays},
        source_sha256={name: file_hash(ROOT / name) for name in source},
        gpus=args.gpus, eval_batch_size=args.eval_batch_size,
    )


def preflight(args, settings):
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'training/src'))
    from evaluation.packed_cache import load_packed_cache, validate_packed_identity
    from training.export_hf import read_tokenizer
    from training_data import normalize_data_metadata

    for name in ('metadata.json', 'train.npy', 'validation.npy'):
        if not (args.data_dir / name).is_file():
            raise ValueError(f'Missing training cache file: {name}')
    metadata = json.loads((args.data_dir / 'metadata.json').read_text())
    data = normalize_data_metadata(metadata, settings['model']['context_length'])
    if data['vocab_size'] != settings['model']['vocab_size']:
        raise ValueError('Training cache vocabulary differs from model')
    for name in ('train.npy', 'validation.npy'):
        values = np.load(args.data_dir / name, mmap_mode='r', allow_pickle=False)
        if values.ndim != 1 or values.size <= settings['model']['context_length']:
            raise ValueError(f'Invalid training cache: {name}')
    tokenizer, tokenizer_id = read_tokenizer(args.tokenizer_dir, data)
    tokens, _, eval_metadata = load_packed_cache(args.eval_cache_dir)
    identity = validate_packed_identity(eval_metadata, dict(
        vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
        tokenizer_json_sha256=tokenizer_id['tokenizer_json_sha256'],
    ), settings['model']['vocab_size'], tokens=tokens)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this environment')
    devices = []
    for gpu in args.gpus:
        index = int(gpu)
        if index >= torch.cuda.device_count() or torch.cuda.get_device_properties(index).major < 8:
            raise ValueError(f'GPU {gpu} is unavailable or lacks BF16 support')
        devices.append(torch.cuda.get_device_name(index))
    return dict(gpus=devices, eval_tokens=eval_metadata['total_tokens'],
                eval_bytes=eval_metadata['total_bytes'], tokenizer_identity=identity)


def commands(args, settings, resume=None):
    output = args.output_dir
    step = settings['training']['max_steps']
    checkpoint = output / 'train' / f'step-{step}.pt'
    train = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
             '--local-addr=127.0.0.1', f'--nproc-per-node={len(args.gpus)}',
             str(ROOT / 'training/src/train_ddp.py'), '--config', str(output / 'config.yaml'),
             '--data-dir', str(args.data_dir), '--output-dir', str(output / 'train')]
    if resume is not None:
        train += ['--resume', str(resume)]
    export = [sys.executable, str(ROOT / 'training/export_hf.py'),
              '--checkpoint', str(checkpoint), '--tokenizer-dir', str(args.tokenizer_dir),
              '--output-dir', str(output / 'hf-model'), '--validation-device', 'cuda:0']
    evaluate = ['bash', str(ROOT / 'evaluation/eval.sh'), str(args.eval_cache_dir),
                str(output / 'hf-model'), str(output / 'result.json')]
    return dict(train=train, export=export, evaluate=evaluate)


def run_stage(command, path, environment):
    print(f'{path.name}: {path}', flush=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        return subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode


def latest_checkpoint(directory, target):
    candidates = [path for path in directory.glob('step-*.pt')
                  if re.fullmatch(r'step-\d+\.pt', path.name) and path.stat().st_size
                  and int(path.stem.split('-')[-1]) <= target]
    return max(candidates, key=lambda path: int(path.stem.split('-')[-1]), default=None)


def train_to_completion(args, settings, environment):
    output = args.output_dir
    target = settings['training']['max_steps']
    train_dir = output / 'train'
    train_dir.mkdir(exist_ok=True)
    final = train_dir / f'step-{target}.pt'
    if final.is_file() and final.stat().st_size:
        return final
    if args.evaluate_only:
        raise ValueError('Final checkpoint is missing')
    latest = latest_checkpoint(train_dir, target)
    if latest and not args.resume:
        raise ValueError('Checkpoint already exists; use --resume')
    stalls = restarts = 0
    while True:
        before = latest_checkpoint(train_dir, target)
        command = commands(args, settings, resume=before)['train']
        code = run_stage(command, output / 'train.log', environment)
        if final.is_file() and final.stat().st_size:
            return final
        after = latest_checkpoint(train_dir, target)
        if after is None:
            raise RuntimeError(f'Training exited {code} before its first checkpoint; see train.log')
        stalls = stalls + 1 if after == before else 0
        restarts += 1
        with (output / 'auto-resume.tsv').open('a') as stream:
            stream.write(f'{restarts}\t{code}\t{before}\t{after}\t{stalls}\n')
        if stalls >= args.max_stalls or restarts > args.max_restarts:
            raise RuntimeError(f'Training made no progress or exceeded restart limit; see train.log')
        print(f'Resuming from {after.name} after exit {code}', flush=True)
        time.sleep(args.retry_delay)


def validate_export(output, checkpoint, settings):
    record = json.loads((output / 'export_record.json').read_text())
    if (record.get('status') not in ('passed', 'passed_with_bf16_differences')
            or record.get('source_checkpoint') != str(checkpoint)
            or record.get('source_step') != settings['training']['max_steps']
            or record.get('source_model_config') != settings['model']):
        raise ValueError('HF export validation differs from the final checkpoint')


def postprocess(args, settings, checked, environment, checkpoint):
    output = args.output_dir
    environment = dict(environment, CUDA_VISIBLE_DEVICES=args.gpus[0], PYTHON=sys.executable,
                       NPROC_PER_NODE='1', BATCH_SIZE=str(args.eval_batch_size),
                       FUSION_MODE='ffn_rmsnorm', SEQUENCE_LENGTH='8192')
    hf = output / 'hf-model'
    if not hf.is_dir():
        with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as temporary:
            staging = Path(temporary) / 'hf-model'
            command = commands(args, settings)['export']
            command[command.index('--output-dir') + 1] = str(staging)
            if run_stage(command, output / 'export.log', environment):
                raise RuntimeError('HF export failed; see export.log')
            validate_export(staging, checkpoint, settings)
            staging.rename(hf)
    validate_export(hf, checkpoint, settings)
    result_path = output / 'result.json'
    if not result_path.is_file():
        if run_stage(commands(args, settings)['evaluate'], output / 'eval.log', environment):
            raise RuntimeError('HF evaluation failed; see eval.log')
    result = json.loads(result_path.read_text())
    if (not math.isfinite(result.get('bpb', math.nan))
            or not math.isfinite(result.get('evaltime', math.nan))
            or result.get('model_path') != str(hf)
            or result.get('distributed_world_size') != 1
            or result.get('sequence_length') != 8192
            or result.get('scored_tokens') != checked['eval_tokens']
            or result.get('data_source', {}).get('total_bytes') != checked['eval_bytes']
            or result.get('data_source', {}).get('path') != str(args.eval_cache_dir)
            or result.get('batch_size_per_gpu') != args.eval_batch_size
            or result.get('fusion', {}).get('mode') != 'ffn_rmsnorm'):
        raise ValueError('HF evaluation result does not match the requested cache/settings')
    return dict(status='passed', checkpoint=str(checkpoint), hf_model=str(hf),
                result=str(result_path), bpb=result['bpb'], evaltime=result['evaltime'])


def main(argv=None):
    args = arguments(argv)
    settings = read_config(args)
    plan = dict(config=str(args.config), output=str(args.output_dir), gpus=args.gpus,
                train_steps=settings['training']['max_steps'],
                training_flops=settings['budget']['planned_training_flops'],
                commands=commands(args, settings))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    checked = preflight(args, settings)
    if args.preflight_only:
        print(json.dumps(dict(status='preflight_passed', **checked, **plan), indent=2))
        return 0
    if args.resume or args.evaluate_only:
        if not args.output_dir.is_dir():
            raise ValueError('Output directory does not exist')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    with (output / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = output / 'config.yaml'
        manifest = output / 'experiment.json'
        identity = run_identity(args)
        if args.resume or args.evaluate_only:
            if not snapshot.is_file() or not manifest.is_file():
                raise ValueError('Saved run config or identity is missing')
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('Run input, code, GPU mapping or evaluation changed')
        if snapshot.is_file():
            if yaml.safe_load(snapshot.read_text()) != settings:
                raise ValueError('Saved training config differs; use a new output directory')
        else:
            snapshot.write_text(yaml.safe_dump(settings, sort_keys=False))
            manifest.write_text(json.dumps(identity, indent=2) + '\n')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(args.gpus),
                           TRITON_F32_DEFAULT='ieee', PYTHON=sys.executable)
        environment.setdefault('OMP_NUM_THREADS', '1')
        environment['TOKEN_CACHE_VALIDATION_DIR'] = str(output / 'cache-validation')
        cache = output / 'compiler-cache'
        for name in ('inductor', 'triton', 'tmp'):
            (cache / name).mkdir(parents=True, exist_ok=True)
        environment['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
        environment['TRITON_CACHE_DIR'] = str(cache / 'triton')
        environment['TMPDIR'] = str(cache / 'tmp')
        try:
            checkpoint = train_to_completion(args, settings, environment)
            report = postprocess(args, settings, checked, environment, checkpoint)
        except Exception as error:
            report = dict(status='failed', error=f'{type(error).__name__}: {error}')
            (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            raise
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
