#!/usr/bin/env python3
"""Train the configured model with torchrun, using one process per GPU."""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model_config import build_model_config, count_parameters, tie_output_weights
from train import PackedTokenSampler


@torch.no_grad()
def evaluate(model, sampler, batch_size, batches, vocab_size, device, seed):
    model.eval()
    sampler.rng = np.random.default_rng(seed)
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for _ in range(batches):
        inputs, labels = sampler.batch(batch_size, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), labels.reshape(-1))
        total[0] += loss
        total[1] += 1
    dist.all_reduce(total)
    model.train()
    return (total[0] / total[1]).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('training/configs/FLOPs_3e17.yaml'))
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=20)
    p.add_argument('--stop-after', type=int, help='在此累计 step 停止，但不改变原学习率计划')
    p.add_argument('--micro-batch-size', type=int, default=1)
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-batches', type=int, default=2)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--resume', type=Path)
    p.add_argument('--wandb', action='store_true', help='Upload metrics to W&B from rank 0')
    p.add_argument('--wandb-entity', help='Override the configured W&B entity')
    p.add_argument('--wandb-project', help='Override the configured W&B project')
    p.add_argument('--wandb-name', help='W&B run name (default: output directory name)')
    args = p.parse_args()
    for name in ['max_steps', 'micro_batch_size', 'grad_accum_steps', 'eval_every', 'eval_batches', 'save_every']:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} 必须为正数')
    if args.learning_rate <= 0 or not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError('检查 learning-rate 和 warmup-steps')
    stop_step = args.stop_after if args.stop_after is not None else args.max_steps
    if not 1 <= stop_step <= args.max_steps:
        raise ValueError('stop-after 必须在 1 到 max-steps 之间')
    if 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('请通过 torchrun 启动此训练入口')

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError('CUDA 不可用，或训练进程数超过可见 GPU 数量')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('本配置需要支持 BF16 的 GPU')
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    wandb_run = None
    try:
        metadata = json.loads((args.data_dir / 'metadata.json').read_text())
        vocab_size = int(metadata['vocab_size'])
        config, context_length, settings = build_model_config(args.config, vocab_size)
        if metadata['context_length'] != context_length:
            raise ValueError('数据 context_length 与模型配置不一致')
        seed = settings['run']['seed']
        torch.manual_seed(seed)
        config.init_seed = seed
        model = config.build(init_device='cpu', device=device, max_seq_len=context_length)
        tie_output_weights(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            betas=tuple(settings['optimizer']['betas']),
            eps=settings['optimizer']['eps'], weight_decay=settings['optimizer']['weight_decay'],
        )

        def lr_factor(completed):
            if completed < args.warmup_steps:
                return (completed + 1) / args.warmup_steps
            progress = (completed - args.warmup_steps) / (args.max_steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        plan = dict(
            world_size=world, max_steps=args.max_steps, warmup_steps=args.warmup_steps,
            micro_batch_size=args.micro_batch_size, grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate, settings=settings, data=metadata,
        )
        start_step = 0
        resumed_rng = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
            if checkpoint['plan'] != plan:
                raise ValueError('恢复时 GPU 数量、训练计划、配置与数据 metadata 必须一致')
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_step = int(checkpoint['step'])
            resumed_rng = checkpoint['rng_by_rank'][rank]
            del checkpoint
        if start_step >= stop_step:
            raise ValueError('恢复位置已经达到 stop-after/max-steps，请检查训练计划')

        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        if resumed_rng:
            torch.set_rng_state(resumed_rng['cpu'])
            torch.cuda.set_rng_state(resumed_rng['cuda'], device=device)
        train = PackedTokenSampler(args.data_dir / 'train.npy', context_length, vocab_size, seed + rank)
        validation = PackedTokenSampler(args.data_dir / 'validation.npy', context_length, vocab_size, seed + rank)
        global_batch = world * args.micro_batch_size * args.grad_accum_steps
        tokens_per_step = global_batch * context_length
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / 'run-plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
            print(f'params={count_parameters(model):,} world_size={world} global_batch={global_batch} tokens_per_step={global_batch * context_length}', flush=True)
        if args.wandb:
            # Propagate initialization failures so other ranks do not wait in training collectives.
            wandb_error = [None]
            if rank == 0:
                try:
                    import wandb

                    logging = settings['logging']
                    wandb_run = wandb.init(
                        entity=args.wandb_entity or logging['entity'],
                        project=args.wandb_project or logging['project'],
                        name=args.wandb_name or args.output_dir.name,
                        config=plan,
                        dir=str(args.output_dir),
                        mode='online',
                        settings=wandb.Settings(init_timeout=60),
                    )
                    if wandb_run is None:
                        raise RuntimeError('W&B did not create a run')
                    run_info = dict(id=wandb_run.id, url=wandb_run.url)
                    (args.output_dir / 'wandb-run.json').write_text(json.dumps(run_info, indent=2))
                    print(f'W&B run: {wandb_run.url}', flush=True)
                except Exception as error:
                    wandb_error[0] = f'{type(error).__name__}: {error}'
            dist.broadcast_object_list(wandb_error, src=0)
            if wandb_error[0]:
                raise RuntimeError(f'W&B initialization failed: {wandb_error[0]}')
        dist.barrier()
        run_started = time.perf_counter()

        for step in range(start_step + 1, stop_step + 1):
            step_started = time.perf_counter()
            # 可由 seed/step/rank 重建采样位置，恢复时不需要保存 NumPy 状态。
            train.rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank]))
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device)
            used_lr = optimizer.param_groups[0]['lr']
            for micro_step in range(args.grad_accum_steps):
                sync = micro_step == args.grad_accum_steps - 1
                with nullcontext() if sync else ddp.no_sync():
                    inputs, labels = train.batch(args.micro_batch_size, device)
                    logits = ddp(inputs)
                    loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), labels.reshape(-1))
                    (loss / args.grad_accum_steps).backward()
                    local_loss += loss.detach() / args.grad_accum_steps
                del inputs, labels, logits, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm).item():
                raise RuntimeError('梯度包含非有限值，请检查数据、学习率和数值精度')
            optimizer.step()
            scheduler.step()
            dist.all_reduce(local_loss)
            # Reading the CUDA loss synchronizes this step before timing it.
            mean_loss = (local_loss / world).item()
            step_seconds = time.perf_counter() - step_started
            metrics = {
                'step': step, 'train/loss': mean_loss, 'lr': used_lr,
                'train/step_seconds': step_seconds,
                'train/tokens_per_second': tokens_per_step / step_seconds,
            }
            if step % args.eval_every == 0 or step == stop_step:
                metrics['validation/loss'] = evaluate(
                    model, validation, args.micro_batch_size, args.eval_batches,
                    vocab_size, device, seed + 100000 + rank,
                )
            metrics['elapsed_seconds'] = time.perf_counter() - run_started
            if rank == 0:
                print(json.dumps(metrics), flush=True)
                with (args.output_dir / 'metrics.jsonl').open('a') as log:
                    log.write(json.dumps(metrics) + '\n')
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

            if step % args.save_every == 0 or step == stop_step:
                rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
                all_rng = [None] * world
                dist.all_gather_object(all_rng, rng)
                if rank == 0:
                    path = args.output_dir / f'step-{step}.pt'
                    temporary = path.with_suffix('.pt.tmp')
                    torch.save(dict(
                        step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), plan=plan, rng_by_rank=all_rng,
                    ), temporary)
                    temporary.replace(path)
                dist.barrier()
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            if wandb_run is not None:
                wandb_run.finish(exit_code=1 if failed else 0)
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
