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

from fp32_master import FP32MasterWeights
from gdn_stability import install_gdn_stability_guard
from compressed_attention import attach_compressed_attention
from dual_residual import attach_dual_residual
from value_residual import attach_value_residual
from context_curriculum import (
    curriculum_completed_flops,
    curriculum_lr_factor,
    curriculum_stage,
    validate_curriculum,
)
from model_config import build_model_config, count_parameters, tie_output_weights
from model_initialization import repair_zero_gdn_decay_rates, scale_projections
from optimizer_factory import build_optimizer
from packed_token_sampler import PackedTokenSampler
from training_options import resolve_training_options


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


def configured_steps(settings, key):
    values = settings.get('logging', {}).get(key, [])
    if values is None:
        return set()
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError(f'logging.{key} 必须是正整数列表')
    return set(values)


def record_gradient_anomaly(model, output_dir, rank, step, grad_norm, loss, lr, precision):
    """Persist a large spike or a nonfinite gradient with parameter names."""
    def safe_scalar(value):
        number = float(value)
        return number if math.isfinite(number) else str(number)

    details = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = bool(torch.isfinite(gradient).all().item())
        absmax = float(gradient.detach().float().abs().max().item()) if finite else None
        details.append({'name': name, 'finite': finite, 'postclip_absmax': absmax})
    largest = sorted((item for item in details if item['finite']),
                     key=lambda item: item['postclip_absmax'], reverse=True)[:12]
    event = {
        'event': 'gradient_anomaly', 'rank': rank, 'step': step,
        'gdn_recurrence_precision': precision,
        'preclip_grad_norm': safe_scalar(grad_norm.item()),
        'local_loss': safe_scalar(loss.item()), 'learning_rate': lr,
        'nonfinite': [item['name'] for item in details if not item['finite']],
        'largest_postclip_gradients': largest,
    }
    path = output_dir / f'gradient-anomalies-rank-{rank}.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
    print(json.dumps(event, allow_nan=False), flush=True)


def validate_branch_checkpoint(checkpoint, settings, metadata, plan):
    continuation = settings.get('continuation')
    if not isinstance(continuation, dict):
        raise ValueError('--branch-from 只适用于包含 continuation 配置的实验')
    expected_step = continuation.get('source_checkpoint_step')
    if checkpoint.get('step') != expected_step:
        raise ValueError(
            f'分支 checkpoint step={checkpoint.get("step")!r}，配置要求 step={expected_step!r}'
        )
    source_plan = checkpoint.get('plan')
    if not isinstance(source_plan, dict):
        raise ValueError('分支 checkpoint 缺少训练 plan')
    source_settings = source_plan.get('settings') or {}
    if source_settings.get('model') != settings.get('model'):
        raise ValueError('分支 checkpoint 的模型配置与当前配置不一致')
    if source_settings.get('optimizer') != settings.get('optimizer'):
        raise ValueError('分支 checkpoint 的 optimizer 配置与当前配置不一致')
    if (
        continuation.get('scheduler_state') == 'preserve'
        and source_settings.get('scheduler', {}).get('name')
        != settings.get('scheduler', {}).get('name')
    ):
        raise ValueError('保留 scheduler 状态时，源配置与当前 scheduler 类型必须一致')
    if source_plan.get('data') != metadata:
        raise ValueError('分支 checkpoint 的数据 metadata 与当前数据不一致')
    if source_settings.get('run', {}).get('seed') != settings.get('run', {}).get('seed'):
        raise ValueError('分支 checkpoint 的随机种子与当前配置不一致')
    source_tokens_per_step = (
        source_plan.get('world_size', 0)
        * source_plan.get('micro_batch_size', 0)
        * source_plan.get('grad_accum_steps', 0)
        * settings['model']['context_length']
    )
    current_tokens_per_step = (
        plan['world_size']
        * plan['micro_batch_size']
        * plan['grad_accum_steps']
        * settings['model']['context_length']
    )
    if source_tokens_per_step != current_tokens_per_step:
        raise ValueError('分支训练必须保持每次参数更新的 token 数不变')
    rng_by_rank = checkpoint.get('rng_by_rank')
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != plan['world_size']:
        raise ValueError('分支训练必须保持 GPU 数量不变，才能恢复各 rank 的 RNG 状态')
    return continuation


def position_scheduler(scheduler, optimizer, completed_steps, lr_factor):
    """Position a fresh scheduler at an absolute step for a branched run."""
    scheduler.last_epoch = completed_steps
    scheduler._step_count = completed_steps + 1
    lrs = [base_lr * lr_factor(completed_steps) for base_lr in scheduler.base_lrs]
    for group, lr in zip(optimizer.param_groups, lrs):
        group['lr'] = lr
    scheduler._last_lr = lrs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('training/configs/FLOPs_3e17.yaml'))
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--stop-after', type=int, help='在此累计 step 停止，但不改变原学习率计划')
    p.add_argument('--micro-batch-size', type=int)
    p.add_argument('--grad-accum-steps', type=int)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--warmup-steps', type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--eval-batches', type=int)
    p.add_argument('--save-every', type=int)
    p.add_argument('--resume', type=Path)
    p.add_argument(
        '--branch-from', type=Path,
        help='Start a declared continuation plan from a Stage A checkpoint',
    )
    p.add_argument('--wandb', action='store_true', help='Upload metrics to W&B from rank 0')
    p.add_argument('--wandb-entity', help='Override the configured W&B entity')
    p.add_argument('--wandb-project', help='Override the configured W&B project')
    p.add_argument('--wandb-name', help='W&B run name (default: output directory name)')
    args = resolve_training_options(p.parse_args())
    if args.resume and args.branch_from:
        raise ValueError('--resume 与 --branch-from 不能同时使用')
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
        validate_curriculum(
            settings, world_size=world, micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps, max_steps=args.max_steps,
        )
        has_curriculum = curriculum_stage(settings, 1) is not None
        if has_curriculum and args.branch_from:
            raise ValueError('上下文课程支持从头训练或 --resume，不能使用 --branch-from')
        if has_curriculum and args.warmup_steps != settings['scheduler']['warmup_steps']:
            raise ValueError('上下文课程按配置的累计 FLOPs 调度，不能覆盖 --warmup-steps')
        seed = settings['run']['seed']
        torch.manual_seed(seed)
        config.init_seed = seed
        model = config.build(init_device='cpu')
        model.init_weights(max_seq_len=context_length, device=device)
        initialization_repairs = repair_zero_gdn_decay_rates(model)
        tie_output_weights(model)
        scaled_parameters = scale_projections(
            model, float(settings.get('initialization', {}).get('projection_scale', 1.0))
        )
        compressed_parameters = attach_compressed_attention(model, settings, device)
        attach_dual_residual(model, settings)
        attach_value_residual(model, settings)
        residual_parameters = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if '.dual_residual_mixing.' in name
        )
        expected_residual = settings.get('scaling', {}).get('additional_residual_parameters')
        if expected_residual is not None and residual_parameters != expected_residual:
            raise ValueError(f'residual parameters {residual_parameters} != {expected_residual}')
        expected_parameters = settings.get('scaling', {}).get('expected_total_parameters')
        if expected_parameters is not None and count_parameters(model) != expected_parameters:
            raise ValueError(f'model parameter count {count_parameters(model)} != {expected_parameters}')
        expected_branch = settings.get('scaling', {}).get('expected_branch_parameters')
        if expected_branch is not None and compressed_parameters != expected_branch:
            raise ValueError(f'compressed branch parameters {compressed_parameters} != {expected_branch}')
        use_fp32_master = bool(settings['training'].get('fp32_master_weights', False))
        if use_fp32_master and settings['optimizer']['name'].lower() not in {'adamw', 'soap', 'muon', 'normuon'}:
            raise ValueError('FP32 master weights require AdamW, SOAP, Muon or NorMuon')
        # DDP broadcasts initial weights before their FP32 copies are created.
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        master_weights = FP32MasterWeights(model) if use_fp32_master else None
        optimizer, optimizer_report = build_optimizer(
            model,
            settings,
            primary_learning_rate=args.learning_rate,
            parameter_map=None if master_weights is None else master_weights.parameter_map,
        )
        if master_weights is not None:
            optimizer = master_weights.install(optimizer)
            optimizer_report['fp32_master_weights'] = True

        scheduler_settings = settings['scheduler']
        scheduler_name = scheduler_settings['name']
        cooldown_steps = int(scheduler_settings.get('cooldown_steps') or 0)
        if not has_curriculum and scheduler_name not in {'constant', 'cosine', 'linear_cooldown'}:
            raise ValueError(f'不支持的 scheduler.name: {scheduler_name!r}')
        continuation = settings.get('continuation')
        if continuation and not (args.branch_from or args.resume):
            raise ValueError('continuation 配置必须使用 --branch-from，或用 --resume 恢复其自身 checkpoint')
        if not has_curriculum and scheduler_name == 'linear_cooldown':
            if cooldown_steps <= 0 or cooldown_steps >= args.max_steps:
                raise ValueError('linear_cooldown 要求 0 < cooldown_steps < max_steps')
            if isinstance(continuation, dict):
                source_step = continuation.get('source_checkpoint_step')
                if cooldown_steps != args.max_steps - source_step or args.warmup_steps != 0:
                    raise ValueError(
                        'checkpoint 分支的 linear_cooldown 要求 '
                        'cooldown_steps=max_steps-source_step 且 warmup_steps=0'
                    )
            else:
                source_step = args.max_steps - cooldown_steps
                if args.warmup_steps >= source_step:
                    raise ValueError('warmup 必须在 linear cooldown 开始前结束')
            final_lr_ratio = float(scheduler_settings.get('final_lr_ratio', 0.0))
            if not 0.0 <= final_lr_ratio < 1.0:
                raise ValueError('scheduler.final_lr_ratio 必须在 [0, 1)')
        elif not has_curriculum and cooldown_steps:
            raise ValueError('cooldown_steps 只能与 linear_cooldown 一起使用')

        def lr_factor(completed):
            if has_curriculum:
                return curriculum_lr_factor(settings, completed)
            if completed < args.warmup_steps:
                return (completed + 1) / args.warmup_steps
            if scheduler_name == 'linear_cooldown':
                if completed <= source_step:
                    return 1.0
                progress = min((completed - source_step) / cooldown_steps, 1.0)
                return 1.0 + (final_lr_ratio - 1.0) * progress
            if scheduler_name == 'constant':
                return 1.0
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
        checkpoint_path = args.resume or args.branch_from
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if args.resume:
            if checkpoint['plan'] != plan:
                raise ValueError('恢复时 GPU 数量、训练计划、配置与数据 metadata 必须一致')
            if has_curriculum and checkpoint.get('completed_flops') != curriculum_completed_flops(
                settings, checkpoint['step']
            ):
                raise ValueError('课程 checkpoint 的累计 FLOPs 与训练进度不一致')
            if master_weights is not None:
                master_weights.validate_checkpoint(checkpoint)
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if master_weights is not None:
                master_weights.load_state_dict(checkpoint['fp32_master'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_step = int(checkpoint['step'])
            resumed_rng = checkpoint['rng_by_rank'][rank]
            del checkpoint
        elif args.branch_from:
            continuation = validate_branch_checkpoint(checkpoint, settings, metadata, plan)
            if master_weights is not None:
                master_weights.validate_checkpoint(checkpoint)
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if master_weights is not None:
                master_weights.load_state_dict(checkpoint['fp32_master'])
            start_step = int(checkpoint['step'])
            resumed_rng = checkpoint['rng_by_rank'][rank]
            scheduler_state = continuation.get('scheduler_state')
            if scheduler_state == 'preserve':
                scheduler.load_state_dict(checkpoint['scheduler'])
            elif scheduler_state == 'restart':
                position_scheduler(scheduler, optimizer, start_step, lr_factor)
            else:
                raise ValueError('continuation.scheduler_state 必须为 preserve 或 restart')
            del checkpoint
        if start_step >= stop_step:
            raise ValueError('恢复位置已经达到 stop-after/max-steps，请检查训练计划')

        if resumed_rng:
            torch.set_rng_state(resumed_rng['cpu'])
            torch.cuda.set_rng_state(resumed_rng['cuda'], device=device)
        train = PackedTokenSampler(args.data_dir / 'train.npy', context_length, vocab_size, seed + rank, metadata)
        train_samplers = {context_length: train}
        validation = PackedTokenSampler(args.data_dir / 'validation.npy', context_length, vocab_size, seed + rank, metadata)
        global_batch = world * args.micro_batch_size * args.grad_accum_steps
        tokens_per_step = global_batch * context_length
        expected_tokens_per_step = settings.get('experiment', {}).get(
            'tokens_per_update', settings.get('scaling', {}).get('tokens_per_update')
        )
        if expected_tokens_per_step is not None and tokens_per_step != expected_tokens_per_step:
            raise ValueError(
                f'实际 tokens/update={tokens_per_step}，配置要求 {expected_tokens_per_step}；'
                '请检查 GPU 数、micro batch 和梯度累积步数'
            )
        eval_steps = configured_steps(settings, 'eval_steps')
        checkpoint_steps = configured_steps(settings, 'checkpoint_steps')
        for label, steps in [('eval_steps', eval_steps), ('checkpoint_steps', checkpoint_steps)]:
            outside = sorted(step for step in steps if step > args.max_steps)
            if outside:
                raise ValueError(f'logging.{label} 不能超过 max_steps={args.max_steps}：{outside}')
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / 'initialization-repairs.json').write_text(
                json.dumps(initialization_repairs, indent=2) + '\n'
            )
            (args.output_dir / 'scaled-projections.json').write_text(
                json.dumps(scaled_parameters, indent=2) + '\n'
            )
            if initialization_repairs:
                print(f'GDN initialization repairs: {json.dumps(initialization_repairs)}', flush=True)
            (args.output_dir / 'run-plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
            (args.output_dir / 'optimizer-groups.json').write_text(
                json.dumps(optimizer_report, indent=2, ensure_ascii=False) + '\n'
            )
            print(f'resolved training options: micro_batch_size={args.micro_batch_size} '
                  f'grad_accum_steps={args.grad_accum_steps} learning_rate={args.learning_rate} '
                  f'warmup_steps={args.warmup_steps} max_steps={args.max_steps} '
                  f'eval_every={args.eval_every} eval_batches={args.eval_batches} '
                  f'save_every={args.save_every}', flush=True)
            if settings['model']['type'] in {'olmo_hybrid_1b', 'olmo_hybrid_scaling'}:
                from hybrid_flops import runtime_graph_metadata

                runtime_metadata = runtime_graph_metadata(model)
                runtime_metadata['compressed_branch_parameters'] = compressed_parameters
                if residual_parameters:
                    runtime_metadata['residual_parameters'] = residual_parameters
                runtime_metadata.update(first_optimizer_step=start_step + 1, planned_stop_step=stop_step)
                runtime_path = args.output_dir / f'runtime_graph-start-{start_step + 1}.json'
                runtime_path.write_text(json.dumps(runtime_metadata, indent=2, ensure_ascii=False) + '\n')
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
        force_fp32_gdn = settings['training'].get('gdn_recurrent_fp32', False)
        if not isinstance(force_fp32_gdn, bool):
            raise ValueError('training.gdn_recurrent_fp32 must be a boolean')
        retry_threshold = settings['training'].get('gdn_fp32_retry_grad_norm')
        if retry_threshold is not None and (
            isinstance(retry_threshold, bool)
            or not isinstance(retry_threshold, (int, float))
            or not math.isfinite(retry_threshold)
            or retry_threshold <= 0
        ):
            raise ValueError('training.gdn_fp32_retry_grad_norm must be a positive finite number')
        recovery_steps = settings['training'].get('gdn_fp32_recovery_steps', 16)
        if isinstance(recovery_steps, bool) or not isinstance(recovery_steps, int) or recovery_steps < 1:
            raise ValueError('training.gdn_fp32_recovery_steps must be a positive integer')
        gdn_guard = install_gdn_stability_guard(args.output_dir, rank, force_fp32_gdn)
        if rank == 0:
            print(f'GDN recurrence precision: {"fp32" if force_fp32_gdn else "model_dtype_with_fallback"}; '
                  f'fp32_retry_grad_norm={retry_threshold}; fp32_recovery_steps={recovery_steps}', flush=True)
        fp32_recovery_until = 0

        for step in range(start_step + 1, stop_step + 1):
            step_started = time.perf_counter()
            stage = curriculum_stage(settings, step)
            sequence_length = context_length if stage is None else stage['length']
            micro_batch_size = args.micro_batch_size if stage is None else stage['micro_batch_size']
            grad_accum_steps = args.grad_accum_steps if stage is None else stage['grad_accum_steps']
            if sequence_length not in train_samplers:
                train_samplers[sequence_length] = PackedTokenSampler(
                    args.data_dir / 'train.npy', sequence_length, vocab_size, seed + rank, metadata,
                )
            train = train_samplers[sequence_length]
            used_lr = float(optimizer.param_groups[0]['lr'])
            used_group_lrs = [float(group['lr']) for group in optimizer.param_groups]
            step_cpu_rng = torch.get_rng_state()
            step_cuda_rng = torch.cuda.get_rng_state(device)
            fp32_step = force_fp32_gdn or step <= fp32_recovery_until
            retried_fp32 = False
            while True:
                if retried_fp32:
                    # The retry must use the same tokens and stochastic model state.
                    torch.set_rng_state(step_cpu_rng)
                    torch.cuda.set_rng_state(step_cuda_rng, device=device)
                gdn_guard.force_fp32 = fp32_step
                # Sampling positions are reproducible from seed/step/rank.
                train.rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank]))
                optimizer.zero_grad(set_to_none=True)
                local_loss = torch.zeros((), device=device)
                for micro_step in range(grad_accum_steps):
                    gdn_guard.set_context(step, micro_step)
                    sync = micro_step == grad_accum_steps - 1
                    with nullcontext() if sync else ddp.no_sync():
                        inputs, labels = train.batch(micro_batch_size, device)
                        logits = ddp(inputs)
                        loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), labels.reshape(-1))
                        (loss / grad_accum_steps).backward()
                        local_loss += loss.detach() / grad_accum_steps
                    del inputs, labels, logits, loss
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                local_bad = not bool(torch.isfinite(grad_norm).item()) or not bool(torch.isfinite(local_loss).item())
                local_spike = retry_threshold is not None and grad_norm.item() > retry_threshold
                if retry_threshold is not None:
                    bad_on_any_rank = torch.tensor(int(local_bad or local_spike), device=device)
                    dist.all_reduce(bad_on_any_rank, op=dist.ReduceOp.MAX)
                    bad_on_any_rank = bool(bad_on_any_rank.item())
                else:
                    bad_on_any_rank = local_bad
                if bad_on_any_rank and retry_threshold is not None and not fp32_step:
                    if local_bad or local_spike:
                        record_gradient_anomaly(model, args.output_dir, rank, step, grad_norm,
                                                local_loss, used_lr, 'model_dtype')
                    # No optimizer update has occurred. Drop every bad gradient and
                    # recompute the entire step with the GDN recurrence in IEEE FP32.
                    retried_fp32 = True
                    fp32_step = True
                    fp32_recovery_until = step + recovery_steps - 1
                    continue
                if bad_on_any_rank:
                    record_gradient_anomaly(model, args.output_dir, rank, step, grad_norm,
                                            local_loss, used_lr, 'fp32' if fp32_step else 'model_dtype')
                    raise RuntimeError('FP32 GDN 步的梯度仍异常；详见 gradient-anomalies-rank-*.jsonl'
                                       if fp32_step else '梯度包含非有限值；详见 gradient-anomalies-rank-*.jsonl')
                break
            gdn_guard.force_fp32 = force_fp32_gdn
            optimizer.step()
            scheduler.step()
            dist.all_reduce(local_loss)
            mean_loss = (local_loss / world).item()
            # Dion may use auxiliary CUDA streams. Synchronize explicitly so the
            # optimizer calibration measures the complete training update.
            torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_started
            metrics = {
                'step': step, 'train/loss': mean_loss, 'lr': used_lr,
                'train/tokens': step * tokens_per_step,
                'train/step_seconds': step_seconds,
                'train/tokens_per_second': tokens_per_step / step_seconds,
                'train/grad_norm': grad_norm.item(),
                'train/gdn_fp32_retried': int(retried_fp32),
                'train/gdn_fp32_recovery_step': int(fp32_step and not retried_fp32),
                'system/max_memory_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'system/max_memory_reserved_bytes': torch.cuda.max_memory_reserved(device),
            }
            for index, group in enumerate(optimizer_report['groups']):
                metrics[f"lr/{group['name']}"] = used_group_lrs[index]
            stage_boundary = False
            if stage is not None:
                metrics.update({
                    'train/context_length': sequence_length,
                    'train/micro_batch_size': micro_batch_size,
                    'train/grad_accum_steps': grad_accum_steps,
                    'train/curriculum_stage': stage['stage'],
                    'train/curriculum_stage_index': stage['stage_index'],
                    'train/completed_flops': curriculum_completed_flops(settings, step),
                })
                stage_boundary = step == stage['last_step']
            if step % args.eval_every == 0 or step in eval_steps or step == stop_step:
                eval_batches = args.eval_batches
                if step == args.max_steps:
                    eval_batches = int(
                        settings.get('logging', {}).get('final_eval_batches', eval_batches)
                    )
                    if eval_batches <= 0:
                        raise ValueError('logging.final_eval_batches 必须为正整数')
                validation_loss = evaluate(
                    model, validation, args.micro_batch_size, eval_batches,
                    vocab_size, device, seed + 100000 + rank,
                )
                metrics['validation/loss'] = validation_loss
                metrics['validation/token_nll'] = validation_loss
                metrics['validation/batches'] = eval_batches
            metrics['elapsed_seconds'] = time.perf_counter() - run_started
            if rank == 0:
                print(json.dumps(metrics), flush=True)
                with (args.output_dir / 'metrics.jsonl').open('a') as log:
                    log.write(json.dumps(metrics) + '\n')
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

            if step % args.save_every == 0 or step in checkpoint_steps or step == stop_step or stage_boundary:
                rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
                all_rng = [None] * world
                dist.all_gather_object(all_rng, rng)
                if rank == 0:
                    path = args.output_dir / f'step-{step}.pt'
                    temporary = path.with_suffix('.pt.tmp')
                    checkpoint_payload = dict(
                        step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), plan=plan, rng_by_rank=all_rng,
                    )
                    if master_weights is not None:
                        checkpoint_payload['fp32_master'] = master_weights.state_dict()
                    if stage is not None:
                        checkpoint_payload.update(
                            completed_flops=curriculum_completed_flops(settings, step),
                            curriculum_stage=stage['stage'], sequence_length=sequence_length,
                        )
                    torch.save(checkpoint_payload, temporary)
                    temporary.replace(path)
                    keep_last = settings.get('logging', {}).get('checkpoint_keep_last')
                    if keep_last is not None:
                        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
                            raise ValueError('logging.checkpoint_keep_last must be a positive integer')
                        checkpoints = sorted(
                            args.output_dir.glob('step-*.pt'),
                            key=lambda item: int(item.stem.split('-')[-1]),
                        )
                        for old_path in checkpoints[:-keep_last]:
                            old_path.unlink()
                dist.barrier()
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            if wandb_run is not None:
                wandb_run.finish(exit_code=1 if failed else 0)
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    if os.environ.get('SCICOMP_DIAGNOSTICS') == '1':
        from diagnostic_hooks import install
        install()
    main()
