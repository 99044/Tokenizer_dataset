"""Compute optimizer steps from a token budget, independently of GPU libraries."""

import math
from pathlib import Path
import sys

# Reuse the submission's FLOPs counter; never maintain a second formula here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from count_flops import (
    PDF_TOLERANCE_DENOMINATOR, PDF_TOLERANCE_NUMERATOR,
    model_spec_from_config, megatron_dense_flops_per_step, positive_flops,
)

STAGE_TOKENS = {'stage1': 1_000_000_000, 'stage2': 100_000_000}
GLOBAL_BATCH_TOKENS = 2_097_152


def resolve_training_plan(stage, world_size, context_length, micro_batch_size=1,
                          grad_accum_steps=None, training_tokens=None,
                          global_batch_tokens=None, max_steps=None):
    for value in [world_size, context_length, micro_batch_size]:
        if value <= 0:
            raise ValueError('GPU 数、context_length 和 micro batch 必须为正数')
    if stage not in STAGE_TOKENS:
        raise ValueError('未知训练阶段')
    if training_tokens is None:
        training_tokens = STAGE_TOKENS[stage]
    if global_batch_tokens is None:
        global_batch_tokens = GLOBAL_BATCH_TOKENS
    for value in [training_tokens, global_batch_tokens, grad_accum_steps, max_steps]:
        if value is not None and value <= 0:
            raise ValueError('token 预算、目标 batch、梯度累积和步数必须为正数')
    micro_tokens = world_size * micro_batch_size * context_length
    if grad_accum_steps is None:
        grad_accum_steps = (global_batch_tokens + micro_tokens - 1) // micro_tokens
    tokens_per_step = micro_tokens * grad_accum_steps
    budget_steps = (training_tokens + tokens_per_step - 1) // tokens_per_step
    if max_steps is not None and max_steps != budget_steps:
        raise ValueError('--max-steps/--steps 与 token 预算计算的步数不一致；请调整 --training-tokens')
    max_steps = budget_steps
    return dict(
        stage=stage, world_size=world_size, micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps, max_steps=max_steps,
        training_tokens=training_tokens, target_global_batch_tokens=global_batch_tokens,
        tokens_per_step=tokens_per_step, planned_training_tokens=max_steps * tokens_per_step,
    )


def resolve_flops_training_plan(stage, world_size, model, vocab_size, target_flops,
                                micro_batch_size=1, grad_accum_steps=None,
                                global_batch_tokens=None):
    """Fit 8192-token Stage 1 updates under the cap and within the supplied tolerance."""
    if stage != 'stage1':
        raise ValueError('--target-flops 当前仅支持 stage1；多阶段须拆分总预算')
    if isinstance(target_flops, bool) or not isinstance(target_flops, int) or target_flops <= 0:
        raise ValueError('target_flops 必须为正整数')
    context = model['context_length']
    provisional = resolve_training_plan(
        stage, world_size, context, micro_batch_size, grad_accum_steps,
        training_tokens=1, global_batch_tokens=global_batch_tokens,
    )
    # The shared builder defaults to MHA if no separate KV-head count is declared.
    groups = model.get('num_key_value_heads', model['num_attention_heads'])
    spec = model_spec_from_config(model, groups, vocab_size)
    batch = world_size * micro_batch_size * provisional['grad_accum_steps']
    cost = megatron_dense_flops_per_step(spec, batch)['total']
    steps = target_flops // cost
    if steps < 1:
        raise ValueError('FLOPs 预算不足一次完整更新；请减小 global batch 或提高预算')
    planned_flops = steps * cost
    lower = target_flops * (PDF_TOLERANCE_DENOMINATOR - PDF_TOLERANCE_NUMERATOR) // PDF_TOLERANCE_DENOMINATOR
    upper = target_flops * (PDF_TOLERANCE_DENOMINATOR + PDF_TOLERANCE_NUMERATOR) // PDF_TOLERANCE_DENOMINATOR
    if not lower <= planned_flops <= upper:
        raise ValueError(
            'planned FLOPs fall outside the PDF ±1% interval; '
            '请减小 --grad-accum-steps 或 --global-batch-tokens 后重新核算'
        )
    tokens = steps * provisional['tokens_per_step']
    plan = resolve_training_plan(
        stage, world_size, context, micro_batch_size, provisional['grad_accum_steps'],
        training_tokens=tokens, global_batch_tokens=global_batch_tokens, max_steps=steps,
    )
    plan.update(
        target_training_flops=target_flops, flops_per_step=cost,
        flops_per_token=cost // plan['tokens_per_step'],
        planned_training_flops=planned_flops, flops_rounding='floor',
        flops_protocol='scicomp26-pdf-megatron-reference-dense-8192-v2',
    )
    return plan


def default_warmup_steps(stage, max_steps):
    if stage not in STAGE_TOKENS:
        raise ValueError('未知训练阶段')
    return min(max_steps - 1, math.ceil(max_steps * 0.05))


def validate_stage_data(stage, metadata):
    if stage not in STAGE_TOKENS:
        raise ValueError('未知训练阶段')
    for key in ['data_version', 'tokenizer']:
        if not metadata.get(key):
            raise ValueError(f'正式训练 metadata 必须记录 {key}')
    description = ' '.join(str(metadata.get(key, '')) for key in ['data_version', 'source', 'notes']).lower()
    if 'smoke' in description or 'synthetic' in description:
        raise ValueError('训练阶段必须使用正式数据，不能使用小样本或合成数据')
