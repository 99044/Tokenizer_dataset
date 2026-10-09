"""Load model weights for a new stage without restoring training state."""

import torch


def initialize_from_checkpoint(path, model, settings, metadata, stage):
    # Our torch.save checkpoints use ZIP serialization. mmap avoids reading
    # large Stage 1 optimizer tensors when only model weights are needed.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if 'model' not in checkpoint:
        raise ValueError('--init-from 需要包含 model 字段的 checkpoint 文件')
    source_plan = checkpoint.get('plan')
    if stage == 'stage2':
        if not source_plan:
            raise ValueError('Stage 2 初始化需要含 plan 的 Stage 1 checkpoint，以核对模型和数据版本')
        if source_plan.get('stage', 'stage1') != 'stage1':
            raise ValueError('Stage 2 必须从 Stage 1 checkpoint 初始化')
        if not source_plan.get('max_steps') or source_plan['max_steps'] != checkpoint.get('step'):
            raise ValueError('Stage 2 必须从已完成 Stage 1 计划的最终 checkpoint 初始化')
    if source_plan:
        source_data = source_plan['data']
        if source_plan['settings']['model'] != settings['model']:
            raise ValueError('初始化 checkpoint 的模型配置不一致')
        for key in ['vocab_size', 'tokenizer', 'eos_token_id']:
            if source_data.get(key) != metadata.get(key):
                raise ValueError(f'初始化 checkpoint 的 {key} 不一致；不能混用 tokenizer')
        if stage == 'stage2' and source_data.get('data_version') == metadata.get('data_version'):
            raise ValueError('Stage 2 必须使用与 Stage 1 不同的 data_version')
    model.load_state_dict(checkpoint['model'], strict=True)
    return dict(
        checkpoint=str(path), source_step=checkpoint.get('step'),
        source_data_version=source_plan['data'].get('data_version') if source_plan else None,
    )
