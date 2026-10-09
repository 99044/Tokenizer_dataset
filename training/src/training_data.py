"""Read legacy training metadata and native flat packed data with label masks."""

PACKED_FORMAT = 'flat_npy_token_stream_with_is_text_sidecar'
MASK_RULE = 'after causal shift, ignore targets whose aligned is_text is 0; temporary PAD targets are -100'


def data_vocab_size(metadata):
    if 'vocab_size' in metadata:
        value = metadata['vocab_size']
    elif metadata.get('format') == PACKED_FORMAT:
        value = metadata['tokenizer']['vocab_size']
    else:
        raise ValueError('数据 metadata 缺少词表大小')
    value = int(value)
    if value <= 0:
        raise ValueError('数据词表大小必须为正数')
    return value


def normalize_data_metadata(metadata, context_length):
    result = dict(metadata)
    if metadata.get('format') != PACKED_FORMAT:
        if metadata['context_length'] != context_length:
            raise ValueError('数据 context_length 与模型配置不一致')
        return result
    if metadata.get('status') != 'generated_and_verified':
        raise ValueError('原生打包数据必须已完整生成并验证')
    sequence = metadata['sequence']
    if not sequence['padding_policy'].startswith('no_padding_in_flat_npy'):
        raise ValueError('原生打包数据必须是不含 padding 的连续 token 流')
    arrays = metadata['arrays']
    for key, expected in [('train', 'train.npy'), ('validation', 'validation.npy'),
                          ('train_is_text', 'train_is_text.npy'),
                          ('validation_is_text', 'validation_is_text.npy')]:
        if arrays[key] != expected:
            raise ValueError(f'原生打包数据的 {key} 文件名必须为 {expected}')
    if arrays['label_mask_rule'] != MASK_RULE:
        raise ValueError('不支持原生打包数据的标签掩码规则')
    tokenizer = metadata['tokenizer']
    vocab_size = data_vocab_size(metadata)
    if not 0 <= tokenizer['eos_token_id'] < vocab_size:
        raise ValueError('数据 EOS ID 超出词表范围')
    stats = metadata['splits']
    for split in ['train', 'validation']:
        if stats[split]['stream_tokens'] < context_length + 1:
            raise ValueError(f'{split} 数据不足一个训练序列')
    result.update(
        tokenizer_details=tokenizer, tokenizer=tokenizer['revision'],
        vocab_size=vocab_size, eos_token_id=tokenizer['eos_token_id'],
        original_packing_context_length=sequence['context_length'],
        context_length=context_length, loss_mask='is_text',
        train_tokens=stats['train']['stream_tokens'],
        validation_tokens=stats['validation']['stream_tokens'],
    )
    return result
