#!/usr/bin/env python3
"""Export native OLMo3/Hybrid checkpoints or repack HF models for offline loading."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

TOKENIZER_FILES = (
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'added_tokens.json', 'tokenizer.model', 'vocab.json', 'merges.txt',
    'training_metadata.json',
)
TOKENIZER_PROBES = (
    '中文分词测试：科学计算与预训练。', 'Hello, world!\nSecond line.',
    '  spaces\tand\r\nnewlines  ', 'café Ελληνικά 🙂',
    '<|scicomp26_bos_v1|><|scicomp26_eos_v1|><|scicomp26_pad_v1|>',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_tokenizer(source, data):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
    if len(tokenizer) != data['vocab_size']:
        raise ValueError('tokenizer vocabulary size differs from checkpoint')
    token_hash = sha256_file(source / 'tokenizer.json')
    details = data.get('tokenizer_details') or data.get('tokenizer') or {}
    expected_hash = details.get('tokenizer_json_sha256') if isinstance(details, dict) else None
    if expected_hash and token_hash != expected_hash:
        raise ValueError('tokenizer.json SHA-256 differs from checkpoint data metadata')
    if tokenizer.eos_token_id != data['eos_token_id']:
        raise ValueError('EOS token ID differs from checkpoint data metadata')
    for name in ('unk', 'bos', 'eos', 'pad'):
        value = getattr(tokenizer, f'{name}_token_id')
        if value is None or not 0 <= value < len(tokenizer):
            raise ValueError(f'invalid {name} token ID')
    metadata_path = source / 'training_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    expected_revision = data.get('tokenizer')
    if isinstance(expected_revision, dict):
        expected_revision = expected_revision.get('revision')
    if expected_revision and metadata.get('revision') != expected_revision:
        raise ValueError('tokenizer revision differs from checkpoint data metadata')
    for name, token_id in metadata.get('special_token_ids', {}).items():
        if getattr(tokenizer, f'{name}_token_id') != token_id:
            raise ValueError(f'{name} token ID differs from tokenizer training metadata')
    return tokenizer, {'revision': metadata.get('revision'), 'tokenizer_json_sha256': token_hash}


def preserve_tokenizer(source, output, tokenizer):
    """Keep the original backend/config bytes; add standalone BPE vocab/merges."""
    hashes = {}
    for name in TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            if path.stat().st_size == 0:
                raise ValueError(f'empty tokenizer file: {path}')
            shutil.copyfile(path, output / name)
            hashes[name] = sha256_file(path)
    if tokenizer.backend_tokenizer.model.__class__.__name__ == 'BPE':
        # Only supplement files absent from the source; never rewrite originals.
        with tempfile.TemporaryDirectory() as temporary:
            for filename in tokenizer.backend_tokenizer.model.save(temporary):
                path = Path(filename)
                if not (output / path.name).exists():
                    shutil.copyfile(path, output / path.name)
    for name, expected in hashes.items():
        if sha256_file(output / name) != expected:
            raise ValueError(f'tokenizer file changed during export: {name}')
    return hashes


def validate_tokenizer(original, output):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    for loader in (AutoTokenizer, PreTrainedTokenizerFast):
        loaded = loader.from_pretrained(output, local_files_only=True)
        if loaded.get_vocab() != original.get_vocab():
            raise ValueError('offline tokenizer vocabulary changed')
        if loaded.special_tokens_map != original.special_tokens_map:
            raise ValueError('offline tokenizer special token mapping changed')
        if loaded.backend_tokenizer.to_str() != original.backend_tokenizer.to_str():
            raise ValueError('offline tokenizer backend changed')
        for text in TOKENIZER_PROBES:
            for add_special_tokens in (False, True):
                ids = original.encode(text, add_special_tokens=add_special_tokens)
                if loaded.encode(text, add_special_tokens=add_special_tokens) != ids:
                    raise ValueError('offline tokenizer encoding changed')
                if loaded.decode(ids) != original.decode(ids):
                    raise ValueError('offline tokenizer decoding changed')
    return {'status': 'passed', 'vocab_size': len(original), 'special_token_ids': {
        name: getattr(original, f'{name}_token_id') for name in ('unk', 'bos', 'eos', 'pad')
    }}


def logits_metrics(reference, candidate, atol, rtol):
    """Compare all logits in bounded chunks, including nonfinite-value checks."""
    import torch

    if reference.shape != candidate.shape:
        raise ValueError('native and HF logits shapes differ')
    maximum = absolute_sum = squared_difference = squared_reference = 0.0
    mismatches = 0
    for start in range(0, reference.shape[1], 128):
        expected = reference[:, start:start + 128].float()
        actual = candidate[:, start:start + 128].float()
        if not (torch.isfinite(expected).all() and torch.isfinite(actual).all()):
            raise ValueError('nonfinite logits')
        difference = (actual - expected).abs()
        maximum = max(maximum, difference.max().item())
        absolute_sum += difference.sum(dtype=torch.float64).item()
        squared_difference += difference.double().square().sum().item()
        squared_reference += expected.double().square().sum().item()
        mismatches += (difference > atol + rtol * expected.abs()).sum().item()
    return {
        'passed': mismatches == 0, 'atol': atol, 'rtol': rtol,
        'max_absolute_error': maximum, 'mean_absolute_error': absolute_sum / reference.numel(),
        'relative_l2_error': math.sqrt(squared_difference / max(squared_reference, 1e-30)),
        'mismatched_logits': mismatches, 'total_logits': reference.numel(),
    }


def load_offline_model(output, trust_remote_code=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.from_pretrained(output, local_files_only=True,
                               trust_remote_code=trust_remote_code).get_text_config()
    model, loading = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True, trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True,
    )
    if any(loading.get(name) for name in (
        'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs',
    )):
        raise ValueError(f'incomplete HF weight loading: {loading}')
    if model.config.tie_word_embeddings:
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise ValueError('HF input/output embeddings are not tied')
    return model.eval(), {name: sorted(value) if isinstance(value, set) else value
                          for name, value in loading.items()}


def convert_qwen3_moe_state(_config, native_state):
    if any(not name.startswith('lm.') for name in native_state):
        raise ValueError('Qwen3 MoE checkpoint keys must belong to the native lm wrapper')
    return {name.removeprefix('lm.'): tensor for name, tensor in native_state.items()}


def export_checkpoint(args):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, GenerationConfig

    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

    # The checkpoint is our own trusted torch.save artifact. mmap avoids reading
    # optimizer tensors while exporting model weights.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    plan = checkpoint['plan']
    settings = plan['settings']
    model_type = settings['model']['type']
    hybrid = model_type in {'olmo_hybrid_scaling', 'olmo_hybrid_1b'}
    qwen_moe = model_type == 'qwen3_moe'
    compressed = settings['model'].get('compressed_attention') is not None
    if model_type != 'olmo3_1b' and not hybrid and not qwen_moe:
        raise ValueError(f'unsupported native export architecture: {model_type}')
    training_length = settings['model']['context_length']
    data = plan['data']
    if qwen_moe:
        from training_data import normalize_data_metadata

        data = normalize_data_metadata(data, training_length)
        if data['tokenizer_details']['tokenizer_json_sha256'] != settings['data']['tokenizer_json_sha256']:
            raise ValueError('Qwen3 MoE checkpoint cache and model tokenizer identities differ')
    tokenizer, token_record = read_tokenizer(args.tokenizer_dir, data)
    max_length = args.max_sequence_length or training_length
    native_state = checkpoint['model']
    ema_variant = getattr(args, 'ema', None)
    if ema_variant:
        from weight_ema import select_ema_weights
        native_state = select_ema_weights(checkpoint, ema_variant)
    if not qwen_moe and settings['model']['tie_word_embeddings']:
        embedding, projection = native_state['embeddings.weight'], native_state['lm_head.w_out.weight']
        if embedding.data_ptr() != projection.data_ptr() and not torch.equal(embedding, projection):
            raise ValueError('checkpoint declares tied embeddings but stores different weights')
    if hybrid:
        from hybrid_export import NativeHybridCPUReference, convert_hybrid_state, hybrid_hf_config

        hf_config = hybrid_hf_config(settings, len(tokenizer))
        convert_state_to_hf = convert_hybrid_state
    # CPU Hybrid checks use an independent token recurrence. Native FLA imports
    # require a GPU driver even before executing kernels; never fake that import.
    cpu_reference = hybrid and torch.device(args.validation_device).type == 'cpu'
    if cpu_reference:
        core = NativeHybridCPUReference(settings, native_state)
    else:
        from model_config import build_model_config, tie_output_weights

        # Use the architecture saved with the weights, not today's YAML.
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / 'checkpoint.yaml'
            config_path.write_text(yaml.safe_dump(settings))
            core_config, _, _ = build_model_config(config_path, len(tokenizer))
        # Qwen's rotary frequencies are nonpersistent buffers. Construct them
        # on CPU so they remain available after assigning the saved weights.
        core = core_config.build(init_device='cpu' if qwen_moe else 'meta')
        if compressed:
            from compressed_attention import attach_compressed_attention
            attach_compressed_attention(core, settings, torch.device('meta'))
        if hybrid:
            from dual_residual import attach_dual_residual
            from value_residual import attach_value_residual

            attach_dual_residual(core, settings)
            attach_value_residual(core, settings)
        core.load_state_dict(native_state, strict=True, assign=True)
        if not qwen_moe:
            tie_output_weights(core)
    if qwen_moe:
        from copy import deepcopy

        hf_config = deepcopy(core.lm.config)
        convert_state_to_hf = convert_qwen3_moe_state
    elif not hybrid:
        from olmo_core.nn.hf import convert_state_to_hf, get_hf_config

        hf_config = get_hf_config(core)
    hf_config.max_position_embeddings = max_length
    hf_config.bos_token_id = tokenizer.bos_token_id
    hf_config.eos_token_id = tokenizer.eos_token_id
    hf_config.pad_token_id = tokenizer.pad_token_id
    hf_config.dtype = torch.bfloat16
    converted = {name: value.contiguous() for name, value in
                 convert_state_to_hf(hf_config, native_state).items()}
    for name, value in converted.items():
        fp32_parameter = hybrid and (
            name.endswith('.o_norm.weight') or '.dual_residual_mixing.' in name
        )
        expected_dtype = torch.float32 if fp32_parameter else torch.bfloat16
        if value.dtype != expected_dtype:
            raise ValueError(f'checkpoint weight {name} has dtype {value.dtype}, expected {expected_dtype}')
    windowed_moe = qwen_moe and getattr(hf_config, 'local_attention_window', None)
    trust_local_model_code = hybrid or bool(windowed_moe)
    with torch.device('meta'):
        if windowed_moe:
            model_class = type(core.lm)
            model_class.register_for_auto_class('AutoModelForCausalLM')
            hf = model_class._from_config(hf_config, dtype=torch.bfloat16, attn_implementation='sdpa')
        else:
            hf = AutoModelForCausalLM.from_config(hf_config, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa')
    hf.load_state_dict(converted, strict=True, assign=True)
    hf.tie_weights()
    hf.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, do_sample=False,
        **({'use_cache': False} if hybrid else {}),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        'status': 'validating', 'source_checkpoint': str(args.checkpoint),
        'source_step': checkpoint['step'], 'source_model_config': settings['model'],
        'source_weights': ema_variant or 'raw',
        'training_context_length': training_length, 'max_position_embeddings': max_length,
        'model_type': hf_config.model_type, 'mapped_tensor_count': len(converted),
        'dtype': ('bfloat16_with_fp32_gdn_norms_and_residual_mixing'
                  if hybrid and getattr(hf_config, 'dual_residual', None)
                  else 'bfloat16_with_fp32_gdn_norms' if hybrid else 'bfloat16'),
        'trust_remote_code_required': trust_local_model_code,
        'validation_reference': (
            'native_qwen3_moe' if qwen_moe else
            'unfused_native_cpu_recurrence' if cpu_reference else 'native_olmo_core'
        ),
        'tokenizer': token_record, 'logits_validation': [],
        'validation_policy': 'Exact weights and original dtypes, plus strict FP32 logits required; '
                             'BF16 logits differences are recorded separately.',
        'versions': {name: importlib.metadata.version(name) for name in (
            'torch', 'transformers', 'ai2-olmo-core', 'tokenizers',
        )},
    }
    report_path = args.output_dir / 'export_record.json'
    try:
        hf.save_pretrained(args.output_dir, safe_serialization=True,
                           max_shard_size=args.max_shard_size)
        token_record['source_file_sha256'] = preserve_tokenizer(
            args.tokenizer_dir, args.output_dir, tokenizer,
        )
        record['tokenizer_validation'] = validate_tokenizer(tokenizer, args.output_dir)
        del hf, converted, native_state, checkpoint
        gc.collect()
        hf, record['offline_weight_loading'] = load_offline_model(
            args.output_dir, trust_remote_code=trust_local_model_code,
        )
        expected_state = convert_state_to_hf(hf_config, core.state_dict())
        loaded_state = hf.state_dict()
        for name, expected in expected_state.items():
            if expected.dtype != loaded_state[name].dtype or not torch.equal(expected, loaded_state[name]):
                raise ValueError(f'exported weight differs from native checkpoint: {name} '
                                 f'({expected.dtype} -> {loaded_state[name].dtype})')
        record['weight_validation'] = {'status': 'passed', 'verified_tensors': len(expected_state)}
        del expected_state, loaded_state
        device = torch.device(args.validation_device)
        core = core.to(device).eval()
        hf = hf.to(device).eval()
        default_lengths = (16, 64) if qwen_moe else (32, 1024, max_length)
        lengths = args.validation_lengths or [min(length, max_length) for length in default_lengths]
        if hybrid and not args.validation_lengths:
            lengths.extend(length for length in (63, 64, 65, 129) if length <= max_length)
        window = getattr(hf_config, 'local_attention_window', None) or getattr(hf_config, 'sliding_window', None)
        if not args.validation_lengths and window and window < max_length:
            lengths.append(window + 1)
        if any(not 0 < length <= max_length for length in lengths):
            raise ValueError('validation length must be positive and <= max sequence length')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for dtype in (torch.bfloat16, torch.float32):
            # FP32 checks mapping and architecture without BF16 attention rounding
            # amplification. Saved weights always remain the original BF16 values.
            # Both are loaded with BF16 weights already. Calling .bfloat16()
            # would also quantize HF's FP32 RoPE frequency buffers, changing the
            # forward pass. Only widen to FP32 for the reference comparison.
            if dtype == torch.float32:
                core.float()
                hf.float()
                if not cpu_reference and not qwen_moe:
                    # FlashAttention 2 does not accept FP32. Keep the native
                    # modules and RoPE, but use OLMo's SDPA backend for this
                    # high-precision comparison only.
                    from olmo_core.nn.attention import AttentionBackendName

                    for block in core.blocks.values():
                        attention = block.attention
                        if hasattr(attention, 'base'):
                            attention = attention.base
                        if not hasattr(attention, 'backend'):
                            continue
                        window = attention.window_size
                        attention.backend = AttentionBackendName.torch.build(
                            head_dim=attention.head_dim, n_heads=attention.n_heads,
                            n_kv_heads=attention.n_kv_heads, dropout_p=0.0,
                            window_size=(window - 1, 0) if window is not None else (-1, -1),
                        )
                    record['fp32_native_attention_backend'] = 'torch_sdpa'
            generator = torch.Generator().manual_seed(42)
            for length in sorted(set(lengths)):
                inputs = torch.randint(len(tokenizer), (1, length), generator=generator).to(device)
                with torch.inference_mode():
                    if qwen_moe:
                        native_logits = core.lm(input_ids=inputs, use_cache=False).logits
                    else:
                        native_logits = core(inputs)
                    hf_logits = hf(inputs, use_cache=False, return_dict=True).logits
                    if tuple(hf_logits.shape) != (1, length, len(tokenizer)):
                        raise ValueError('HF model did not return full causal LM logits')
                    metrics = logits_metrics(native_logits, hf_logits, args.logits_atol, args.logits_rtol)
                metrics.update(sequence_length=length, shape=list(hf_logits.shape),
                               device=str(device), dtype=str(dtype))
                record['logits_validation'].append(metrics)
                report_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
                print(json.dumps(metrics), flush=True)
                del native_logits, hf_logits, inputs
                if dtype == torch.float32 and not metrics['passed']:
                    raise ValueError(f'FP32 native/HF logits comparison failed at length {length}')
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        bf16_differs = any(not metrics['passed'] for metrics in record['logits_validation']
                           if metrics['dtype'] == 'torch.bfloat16')
        record['status'] = 'passed_with_bf16_differences' if bf16_differs else 'passed'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(f'Native checkpoint export validation: {record["status"]}: {args.output_dir}')


def repack_hf(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.source_hf, local_files_only=True, trust_remote_code=True, dtype='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.source_hf, local_files_only=True, trust_remote_code=True,
    )
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    model.generation_config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True, trust_remote_code=True)
    del model
    gc.collect()
    load_offline_model(args.output_dir, trust_remote_code=True)
    print(f'Offline HF reload passed: {args.output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--source-hf', type=Path)
    parser.add_argument('--tokenizer-dir', type=Path)
    parser.add_argument('--ema', help='Export an EMA variant from the checkpoint')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-shard-size', default='2GB')
    parser.add_argument('--max-sequence-length', type=int)
    parser.add_argument('--validation-device', default='cpu')
    parser.add_argument('--validation-lengths', type=int, nargs='+')
    parser.add_argument('--logits-atol', type=float, default=1e-4)
    parser.add_argument('--logits-rtol', type=float, default=1e-4)
    args = parser.parse_args()
    if args.ema and not args.checkpoint:
        parser.error('--ema requires --checkpoint')
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(
            path.name != '.gitkeep' or not path.is_file() or path.stat().st_size != 0
            for path in args.output_dir.iterdir()
        )
    ):
        parser.error('输出目录必须为空（允许空 .gitkeep）；导出不会覆盖已有文件')
    if args.max_sequence_length is not None and args.max_sequence_length <= 0:
        parser.error('--max-sequence-length 必须大于 0')
    if args.validation_lengths and any(length <= 0 for length in args.validation_lengths):
        parser.error('--validation-lengths 必须大于 0')
    if any(not math.isfinite(value) or value < 0 for value in (args.logits_atol, args.logits_rtol)):
        parser.error('logits 容差必须是非负有限数值')
    if args.checkpoint:
        if not args.checkpoint.is_file() or args.tokenizer_dir is None or not args.tokenizer_dir.is_dir():
            parser.error('--checkpoint 需要有效检查点文件和 --tokenizer-dir')
        export_checkpoint(args)
    else:
        if not args.source_hf.is_dir():
            parser.error('--source-hf 必须是已转换的 HF 模型目录')
        repack_hf(args)


if __name__ == '__main__':
    main()
