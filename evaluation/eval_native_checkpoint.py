#!/usr/bin/env python3
"""Compute canonical BPB directly from a native OLMo-core checkpoint.

This reuses ``pack_jsonl`` and ``packed_batch`` from ``evaluation/eval.py`` so
scaling checkpoints and final exported models use the same UTF-8 byte and label
mask definitions. Launch with torchrun; the GPU count is intentionally dynamic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parent
ROOT = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(ROOT / "training" / "src"))

from eval import pack_jsonl, packed_batch  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_training_progress(checkpoint):
    """Read the saved training topology and variable-length compute schedule."""
    from context_curriculum import curriculum_completed_flops, validate_curriculum

    plan = checkpoint['plan']
    settings = plan['settings']
    step = int(checkpoint['step'])
    if not 0 < step <= int(plan['max_steps']):
        raise ValueError('checkpoint step is outside its training plan')
    tokens_per_update = (
        int(plan['world_size']) * int(plan['micro_batch_size'])
        * int(plan['grad_accum_steps']) * settings['model']['context_length']
    )
    curriculum = settings['training'].get('length_curriculum')
    if curriculum is not None:
        validate_curriculum(settings, plan['world_size'], plan['micro_batch_size'],
                            plan['grad_accum_steps'], plan['max_steps'])
        flops = curriculum_completed_flops(settings, step)
        if checkpoint.get('completed_flops') != flops:
            raise ValueError('checkpoint completed FLOPs differ from the curriculum')
        return step * tokens_per_update, flops
    budget = settings.get('budget')
    flops = None
    if isinstance(budget, dict) and budget.get('total_training_flops') is not None:
        full_run_flops = int(budget.get('planned_training_flops', budget['total_training_flops']))
        value = Fraction(full_run_flops, int(plan['max_steps'])) * step
        if value.denominator != 1:
            raise ValueError('checkpoint FLOPs are not integral under the stored run plan')
        flops = value.numerator
    return step * tokens_per_update, flops


def checkpoint_primary_learning_rate(checkpoint):
    optimizer = checkpoint['optimizer']
    if optimizer.get('kind') == 'soap_adamw':
        optimizer = optimizer['optimizers'][0]
    return float(optimizer['param_groups'][0]['lr'])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True, help="JSONL with a text field")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Native step-*.pt checkpoint")
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences per GPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import yaml
    from transformers import AutoTokenizer

    from model_config import build_model_config, count_parameters, tie_output_weights
    from compressed_attention import attach_compressed_attention
    from dual_residual import attach_dual_residual
    from value_residual import attach_value_residual
    from training_data import data_vocab_size

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    temporary = None
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        checkpoint_step = int(checkpoint["step"])
        plan = checkpoint["plan"]
        settings = plan["settings"]
        optimizer_settings = settings["optimizer"]
        peak_learning_rate = float(optimizer_settings["learning_rate"])
        checkpoint_learning_rate = checkpoint_primary_learning_rate(checkpoint)
        training_tokens, flops_at_checkpoint = checkpoint_training_progress(checkpoint)
        if settings["model"]["type"] not in {"olmo_hybrid_1b", "olmo_hybrid_scaling"}:
            raise ValueError("native BPB evaluation supports the OLMo Hybrid family")
        vocab_size = data_vocab_size(plan["data"])
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
        if len(tokenizer) != vocab_size:
            raise ValueError("tokenizer vocabulary differs from the checkpoint")
        tokenizer_metadata = plan["data"].get("tokenizer_details")
        if not isinstance(tokenizer_metadata, dict):
            tokenizer_metadata = plan["data"].get("tokenizer")
        if isinstance(tokenizer_metadata, dict):
            expected_eos = tokenizer_metadata.get("eos_token_id")
            if expected_eos is not None and tokenizer.eos_token_id != expected_eos:
                raise ValueError("tokenizer EOS ID differs from the checkpoint")
            expected_hash = tokenizer_metadata.get("tokenizer_json_sha256")
            tokenizer_json = args.tokenizer_dir / "tokenizer.json"
            if expected_hash and file_sha256(tokenizer_json) != expected_hash:
                raise ValueError("tokenizer.json differs from the checkpoint")

        shared = [None]
        if rank == 0:
            temporary = tempfile.TemporaryDirectory(prefix=".native-bpb-", dir=args.output.parent)
            cache_dir = Path(temporary.name).resolve()
            metadata = pack_jsonl(args.dataset_path, tokenizer, cache_dir)
            shared[0] = (str(cache_dir), metadata)
        dist.broadcast_object_list(shared, src=0, device=device)
        cache_path, metadata = shared[0]
        tokens = np.memmap(Path(cache_path) / "tokens.bin", dtype=np.int32, mode="r")
        is_text = np.memmap(Path(cache_path) / "is_text.bin", dtype=np.bool_, mode="r")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
            yaml.safe_dump(settings, config_file)
            config_file.flush()
            model_config, sequence_length, _ = build_model_config(
                Path(config_file.name), vocab_size
            )
        model = model_config.build(init_device="meta")
        attach_compressed_attention(model, settings, torch.device('meta'))
        attach_dual_residual(model, settings)
        attach_value_residual(model, settings)
        model.load_state_dict(checkpoint["model"], strict=True, assign=True)
        tie_output_weights(model)
        model = model.to(device).eval()
        parameters = count_parameters(model)
        nonembedding_parameters = parameters - settings["model"]["hidden_size"] * vocab_size
        del checkpoint

        num_sequences = math.ceil((len(tokens) - 1) / sequence_length)
        global_batch_size = args.batch_size * world_size
        num_steps = math.ceil(num_sequences / global_batch_size)
        total_nll = torch.zeros((), device=device, dtype=torch.float64)
        scored_tokens = 0
        with torch.inference_mode():
            for step in range(num_steps):
                start = step * global_batch_size + rank * args.batch_size
                stop = min(start + args.batch_size, num_sequences)
                if start >= stop:
                    continue
                inputs, labels = packed_batch(
                    tokens,
                    is_text,
                    start,
                    stop,
                    sequence_length,
                    metadata["pad_token_id"],
                )
                batch_scored_tokens = int(np.count_nonzero(labels != -100))
                inputs = torch.from_numpy(inputs).to(device)
                labels = torch.from_numpy(labels).to(device)
                logits = model(inputs)
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_labels = labels.reshape(-1)
                for offset in range(0, flat_labels.numel(), 512):
                    target = flat_labels[offset : offset + 512]
                    loss = F.cross_entropy(
                        flat_logits[offset : offset + 512].float(),
                        target,
                        ignore_index=-100,
                        reduction="none",
                    )
                    total_nll += loss.double().sum()
                scored_tokens += batch_scored_tokens
                del inputs, labels, logits, flat_logits, flat_labels, loss

        count = torch.tensor(scored_tokens, device=device, dtype=torch.int64)
        dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() != metadata["total_tokens"]:
            raise RuntimeError("scored token count does not match the raw validation text")
        bpb = total_nll.item() / (math.log(2) * metadata["total_bytes"])
        if not math.isfinite(bpb):
            raise RuntimeError("evaluation produced non-finite BPB")
        if rank == 0:
            result = {
                "checkpoint": str(args.checkpoint),
                "step": checkpoint_step,
                "run_id": settings["run"]["name"],
                "model_type": settings["model"]["type"],
                "N_total": parameters,
                "N_nonembed": nonembedding_parameters,
                "D": training_tokens,
                "FLOPs": flops_at_checkpoint,
                "optimizer": optimizer_settings["name"],
                "learning_rate": peak_learning_rate,
                "checkpoint_learning_rate": checkpoint_learning_rate,
                "scalar_learning_rate": optimizer_settings.get("scalar_learning_rate"),
                "schedule": settings["scheduler"]["name"],
                "seed": settings["run"]["seed"],
                "sequence_length": sequence_length,
                "scored_tokens": count.item(),
                "utf8_bytes": metadata["total_bytes"],
                "total_nll_nats": total_nll.item(),
                "bpb": bpb,
                "definition": "total_nll_nats / (ln(2) * original_utf8_bytes)",
                "packing_implementation": "evaluation/eval.py",
            }
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
        dist.barrier()
        del tokens, is_text
    finally:
        if temporary is not None:
            temporary.cleanup()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
