#!/usr/bin/env python3
"""Validate the OLMo Hybrid short-pretraining family and print its exact plan."""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from src.graph_accounting import audit_ledger  # noqa: E402
from src.hybrid_flops import complete_hybrid_ledger, expected_parameter_count  # noqa: E402

EXPECTED_FILES = {
    "hybrid_80m.yaml",
    "hybrid_110m.yaml",
    "hybrid_190m.yaml",
    "hybrid_250m.yaml",
    "hybrid_330m.yaml",
    "hybrid_500m.yaml",
}
COMMON = {
    "run.seed": 42,
    "model.context_length": 8192,
    "model.vocab_size": 32768,
    "model.attention_head_dim": 128,
    "model.block_pattern": ["gdn", "gdn", "gdn", "attn"],
    "training.micro_batch_size": 1,
    "training.gradient_accumulation_steps": 32,
    "training.gradient_checkpointing": False,
    "training.dtype": "bfloat16",
    "optimizer.name": "adamw",
    "optimizer.betas": [0.9, 0.95],
    "optimizer.eps": 1.0e-8,
    "optimizer.weight_decay": 0.1,
    "scheduler.name": "constant",
    "scheduler.warmup_steps": 20,
    "scheduler.cooldown_steps": 0,
    "logging.eval_every_steps": 24,
    "logging.eval_batches": 8,
    "logging.checkpoint_every_steps": 48,
    "logging.project": "scicomp26-scaling",
    "data.format": "npy_packed_tokens",
    "data.requires_document_boundaries": True,
}


def nested(settings: dict, dotted: str):
    value = settings
    for key in dotted.split("."):
        value = value[key]
    return value


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate(path: Path, data_parallel_size: int, scalar_op_cost: int) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    for name, expected in COMMON.items():
        actual = nested(settings, name)
        if actual != expected:
            raise ValueError(f"{path}: {name}={actual!r}, expected {expected!r}")

    model = settings["model"]
    training = settings["training"]
    scaling = settings["scaling"]
    if model["type"] != "olmo_hybrid_scaling":
        raise ValueError(f"{path}: model.type must be olmo_hybrid_scaling")
    vocab_size = positive_integer(model["vocab_size"], f"{path}: model.vocab_size")
    micro_batch_size = positive_integer(
        training["micro_batch_size"], f"{path}: training.micro_batch_size"
    )
    tokens_per_update = (
        data_parallel_size
        * micro_batch_size
        * training["gradient_accumulation_steps"]
        * model["context_length"]
    )
    if scaling["tokens_per_update"] != tokens_per_update:
        raise ValueError(f"{path}: tokens_per_update does not match the 2-GPU batch")
    target_tokens = positive_integer(
        scaling["target_training_tokens"], f"{path}: scaling.target_training_tokens"
    )
    max_steps = math.ceil(target_tokens / tokens_per_update)
    if training["max_steps"] != max_steps:
        raise ValueError(f"{path}: max_steps must be ceil(target_tokens/tokens_per_update)")
    planned_tokens = max_steps * tokens_per_update
    if scaling["planned_training_tokens"] != planned_tokens:
        raise ValueError(f"{path}: planned_training_tokens does not match max_steps")
    if not target_tokens <= planned_tokens < target_tokens + tokens_per_update:
        raise ValueError(f"{path}: planned token rounding is outside one update")
    checkpoint_tokens = settings["logging"]["checkpoint_every_steps"] * tokens_per_update
    if scaling["checkpoint_interval_tokens"] != checkpoint_tokens:
        raise ValueError(f"{path}: checkpoint_interval_tokens is inconsistent")
    expected_lr = round(2e-4 * math.sqrt(1792 / model["hidden_size"]), 6)
    if settings["optimizer"]["learning_rate"] != expected_lr:
        raise ValueError(f"{path}: learning_rate does not follow the recorded scaling rule")

    ledger = complete_hybrid_ledger(
        path, settings, micro_batch_size, vocab_size, scalar_op_cost
    )
    audit = audit_ledger(
        ledger, path, settings, micro_batch_size, vocab_size, ROOT
    )
    parameters = expected_parameter_count(ledger["model_shape"], vocab_size)
    if scaling["expected_total_parameters"] != parameters:
        raise ValueError(f"{path}: expected_total_parameters={scaling['expected_total_parameters']}, calculated {parameters}")
    flops_per_token = Fraction(
        audit["totals"]["total"], micro_batch_size * model["context_length"]
    )
    total_flops = flops_per_token * planned_tokens
    if total_flops.denominator != 1:
        raise ValueError(f"{path}: planned FLOPs are not integral")
    if settings["budget"]["total_training_flops"] != total_flops.numerator:
        raise ValueError(
            f"{path}: total_training_flops={settings['budget']['total_training_flops']}, "
            f"calculated {total_flops.numerator}"
        )
    return {
        "config": path.name,
        "parameters": parameters,
        "tokens": planned_tokens,
        "steps": max_steps,
        "flops_per_token": float(flops_per_token),
        "total_flops": total_flops.numerator,
        "learning_rate": settings["optimizer"]["learning_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "training" / "configs" / "scaling",
    )
    parser.add_argument("--data-parallel-size", type=int, default=2)
    parser.add_argument("--scalar-op-cost", type=int, default=1)
    args = parser.parse_args()
    if args.data_parallel_size <= 0 or args.scalar_op_cost < 0:
        parser.error("data parallel size must be positive and scalar op cost nonnegative")

    paths = sorted(args.config_dir.glob("*.yaml"))
    names = {path.name for path in paths}
    if names != EXPECTED_FILES:
        raise ValueError(
            f"scaling config set differs: missing={sorted(EXPECTED_FILES - names)}, "
            f"unexpected={sorted(names - EXPECTED_FILES)}"
        )
    records = [validate(path, args.data_parallel_size, args.scalar_op_cost) for path in paths]
    if len({record["parameters"] for record in records}) != len(records):
        raise ValueError("scaling parameter counts must be unique")

    print("config\tparameters\tsteps\ttokens\tFLOPs/token\ttotal FLOPs\tpeak LR")
    for record in sorted(records, key=lambda item: item["parameters"]):
        print(
            f"{record['config']}\t{record['parameters']}\t{record['steps']}\t"
            f"{record['tokens']}\t{record['flops_per_token']:.3f}\t"
            f"{record['total_flops']}\t{record['learning_rate']:.6g}"
        )
    total = sum(record["total_flops"] for record in records)
    print(f"TOTAL\t\t\t\t\t{total}\t")
    if total > 2_000_000_000_000_000_000:
        raise ValueError("stage A exceeds the 2e18 exploratory FLOPs budget")


if __name__ == "__main__":
    main()
