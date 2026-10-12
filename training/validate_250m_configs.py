#!/usr/bin/env python3
"""Validate the four fixed-architecture OLMo Hybrid 250M training plans."""

from __future__ import annotations

import argparse
import copy
import sys
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from src.graph_accounting import audit_ledger  # noqa: E402
from src.hybrid_flops import complete_hybrid_ledger, expected_parameter_count  # noqa: E402


TARGETS = {
    "FLOPs_3e17.yaml": 300_000_000_000_000_000,
    "FLOPs_1e18.yaml": 1_000_000_000_000_000_000,
    "FLOPs_3e18.yaml": 3_000_000_000_000_000_000,
    "FLOPs_1e19.yaml": 10_000_000_000_000_000_000,
}
EXPECTED_PARAMETERS = 254_467_368
EXPECTED_FLOPS_PER_TOKEN = Fraction(1_800_713_490_309, 1024)
DATA_PARALLEL_SIZE = 2


def load_yaml(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return settings


def require_equal(path: Path, name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{path}: {name}={actual!r}, expected {expected!r}")


def validate(path: Path, target_flops: int, scalar_op_cost: int) -> dict:
    settings = load_yaml(path)
    model = settings["model"]
    training = settings["training"]
    optimizer = settings["optimizer"]
    scheduler = settings["scheduler"]
    scaling = settings["scaling"]
    budget = settings["budget"]

    require_equal(path, "budget.total_training_flops", budget["total_training_flops"], target_flops)
    require_equal(path, "model.type", model["type"], "olmo_hybrid_scaling")
    require_equal(path, "model.context_length", model["context_length"], 8192)
    require_equal(path, "model.vocab_size", model["vocab_size"], 32768)
    require_equal(path, "training.micro_batch_size", training["micro_batch_size"], 1)
    require_equal(path, "training.gradient_accumulation_steps", training["gradient_accumulation_steps"], 32)
    require_equal(path, "optimizer.learning_rate", optimizer["learning_rate"], 0.000283)
    require_equal(path, "scheduler.name", scheduler["name"], "linear_cooldown")
    require_equal(path, "scheduler.warmup_steps", scheduler["warmup_steps"], 20)
    require_equal(path, "scheduler.final_lr_ratio", scheduler["final_lr_ratio"], 0.0)
    require_equal(path, "scaling.family", scaling["family"], "olmo_hybrid_v1")
    require_equal(path, "scaling.nominal_size", scaling["nominal_size"], "250m")

    tokens_per_update = (
        DATA_PARALLEL_SIZE
        * training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        * model["context_length"]
    )
    require_equal(path, "scaling.tokens_per_update", scaling["tokens_per_update"], tokens_per_update)

    ledger = complete_hybrid_ledger(
        path,
        settings,
        training["micro_batch_size"],
        model["vocab_size"],
        scalar_op_cost,
    )
    audit = audit_ledger(
        ledger,
        path,
        settings,
        training["micro_batch_size"],
        model["vocab_size"],
        ROOT,
    )
    parameters = expected_parameter_count(ledger["model_shape"], model["vocab_size"])
    require_equal(path, "scaling.expected_total_parameters", scaling["expected_total_parameters"], parameters)
    require_equal(path, "calculated parameter count", parameters, EXPECTED_PARAMETERS)

    flops_per_token = Fraction(
        audit["totals"]["total"],
        training["micro_batch_size"] * model["context_length"],
    )
    require_equal(path, "calculated FLOPs/token", flops_per_token, EXPECTED_FLOPS_PER_TOKEN)
    require_equal(path, "budget.flops_per_token_numerator", budget["flops_per_token_numerator"], flops_per_token.numerator)
    require_equal(path, "budget.flops_per_token_denominator", budget["flops_per_token_denominator"], flops_per_token.denominator)

    flops_per_update = flops_per_token * tokens_per_update
    if flops_per_update.denominator != 1:
        raise ValueError(f"{path}: FLOPs/update must be integral")
    max_steps = target_flops // flops_per_update.numerator
    require_equal(path, "training.max_steps", training["max_steps"], max_steps)
    planned_tokens = max_steps * tokens_per_update
    planned_flops = max_steps * flops_per_update.numerator
    require_equal(path, "budget.planned_training_tokens", budget["planned_training_tokens"], planned_tokens)
    require_equal(path, "budget.planned_training_flops", budget["planned_training_flops"], planned_flops)
    if not 99 * target_flops <= 100 * planned_flops <= 100 * target_flops:
        raise ValueError(f"{path}: planned FLOPs must be within 1% of the budget without exceeding it")

    # B3 used 58 cooldown updates out of 345 total updates. Keep that measured
    # fraction when scaling the schedule to each official compute budget.
    cooldown_steps = round(max_steps * 58 / 345)
    require_equal(path, "scheduler.cooldown_steps", scheduler["cooldown_steps"], cooldown_steps)
    if scheduler["warmup_steps"] >= max_steps - cooldown_steps:
        raise ValueError(f"{path}: warmup must finish before cooldown starts")

    return {
        "config": path.name,
        "steps": max_steps,
        "tokens": planned_tokens,
        "flops": planned_flops,
        "cooldown": cooldown_steps,
        "settings": settings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "training" / "configs",
    )
    parser.add_argument("--scalar-op-cost", type=int, default=1)
    args = parser.parse_args()
    if args.scalar_op_cost < 0:
        parser.error("scalar op cost must be nonnegative")

    records = [
        validate(args.config_dir / name, target, args.scalar_op_cost)
        for name, target in TARGETS.items()
    ]
    reference = copy.deepcopy(records[0]["settings"])
    for key in ("run", "budget", "scheduler", "output"):
        reference.pop(key)
    reference["training"].pop("max_steps")
    for record in records[1:]:
        candidate = copy.deepcopy(record["settings"])
        for key in ("run", "budget", "scheduler", "output"):
            candidate.pop(key)
        candidate["training"].pop("max_steps")
        require_equal(args.config_dir / record["config"], "shared training configuration", candidate, reference)

    print("config\tparameters\tsteps\ttokens\tcooldown\tplanned FLOPs")
    for record in records:
        print(
            f"{record['config']}\t{EXPECTED_PARAMETERS}\t{record['steps']}\t"
            f"{record['tokens']}\t{record['cooldown']}\t{record['flops']}"
        )


if __name__ == "__main__":
    main()
