#!/usr/bin/env python3
"""Validate the three OLMo Hybrid Stage B continuation plans."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from src.graph_accounting import audit_ledger  # noqa: E402
from src.hybrid_flops import complete_hybrid_ledger, expected_parameter_count  # noqa: E402


STAGE_A_FLOPS = 1_687_924_844_350_648_320
EXPERIMENTS = {
    "b1_hybrid_330m_constant.yaml": {
        "id": "B1",
        "source": "hybrid_330m.yaml",
        "source_step": 258,
        "max_steps": 411,
        "schedule": "constant",
        "scheduler_state": "preserve",
        "eval_steps": [306, 354, 382, 411],
        "incremental_flops": 178_887_698_756_628_480,
    },
    "b2_hybrid_190m_cooldown.yaml": {
        "id": "B2",
        "source": "hybrid_190m.yaml",
        "source_step": 382,
        "max_steps": 459,
        "schedule": "linear_cooldown",
        "scheduler_state": "restart",
        "eval_steps": [420, 459],
        "incremental_flops": 54_178_083_666_646_016,
    },
    "b3_hybrid_250m_cooldown.yaml": {
        "id": "B3",
        "source": "hybrid_250m.yaml",
        "source_step": 287,
        "max_steps": 345,
        "schedule": "linear_cooldown",
        "scheduler_state": "restart",
        "eval_steps": [325, 345],
        "incremental_flops": 53_473_987_808_216_064,
    },
}


def load_yaml(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return settings


def require_equal(path: Path, name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{path}: {name}={actual!r}, expected {expected!r}")


def validate(
    path: Path,
    expected: dict,
    source_dir: Path,
    data_parallel_size: int,
    scalar_op_cost: int,
) -> dict:
    settings = load_yaml(path)
    source = load_yaml(source_dir / expected["source"])
    continuation = settings.get("continuation") or {}
    scaling = settings["scaling"]
    training = settings["training"]
    scheduler = settings["scheduler"]
    logging = settings["logging"]

    require_equal(path, "continuation.experiment", continuation.get("experiment"), expected["id"])
    require_equal(
        path,
        "continuation.source_config",
        continuation.get("source_config"),
        f'training/configs/scaling/{expected["source"]}',
    )
    require_equal(
        path,
        "continuation.source_checkpoint_step",
        continuation.get("source_checkpoint_step"),
        expected["source_step"],
    )
    require_equal(
        path,
        "continuation.source_run",
        continuation.get("source_run"),
        source["run"]["name"],
    )
    for key in ("restore_model", "restore_optimizer", "restore_rng"):
        require_equal(path, f"continuation.{key}", continuation.get(key), True)
    require_equal(
        path,
        "continuation.scheduler_state",
        continuation.get("scheduler_state"),
        expected["scheduler_state"],
    )
    require_equal(path, "model", settings["model"], source["model"])
    require_equal(path, "optimizer", settings["optimizer"], source["optimizer"])
    require_equal(path, "run.seed", settings["run"]["seed"], source["run"]["seed"])
    require_equal(path, "scaling.stage", scaling.get("stage"), "B")
    require_equal(path, "training.max_steps", training["max_steps"], expected["max_steps"])
    require_equal(path, "scheduler.name", scheduler["name"], expected["schedule"])
    require_equal(path, "scheduler.warmup_steps", scheduler["warmup_steps"], 0)
    require_equal(path, "logging.eval_steps", logging.get("eval_steps"), expected["eval_steps"])
    require_equal(
        path,
        "logging.checkpoint_steps",
        logging.get("checkpoint_steps"),
        expected["eval_steps"],
    )

    tokens_per_update = (
        data_parallel_size
        * training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        * settings["model"]["context_length"]
    )
    require_equal(path, "scaling.tokens_per_update", scaling["tokens_per_update"], tokens_per_update)
    source_tokens = expected["source_step"] * tokens_per_update
    final_tokens = expected["max_steps"] * tokens_per_update
    incremental_steps = expected["max_steps"] - expected["source_step"]
    incremental_tokens = incremental_steps * tokens_per_update
    require_equal(path, "scaling.source_training_tokens", scaling["source_training_tokens"], source_tokens)
    require_equal(
        path,
        "scaling.incremental_training_tokens",
        scaling["incremental_training_tokens"],
        incremental_tokens,
    )
    require_equal(path, "scaling.planned_training_tokens", scaling["planned_training_tokens"], final_tokens)
    require_equal(path, "scaling.target_training_tokens", scaling["target_training_tokens"], final_tokens)
    expected_cooldown = incremental_steps if expected["schedule"] == "linear_cooldown" else 0
    require_equal(path, "scheduler.cooldown_steps", scheduler["cooldown_steps"], expected_cooldown)
    if expected_cooldown:
        require_equal(path, "scheduler.final_lr_ratio", scheduler.get("final_lr_ratio"), 0.0)

    vocab_size = settings["model"]["vocab_size"]
    ledger = complete_hybrid_ledger(
        path, settings, training["micro_batch_size"], vocab_size, scalar_op_cost
    )
    audit = audit_ledger(
        ledger, path, settings, training["micro_batch_size"], vocab_size, ROOT
    )
    parameters = expected_parameter_count(ledger["model_shape"], vocab_size)
    require_equal(
        path,
        "scaling.expected_total_parameters",
        scaling["expected_total_parameters"],
        parameters,
    )
    flops_per_token = Fraction(
        audit["totals"]["total"],
        training["micro_batch_size"] * settings["model"]["context_length"],
    )
    incremental_flops = flops_per_token * incremental_tokens
    if incremental_flops.denominator != 1:
        raise ValueError(f"{path}: incremental FLOPs are not integral")
    require_equal(
        path,
        "budget.incremental_training_flops",
        settings["budget"]["incremental_training_flops"],
        incremental_flops.numerator,
    )
    require_equal(
        path,
        "recorded incremental FLOPs",
        incremental_flops.numerator,
        expected["incremental_flops"],
    )
    return {
        "experiment": expected["id"],
        "parameters": parameters,
        "source_step": expected["source_step"],
        "max_steps": expected["max_steps"],
        "incremental_tokens": incremental_tokens,
        "incremental_flops": incremental_flops.numerator,
        "schedule": expected["schedule"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=ROOT / "training" / "configs" / "scaling_stage_b",
    )
    parser.add_argument(
        "--source-dir",
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
    expected_names = set(EXPERIMENTS)
    if names != expected_names:
        raise ValueError(
            f"Stage B config set differs: missing={sorted(expected_names - names)}, "
            f"unexpected={sorted(names - expected_names)}"
        )
    records = [
        validate(
            path,
            EXPERIMENTS[path.name],
            args.source_dir,
            args.data_parallel_size,
            args.scalar_op_cost,
        )
        for path in paths
    ]
    print("experiment\tparameters\tsource step\tend step\tadded tokens\tadded FLOPs\tschedule")
    for record in records:
        print(
            f'{record["experiment"]}\t{record["parameters"]}\t{record["source_step"]}\t'
            f'{record["max_steps"]}\t{record["incremental_tokens"]}\t'
            f'{record["incremental_flops"]}\t{record["schedule"]}'
        )
    stage_b_flops = sum(record["incremental_flops"] for record in records)
    total = STAGE_A_FLOPS + stage_b_flops
    remaining = 2_000_000_000_000_000_000 - total
    print(f"STAGE_B_TOTAL\t\t\t\t\t{stage_b_flops}\t")
    print(f"STAGE_A_PLUS_B\t\t\t\t\t{total}\t")
    print(f"REMAINING_TO_2E18\t\t\t\t\t{remaining}\t")
    if remaining < 0:
        raise ValueError("Stage A + Stage B exceeds the 2e18 exploratory FLOPs budget")


if __name__ == "__main__":
    main()
