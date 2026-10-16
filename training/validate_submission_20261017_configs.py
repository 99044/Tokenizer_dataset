#!/usr/bin/env python3
"""Validate the four 2026-10-17 SOAP submission configs."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from src.graph_accounting import audit_ledger  # noqa: E402
from src.hybrid_flops import complete_hybrid_ledger, expected_parameter_count  # noqa: E402


EXPECTED = {
    "FLOPs_3e17.yaml": {
        "tier": "3e17", "size": "110m", "budget": 300_000_000_000_000_000,
        "parameters": 107_135_898, "layers": 12, "hidden": 640, "heads": 5,
        "intermediate": 2560,
    },
    "FLOPs_1e18.yaml": {
        "tier": "1e18", "size": "190m", "budget": 1_000_000_000_000_000_000,
        "parameters": 190_571_664, "layers": 16, "hidden": 768, "heads": 6,
        "intermediate": 3072,
    },
    "FLOPs_3e18.yaml": {
        "tier": "3e18", "size": "250m", "budget": 3_000_000_000_000_000_000,
        "parameters": 254_467_368, "layers": 16, "hidden": 896, "heads": 7,
        "intermediate": 3584,
    },
    "FLOPs_1e19.yaml": {
        "tier": "1e19", "size": "330m", "budget": 10_000_000_000_000_000_000,
        "parameters": 327_544_256, "layers": 16, "hidden": 1024, "heads": 8,
        "intermediate": 4096,
    },
}


def require(path: Path, name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{path}: {name}={actual!r}, expected {expected!r}")


def validate(path: Path, expected: dict) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = settings["model"]
    training = settings["training"]
    scaling = settings["scaling"]
    budget = settings["budget"]
    optimizer = settings["optimizer"]
    scheduler = settings["scheduler"]
    logging = settings["logging"]

    require(path, "submission.target_date", settings["submission"]["target_date"], "2026-10-17")
    require(path, "submission.budget_tier", settings["submission"]["budget_tier"], expected["tier"])
    require(path, "submission.data_parallel_size", settings["submission"]["data_parallel_size"], 4)
    require(path, "scaling.nominal_size", scaling["nominal_size"], expected["size"])
    require(path, "scaling.expected_total_parameters", scaling["expected_total_parameters"], expected["parameters"])
    require(path, "scaling.tokens_per_update", scaling["tokens_per_update"], 524_288)
    require(path, "budget.total_training_flops", budget["total_training_flops"], expected["budget"])

    require(path, "model.type", model["type"], "olmo_hybrid_scaling")
    require(path, "model.num_layers", model["num_layers"], expected["layers"])
    require(path, "model.hidden_size", model["hidden_size"], expected["hidden"])
    require(path, "model.num_attention_heads", model["num_attention_heads"], expected["heads"])
    require(path, "model.num_key_value_heads", model["num_key_value_heads"], expected["heads"])
    require(path, "model.intermediate_size", model["intermediate_size"], expected["intermediate"])
    require(path, "model.attention_head_dim", model["attention_head_dim"], 128)
    require(path, "model.block_pattern", model["block_pattern"], ["gdn", "gdn", "gdn", "attn"])
    require(path, "model.context_length", model["context_length"], 8192)
    require(path, "model.vocab_size", model["vocab_size"], 32768)
    require(path, "model.tie_word_embeddings", model["tie_word_embeddings"], True)

    require(path, "training.dtype", training["dtype"], "bfloat16")
    require(path, "training.micro_batch_size", training["micro_batch_size"], 1)
    require(path, "training.gradient_accumulation_steps", training["gradient_accumulation_steps"], 16)
    tokens_per_update = 4 * training["micro_batch_size"] * training["gradient_accumulation_steps"] * model["context_length"]
    require(path, "calculated tokens_per_update", tokens_per_update, 524_288)

    require(path, "optimizer.name", optimizer["name"], "soap")
    require(path, "optimizer.learning_rate", optimizer["learning_rate"], 0.003)
    require(path, "optimizer.betas", optimizer["betas"], [0.95, 0.95])
    require(path, "optimizer.eps", optimizer["eps"], 1.0e-8)
    require(path, "optimizer.weight_decay", optimizer["weight_decay"], 0.1)
    require(path, "optimizer.precondition_frequency", optimizer["precondition_frequency"], 10)
    require(path, "optimizer.max_precond_dim", optimizer["max_precond_dim"], 1024)

    require(path, "scheduler.name", scheduler["name"], "linear_cooldown")
    require(path, "scheduler.warmup_steps", scheduler["warmup_steps"], 20)
    require(path, "scheduler.final_lr_ratio", scheduler["final_lr_ratio"], 0.0)
    max_steps = training["max_steps"]
    require(path, "scheduler.cooldown_steps", scheduler["cooldown_steps"], round(max_steps * 58 / 345))

    checkpoints = sorted({round(max_steps * index / 10) for index in range(1, 10)})
    require(path, "logging.eval_steps", logging["eval_steps"], checkpoints)
    require(path, "logging.checkpoint_steps", logging["checkpoint_steps"], checkpoints)
    require(path, "logging.eval_every_steps", logging["eval_every_steps"], max_steps)
    require(path, "logging.checkpoint_every_steps", logging["checkpoint_every_steps"], max_steps)
    require(path, "logging.final_eval_batches", logging["final_eval_batches"], 64)

    ledger = complete_hybrid_ledger(path, settings, training["micro_batch_size"], model["vocab_size"], 1)
    audit = audit_ledger(ledger, path, settings, training["micro_batch_size"], model["vocab_size"], ROOT)
    parameters = expected_parameter_count(ledger["model_shape"], model["vocab_size"])
    require(path, "calculated parameters", parameters, expected["parameters"])
    flops_per_token = Fraction(audit["totals"]["total"], training["micro_batch_size"] * model["context_length"])
    require(path, "flops_per_token_numerator", budget["flops_per_token_numerator"], flops_per_token.numerator)
    require(path, "flops_per_token_denominator", budget["flops_per_token_denominator"], flops_per_token.denominator)

    flops_per_step = flops_per_token * tokens_per_update
    planned_steps = expected["budget"] // flops_per_step
    if planned_steps.denominator != 1:
        raise ValueError(f"{path}: max step calculation is not integral")
    require(path, "training.max_steps", max_steps, planned_steps.numerator)
    planned_tokens = max_steps * tokens_per_update
    require(path, "budget.planned_training_tokens", budget["planned_training_tokens"], planned_tokens)
    planned_flops = flops_per_step * max_steps
    if planned_flops.denominator != 1:
        raise ValueError(f"{path}: planned FLOPs are not integral")
    require(path, "budget.planned_training_flops", budget["planned_training_flops"], planned_flops.numerator)
    if planned_flops > expected["budget"] or planned_flops + flops_per_step <= expected["budget"]:
        raise ValueError(f"{path}: plan is not the largest whole update within the budget")

    return {
        "config": path.name,
        "size": expected["size"],
        "parameters": parameters,
        "steps": max_steps,
        "tokens": planned_tokens,
        "planned_flops": planned_flops.numerator,
        "cooldown": scheduler["cooldown_steps"],
        "checkpoints": len(checkpoints) + 1,
    }


def main() -> None:
    config_dir = ROOT / "training" / "configs"
    records = [validate(config_dir / name, values) for name, values in EXPECTED.items()]
    print("config\tsize\tparameters\tsteps\ttokens\tcooldown\tcheckpoints\tplanned FLOPs")
    for record in records:
        print(
            f"{record['config']}\t{record['size']}\t{record['parameters']}\t"
            f"{record['steps']}\t{record['tokens']}\t{record['cooldown']}\t"
            f"{record['checkpoints']}\t{record['planned_flops']}"
        )


if __name__ == "__main__":
    main()
