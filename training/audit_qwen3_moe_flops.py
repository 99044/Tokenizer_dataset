#!/usr/bin/env python3
"""Audit a complete logged Qwen3 MoE training path under its declared protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from context_curriculum import curriculum_stage, validate_curriculum
from qwen3_moe_flops import PROTOCOL, REFERENCE_SHA256
from qwen3_moe_rules import validate_family_settings


def audit(settings, metrics_path, *, allow_time_limit=False):
    validate_family_settings(settings)
    training = settings["training"]
    world = settings["submission"]["data_parallel_size"]
    validate_curriculum(settings, world, training["micro_batch_size"],
                        training["gradient_accumulation_steps"], training["max_steps"])
    if settings["budget"].get("flops_protocol") != PROTOCOL:
        raise ValueError("A reference-based Qwen3 MoE FLOPs protocol is required")
    total = rows = 0
    time_limit_seen = False
    with Path(metrics_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            time_limit_seen = row.get("train/stop_reason") == "time_limit"
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step != rows + 1 or step > training["max_steps"]:
                raise ValueError("Metrics must cover steps 1..final once, without gaps or duplicates")
            stage = curriculum_stage(settings, step)
            fields = {
                "train/context_length": stage["length"],
                "train/micro_batch_size": stage["micro_batch_size"],
                "train/grad_accum_steps": stage["grad_accum_steps"],
                "train/curriculum_stage_index": stage["stage_index"],
                "train/tokens": step * stage["tokens_per_update"],
            }
            if any(row.get(key) != value for key, value in fields.items()):
                raise ValueError(f"Step {step}: observed runtime shape/tokens differ from the accounted graph")
            if row.get("train/gdn_fp32_retried", 0) or row.get("train/gdn_fp32_recovery_step", 0):
                raise ValueError("Numerical retries/fallbacks require separate training-path accounting")
            total += stage["flops_per_update"]
            if row.get("train/completed_flops") != total:
                raise ValueError(f"Step {step}: logged cumulative FLOPs disagree with the reference ledger")
            rows += 1
    complete = rows == training["max_steps"]
    if not complete and not (allow_time_limit and time_limit_seen and rows > 0):
        raise ValueError("Training path is incomplete; do not substitute planned FLOPs for actual completed work")
    target = settings["budget"]["total_training_flops"]
    within_one_percent = 99 * target <= 100 * total <= 101 * target
    if complete and not within_one_percent:
        raise ValueError("Logged training path falls outside the target FLOPs +/-1% interval")
    return {
        "status": "passed_log_path_audit" if complete else "passed_time_limited_prefix_audit",
        "organizer_certified": False,
        "tier": settings["submission"]["budget_tier"],
        "architecture_family_id": settings["architecture_family"]["family_id"],
        "flops_protocol": PROTOCOL, "reference_function_sha256": REFERENCE_SHA256,
        "counter_code_sha256": hashlib.sha256((Path(__file__).parent / "src/qwen3_moe_flops.py").read_bytes()).hexdigest(),
        "completed_optimizer_steps": rows, "actual_training_flops": total,
        "training_tokens": rows * settings["scaling"]["tokens_per_update"],
        "final_context_length": stage["length"], "elapsed_seconds": row.get("elapsed_seconds"),
        "target_flops": target, "within_one_percent": within_one_percent,
        "scalar_costs": {"exp": 1, "log": 1, "divide": 1},
        "scalar_costs_status": "Explicit convention; not specified by the supplied rule",
        "limitations": "Audits the complete logged successful-update path; unlogged failed attempts cannot be inferred. Organizer review remains separate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-time-limit", action="store_true",
                        help="Report an audited prefix if the trainer saved and stopped at its deadline")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        report = audit(yaml.safe_load(args.config.read_text()), args.metrics,
                       allow_time_limit=args.allow_time_limit)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    report.update(config=str(args.config.resolve()), metrics=str(args.metrics.resolve()),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  metrics_sha256=hashlib.sha256(args.metrics.read_bytes()).hexdigest())
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
