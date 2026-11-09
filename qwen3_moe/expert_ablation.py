"""Generate and compare matched 3e17 expert-layout experiments."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
from statistics import mean

import yaml

from plan import make_recipe


BASELINE = "e24-k2-w512"
VARIANTS = {
    BASELINE: ((24, 2, 512), {}),
    "e48-k4-w256": ((48, 4, 256), {"granularity_shift": 1, "expert_alignment": 4}),
    "e20-k2-w512": ((20, 2, 512), {"expert_capacity_fraction": 5 / 6}),
}
LAYOUT_FIELDS = ("num_experts", "num_experts_per_tok", "moe_intermediate_size")
ROUTER_METRIC = "train/router_layer_max_to_mean"


def make_expert_recipe(variant):
    layout, options = VARIANTS[variant]
    config, settings, budget = make_recipe("3e17", **options)
    if tuple(config[key] for key in LAYOUT_FIELDS) != layout:
        raise ValueError(f"Scaling family no longer generates {variant}")
    return config, settings, budget


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_run(directory):
    run_plan = load_json(directory / "train/run-plan.json")
    settings = run_plan["settings"]
    config = settings["model"]["hf_config"]
    experts, top_k, width = (config[key] for key in LAYOUT_FIELDS)
    variant = f"e{experts}-k{top_k}-w{width}"
    _, expected, budget = make_expert_recipe(variant)
    if settings != expected:
        raise ValueError(f"{directory}: training settings differ from the {variant} recipe")
    audit = load_json(directory / "flops-report.json")
    if (
        audit.get("status") != "passed_log_path_audit"
        or audit["actual_training_flops"] != budget["estimated_training_flops"]
        or audit["training_tokens"] != budget["planned_training_tokens"]
        or audit["completed_optimizer_steps"] != budget["optimizer_updates"]
    ):
        raise ValueError(f"{directory}: completed training differs from its FLOPs plan")
    evaluation = load_json(directory / "eval-result.json")
    source = evaluation["data_source"]
    if (evaluation["batch_size_per_gpu"], evaluation["sequence_length"],
            evaluation["distributed_world_size"], source["kind"]) != (64, 8192, 1, "packed"):
        raise ValueError(f"{directory}: expected packed single-GPU evaluation at batch 64 and context 8192")
    if evaluation["active_parameters"] != budget["active_non_vocabulary_parameters"]:
        raise ValueError(f"{directory}: evaluated active parameter count differs from the recipe")
    with (directory / "train/metrics.jsonl").open(encoding="utf-8") as stream:
        tail = deque((json.loads(line) for line in stream if line.strip()), maxlen=200)
    last_step = budget["optimizer_updates"]
    if [row["step"] for row in tail] != list(range(last_step - 199, last_step + 1)):
        raise ValueError(f"{directory}: missing the final 200 consecutive training updates")
    ratios = [row[ROUTER_METRIC] for row in tail]
    if any(len(row) != config["num_hidden_layers"] or any(
        not math.isfinite(value) or not 1 <= value <= experts for value in row
    ) for row in ratios):
        raise ValueError(f"{directory}: invalid per-layer router load ratios")
    score = {key: evaluation[key] for key in ("bpb", "forward_seconds", "evaltime")}
    if any(not math.isfinite(value) or value <= 0 for value in score.values()):
        raise ValueError(f"{directory}: invalid evaluation metrics")
    acceleration = evaluation["acceleration"]
    comparison_signature = (
        run_plan["data"],
        source["path"], source["total_bytes"], evaluation["scored_tokens"],
        source["identity"]["cache_tokenizer_sha256"], evaluation["fusion"]["mode"],
        acceleration["experts"], acceleration["attention"], acceleration["compile_swiglu"],
        acceleration["torch_version"], acceleration["cuda_version"],
    )
    return dict(
        variant=variant, run_dir=str(directory), **score, **{
            key: budget[key] for key in (
                "planned_training_tokens", "optimizer_updates", "estimated_training_flops",
                "active_non_vocabulary_fraction",
            )
        },
        evaluation_acceleration=acceleration, router_updates=[last_step - 199, last_step],
        router_layer_max_to_mean=[
            dict(layer=index, mean=mean(values), peak=max(values))
            for index, values in enumerate(zip(*ratios), 1)
        ],
    ), comparison_signature


def compare(baseline_directory, candidate_directories):
    baseline, signature = read_run(baseline_directory)
    if baseline["variant"] != BASELINE:
        raise ValueError(f"Reference must use {BASELINE}")
    comparisons = []
    seen = {BASELINE}
    for directory in candidate_directories:
        candidate, candidate_signature = read_run(directory)
        if candidate_signature != signature:
            raise ValueError(f"{directory}: training cache metadata or evaluation settings differ from the reference")
        if candidate["variant"] in seen:
            raise ValueError(f"Duplicate variant: {candidate['variant']}")
        seen.add(candidate["variant"])
        comparisons.append(dict(
            **candidate,
            delta_bpb=candidate["bpb"] - baseline["bpb"],
            delta_bpb_pct=100 * (candidate["bpb"] / baseline["bpb"] - 1),
            forward_time_ratio=candidate["forward_seconds"] / baseline["forward_seconds"],
        ))
    return dict(baseline=baseline, candidates=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan = subcommands.add_parser("plan")
    plan.add_argument("variant", choices=VARIANTS)
    plan.add_argument("output", type=Path)
    comparison = subcommands.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "plan":
        _, settings, budget = make_expert_recipe(args.variant)
        args.output.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        result = {key: budget[key] for key in (
            "optimizer_updates", "planned_training_tokens", "estimated_training_flops",
            "active_non_vocabulary_fraction",
        )}
        result["variant"] = args.variant
    else:
        result = compare(args.baseline, args.candidates)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
