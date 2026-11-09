"""Generate four budget-matched MoE starting recipes for eight H800 GPUs.

Expert widths follow https://arxiv.org/html/2605.11689 section 3.1.
These are unmeasured candidates, not claims of optimal validation BPB.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from qwen3_moe_a30m.plan import make_plan, make_training_settings
from qwen3_flops import qwen3_moe_cost_breakdown
from qwen3_moe_flops import PROTOCOL
from qwen3_moe_rules import (
    HIDDEN_BY_TIER, architecture_for_hidden, make_family,
)
from qwen3_moe_token_rules import TIER, make_token_architecture, validate_qwen3_moe_settings

# Default paper-scaling training presets.
TIERS = {
    "3e17": dict(flops=3 * 10**17, accumulation=4, learning_rate=0.003),
    "1e18": dict(flops=10**18, accumulation=8, learning_rate=0.002),
    "3e18": dict(flops=3 * 10**18, accumulation=16, learning_rate=0.0015),
    "1e19": dict(flops=10**19, accumulation=16, learning_rate=0.001),
}


def make_recipe(tier, *, granularity_shift=0, expert_capacity_fraction=1.0, expert_alignment=1,
                learning_rate=None, tokens_per_update=None, expert_width=None):
    preset = TIERS[tier]
    # Keep each rank's physical micro-batch at 8192 tokens throughout the
    # curriculum. Larger global batches use more accumulation, not more memory.
    tokens_per_microbatch = 8 * 8192
    if tokens_per_update is None:
        accumulation = preset["accumulation"]
    else:
        if (isinstance(tokens_per_update, bool) or not isinstance(tokens_per_update, int)
                or tokens_per_update < tokens_per_microbatch or tokens_per_update % tokens_per_microbatch):
            raise ValueError("tokens_per_update must be a positive integer multiple of 65536 (8 GPUs x 8192 tokens)")
        accumulation = tokens_per_update // tokens_per_microbatch
    lr = preset["learning_rate"] if learning_rate is None else learning_rate
    if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")

    if expert_width is None:
        family = make_family(granularity_shift, expert_capacity_fraction, expert_alignment)
        config = architecture_for_hidden(HIDDEN_BY_TIER[tier], family)
    else:
        if tier != TIER or (granularity_shift, expert_capacity_fraction, expert_alignment) != (0, 1, 1):
            raise ValueError("The 3e17 expert-width sweep uses the fixed paper-scaling router")
        config, family = make_token_architecture(expert_width)
    top_k, experts = config["num_experts_per_tok"], config["num_experts"]
    run_name = f"qwen3_moe_{tier}_e{experts}_k{top_k}"
    if expert_width is not None:
        run_name += f"_w{expert_width}"
    baseline = make_plan(config, world_size=8, grad_accumulation=accumulation,
                         target_flops=preset["flops"], flops_protocol=PROTOCOL)
    if baseline["optimizer_updates"] < 2:
        raise ValueError("tokens_per_update is too large for the FLOPs budget")
    settings = make_training_settings(
        config, baseline, budget_tier=tier,
        run_name=f"{run_name}_soap",
        learning_rate=lr, warmup_fraction=0.02, cooldown_fraction=0.20,
        enforce_legacy_limits=False,
    )
    settings["architecture_family"] = family
    settings["training"]["h800_acceleration"] = dict(
        experts="auto", attention="auto", compile_swiglu=True,
    )
    activation = validate_qwen3_moe_settings(settings)
    curriculum = settings["training"]["length_curriculum"]
    if not 99 * preset["flops"] <= 100 * curriculum["planned_training_flops"] <= 100 * preset["flops"]:
        raise ValueError("tokens_per_update leaves too much unused compute; require 99%-100% of the FLOPs budget")
    # Evaluate each stage boundary and each tenth of the final 8K stage.
    # Short-context CE is useful for stability; compare candidates at final 8K.
    full = curriculum["stages"][-1]
    milestones = sorted({
        *(stage["last_step"] for stage in curriculum["stages"][:-1]),
        *(full["first_step"] - 1 + max(1, round(full["steps"] * part / 10))
          for part in range(1, 10)),
    })
    settings["logging"].update(checkpoint_steps=milestones, eval_steps=milestones)
    total = baseline["total_parameters"]
    active = baseline["active_parameters_including_input_and_output_vocab"]
    budget = {
        "status": "unmeasured_candidate",
        "paper": "https://arxiv.org/html/2605.11689",
        "tier": tier,
        "hardware_target": "8 x H800, assuming 80 GiB per GPU",
        "total_parameters": total,
        "active_parameters_including_input_and_output_vocab": active,
        "total_non_vocabulary_parameters": activation["total_non_vocabulary_parameters"],
        "active_non_vocabulary_parameters": activation["active_non_vocabulary_parameters"],
        "active_non_vocabulary_fraction": activation["active_non_vocabulary_fraction"],
        "minimum_active_non_vocabulary_fraction": 0.1,
        "architecture_family_id": family["family_id"],
        "expert_granularity": f"1/{top_k}",
        "expert_activation_sparsity": experts / top_k,
        "tokens_per_update": baseline["tokens_per_update"],
        "optimizer_updates": curriculum["max_steps"],
        "planned_training_tokens": curriculum["planned_training_tokens"],
        "target_flops": preset["flops"],
        "estimated_training_flops": curriculum["planned_training_flops"],
        "flops_protocol": PROTOCOL,
        "megatron_reference_matched": True,
        "router_graph_audited": True,
        "organizer_certified": False,
        "flops_utilization": curriculum["planned_training_flops"] / preset["flops"],
        "training_tokens_per_active_parameter": curriculum["planned_training_tokens"] / active,
        "persistent_state_lower_bound_gib_per_gpu": 20 * total / 2**30,
        "memory_limitations": (
            "20 bytes/parameter counts BF16 weights, shared BF16 gradient/DDP buckets, "
            "FP32 masters/gradients and FP32 SOAP moments. Excludes SOAP preconditioners, "
            "optimizer temporaries, activations, logits, CUDA/NCCL and allocator overhead. "
            "DDP replicates all experts on each GPU. Verify peak memory on H800 before a full run."
        ),
        "reference_8k": baseline,
        "flops_audit": {
            str(length): qwen3_moe_cost_breakdown(config, length, protocol=PROTOCOL, micro_batch_size=8192 // length)
            for length in (2048, 4096, 8192)
        },
        "stages": copy.deepcopy(curriculum["stages"]),
    }
    return config, settings, budget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS)
    parser.add_argument("--granularity-shift", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--expert-capacity-fraction", type=float, default=1.0)
    parser.add_argument("--expert-alignment", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--tokens-per-update", type=int,
                        help="Global batch in tokens; multiple of 65536. 1048576 matches the paper's batch.")
    parser.add_argument("--output-dir", type=Path, help="Write TIER/{config.json,training.yaml,budget.json}")
    args = parser.parse_args(argv)
    if args.tier and (args.granularity_shift != 0 or args.expert_capacity_fraction != 1.0 or args.expert_alignment != 1):
        parser.error("Architecture overrides generate a whole four-tier family; omit --tier")
    if args.learning_rate is not None and not args.tier:
        parser.error("A learning-rate override requires --tier")
    rows = []
    for tier in ([args.tier] if args.tier else TIERS):
        try:
            config, settings, budget = make_recipe(
                tier, granularity_shift=args.granularity_shift,
                expert_capacity_fraction=args.expert_capacity_fraction, expert_alignment=args.expert_alignment,
                learning_rate=args.learning_rate,
                tokens_per_update=args.tokens_per_update,
            )
        except ValueError as error:
            parser.error(str(error))
        if args.output_dir is not None:
            import yaml

            destination = args.output_dir / tier
            destination.mkdir(parents=True, exist_ok=True)
            for filename, value in (("config.json", config), ("budget.json", budget)):
                (destination / filename).write_text(json.dumps(value, indent=2) + "\n")
            (destination / "training.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        rows.append({key: budget[key] for key in (
            "tier", "status", "total_parameters", "active_parameters_including_input_and_output_vocab",
            "total_non_vocabulary_parameters", "active_non_vocabulary_parameters",
            "active_non_vocabulary_fraction", "architecture_family_id",
            "expert_granularity", "expert_activation_sparsity", "optimizer_updates",
            "tokens_per_update",
            "planned_training_tokens", "estimated_training_flops",
            "persistent_state_lower_bound_gib_per_gpu",
        )})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
