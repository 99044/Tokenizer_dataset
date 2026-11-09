"""List or materialize the 3e17 active-parameter ladder."""

import argparse
import json
from pathlib import Path

import yaml

from plan import make_recipe
from qwen3_moe_active_ladder import ACTIVE_VARIANTS
from qwen3_moe_token_ladder import LADDER_WIDTHS
from qwen3_moe_token_rules import EXPERT_WIDTHS, TIER


SUMMARY_FIELDS = (
    "total_parameters",
    "active_parameters_including_input_and_output_vocab",
    "total_non_vocabulary_parameters",
    "active_non_vocabulary_parameters",
    "active_non_vocabulary_fraction",
    "tokens_per_update",
    "optimizer_updates",
    "planned_training_tokens",
    "estimated_training_flops",
    "architecture_family_id",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--variant", choices=ACTIVE_VARIANTS)
    selection.add_argument("--width", type=int, choices=LADDER_WIDTHS + EXPERT_WIDTHS)
    parser.add_argument("--config", type=Path, help="Write the selected recipe's training YAML")
    args = parser.parse_args()
    if args.width is not None:
        choices = [(f"w{args.width}", {"expert_width": args.width})]
    else:
        variants = ACTIVE_VARIANTS if args.variant is None else (args.variant,)
        choices = [(variant, {"active_variant": variant}) for variant in variants]
    if args.config is not None and len(choices) != 1:
        parser.error("--config requires --variant or --width")

    rows = []
    for variant, options in choices:
        config, settings, budget = make_recipe(TIER, **options)
        rows.append(dict(variant=variant, expert_width=config["moe_intermediate_size"], hidden_size=config["hidden_size"],
                         **{key: budget[key] for key in SUMMARY_FIELDS}))
        if args.config is not None:
            args.config.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    print(json.dumps(rows[0] if len(rows) == 1 else rows, indent=2))


if __name__ == "__main__":
    main()
