"""List or materialize the eight budget-matched 3e17 token-sweep recipes."""

import argparse
import json
from pathlib import Path

import yaml

from plan import make_recipe
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
    parser.add_argument("--width", type=int, choices=EXPERT_WIDTHS)
    parser.add_argument("--config", type=Path, help="Write the selected width's training YAML")
    args = parser.parse_args()
    if args.config is not None and args.width is None:
        parser.error("--config requires --width")

    rows = []
    for width in EXPERT_WIDTHS if args.width is None else (args.width,):
        _, settings, budget = make_recipe(TIER, expert_width=width)
        rows.append(dict(expert_width=width, **{key: budget[key] for key in SUMMARY_FIELDS}))
        if args.config is not None:
            args.config.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    print(json.dumps(rows if args.width is None else rows[0], indent=2))


if __name__ == "__main__":
    main()
