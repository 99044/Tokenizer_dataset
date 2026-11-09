"""Declare the 3e17 size ladder and its 12M endpoint."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from qwen3_flops import validate_moe_activation_ratio
from qwen3_moe_flops import PROTOCOL
from qwen3_moe_rules import HIDDEN_BY_TIER, architecture_for_hidden, make_family
from qwen3_moe_token_rules import (
    TIER, make_token_architecture, validate_qwen3_moe_settings as validate_previous_settings,
)


LADDER_WIDTHS = (440, 376, 312, 248, 184, 120, 56, 48)
COMPACT_WIDTH = LADDER_WIDTHS[-1]
SCHEMA = "qwen3-moe-3e17-token-ladder-v1"


def make_ladder_architecture(width):
    if type(width) is not int or width not in LADDER_WIDTHS:
        return make_token_architecture(width)

    base = make_family()
    config = architecture_for_hidden(HIDDEN_BY_TIER[TIER], base)
    config["moe_intermediate_size"] = width
    if width == COMPACT_WIDTH:
        config.update(hidden_size=144, intermediate_size=576)
    family = {
        "schema": SCHEMA,
        "tier": TIER,
        "base_family": base,
        "expert_width": width,
        "ladder_rule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    family["family_id"] = hashlib.sha256(json.dumps(family, sort_keys=True).encode()).hexdigest()
    return config, family


def validate_qwen3_moe_settings(settings, *, check_runtime=False):
    family = settings.get("architecture_family")
    if not isinstance(family, dict) or family.get("schema") != SCHEMA:
        return validate_previous_settings(settings, check_runtime=check_runtime)

    config, expected = make_ladder_architecture(family.get("expert_width"))
    if family != expected or settings["model"]["hf_config"] != config:
        raise ValueError("Architecture differs from the declared 3e17 token ladder")
    budget = settings["budget"]
    if settings["submission"]["budget_tier"] != TIER or budget["total_training_flops"] != 3 * 10**17:
        raise ValueError("The token ladder requires a 3e17 FLOPs budget")
    if budget.get("flops_protocol") != PROTOCOL:
        raise ValueError("The token ladder requires the competition-reference MoE FLOPs protocol")
    if check_runtime and version("transformers") != expected["base_family"]["transformers_version"]:
        raise ValueError("Runtime Transformers version differs from the declared model code version")
    return validate_moe_activation_ratio(config)
