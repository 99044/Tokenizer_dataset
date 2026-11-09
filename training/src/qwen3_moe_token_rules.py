"""Declare the 3e17 expert-width sweep on the stock paper-scaling model."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from qwen3_flops import validate_moe_activation_ratio
from qwen3_moe_flops import PROTOCOL
from qwen3_moe_rules import HIDDEN_BY_TIER, architecture_for_hidden, make_family, validate_family_settings


TIER = "3e17"
EXPERT_WIDTHS = (448, 384, 320, 256, 192, 128, 96, 64)
SCHEMA = "qwen3-moe-3e17-expert-width-v1"


def make_token_architecture(width):
    if type(width) is not int or width not in EXPERT_WIDTHS:
        raise ValueError(f"Expert width must be one of {EXPERT_WIDTHS}")
    base = make_family()
    config = architecture_for_hidden(HIDDEN_BY_TIER[TIER], base)
    config["moe_intermediate_size"] = width
    family = {
        "schema": SCHEMA,
        "tier": TIER,
        "base_family": base,
        "expert_width": width,
        "width_rule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    family["family_id"] = hashlib.sha256(json.dumps(family, sort_keys=True).encode()).hexdigest()
    return config, family


def validate_qwen3_moe_settings(settings, *, check_runtime=False):
    family = settings.get("architecture_family")
    if not isinstance(family, dict) or family.get("schema") != SCHEMA:
        return validate_family_settings(settings, check_runtime=check_runtime)

    config, expected = make_token_architecture(family.get("expert_width"))
    if family != expected or settings["model"]["hf_config"] != config:
        raise ValueError("Architecture differs from the declared 3e17 expert-width recipe")
    if settings["submission"]["budget_tier"] != TIER:
        raise ValueError("The expert-width sweep supports only 3e17")
    if settings["budget"].get("flops_protocol") != PROTOCOL:
        raise ValueError("The expert-width sweep requires the competition-reference MoE FLOPs protocol")
    if check_runtime and version("transformers") != expected["base_family"]["transformers_version"]:
        raise ValueError("Runtime Transformers version differs from the declared model code version")
    return validate_moe_activation_ratio(config)
