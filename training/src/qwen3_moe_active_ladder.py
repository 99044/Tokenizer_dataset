"""Declare the 3e17 ladder by active parameters including the vocabulary."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from qwen3_flops import validate_moe_activation_ratio
from qwen3_moe_flops import PROTOCOL
from qwen3_moe_rules import HIDDEN_BY_TIER, architecture_for_hidden, make_family
from qwen3_moe_token_ladder import validate_qwen3_moe_settings as validate_previous_settings
from qwen3_moe_token_rules import TIER


# (Hidden size, expert FFN width); the attention layout and router stay fixed.
ACTIVE_VARIANTS = {
    "a22m": (296, 392),
    "a20p6m": (280, 376),
    "a19p1m": (264, 352),
    "a17p7m": (248, 328),
    "a16p3m": (232, 304),
    "a14p9m": (216, 280),
    "a13p5m": (192, 304),
    "a12m": (176, 264),
}
SCHEMA = "qwen3-moe-3e17-active-ladder-v1"


def make_active_architecture(variant):
    if not isinstance(variant, str) or variant not in ACTIVE_VARIANTS:
        raise ValueError(f"Active-parameter variant must be one of {tuple(ACTIVE_VARIANTS)}")
    hidden, width = ACTIVE_VARIANTS[variant]
    base = make_family()
    config = architecture_for_hidden(HIDDEN_BY_TIER[TIER], base)
    config.update(hidden_size=hidden, intermediate_size=4 * hidden, moe_intermediate_size=width)
    family = {
        "schema": SCHEMA,
        "tier": TIER,
        "base_family": base,
        "variant": variant,
        "active_ladder_rule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    family["family_id"] = hashlib.sha256(json.dumps(family, sort_keys=True).encode()).hexdigest()
    return config, family


def validate_qwen3_moe_settings(settings, *, check_runtime=False):
    family = settings.get("architecture_family")
    if not isinstance(family, dict) or family.get("schema") != SCHEMA:
        return validate_previous_settings(settings, check_runtime=check_runtime)

    config, expected = make_active_architecture(family.get("variant"))
    if family != expected or settings["model"]["hf_config"] != config:
        raise ValueError("Architecture differs from the declared 3e17 active-parameter ladder")
    budget = settings["budget"]
    if settings["submission"]["budget_tier"] != TIER or budget["total_training_flops"] != 3 * 10**17:
        raise ValueError("The active-parameter ladder requires a 3e17 FLOPs budget")
    if budget.get("flops_protocol") != PROTOCOL:
        raise ValueError("The active-parameter ladder requires the competition-reference MoE FLOPs protocol")
    if check_runtime and version("transformers") != expected["base_family"]["transformers_version"]:
        raise ValueError("Runtime Transformers version differs from the declared model code version")
    return validate_moe_activation_ratio(config)
