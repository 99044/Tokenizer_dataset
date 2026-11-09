"""One predeclared scaling family and the supplied non-vocabulary 10% rule."""

import hashlib
import json
import math
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

from qwen3_flops import qwen3_moe_parameters, validate_moe_activation_ratio
from qwen3_moe_flops import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
HIDDEN_BY_TIER = {"3e17": 256, "1e18": 384, "3e18": 512, "1e19": 768}


def maximum_expert_count(config):
    """Largest integer E with (A + E*R + K*P)/(A + E*(R+P)) >= 1/10."""
    layers, hidden, top_k = (config[k] for k in ("num_hidden_layers", "hidden_size", "num_experts_per_tok"))
    per_expert = 3 * layers * hidden * config["moe_intermediate_size"]
    per_router_row = layers * hidden
    dense = qwen3_moe_parameters(dict(config, num_experts=top_k))["total_non_vocabulary_parameters"]
    always_active = dense - top_k * (per_expert + per_router_row)
    denominator = per_expert - 9 * per_router_row
    if denominator <= 0:
        raise ValueError("Expert width is too small for this bounded-capacity rule")
    return (9 * always_active + 10 * top_k * per_expert) // denominator


def architecture_for_hidden(hidden, family):
    """Only size fields vary; layer types, connections and routing stay fixed."""
    if hidden not in HIDDEN_BY_TIER.values():
        raise ValueError("Hidden dimension is outside the predeclared scale ladder")
    config = json.loads((ROOT / "qwen3_moe_a20m/config.json").read_text())
    # At d=256/384/512/768, default K is 2/2/4/8. The algorithm stays fixed.
    top_k = max(2, 2 ** (hidden // 256 + family["granularity_shift"]))
    config.update(
        hidden_size=hidden, num_hidden_layers=hidden // 32,
        num_attention_heads=hidden // 64, num_key_value_heads=2, head_dim=64,
        intermediate_size=4 * hidden, moe_intermediate_size=4 * hidden // top_k,
        num_experts=top_k, num_experts_per_tok=top_k,
        router_aux_loss_coef=0.001, router_z_loss_coef=0.001,
    )
    maximum = maximum_expert_count(config)
    count = int(maximum * Fraction(str(family["expert_capacity_fraction"])))
    alignment = family["expert_alignment"]
    config["num_experts"] = count // alignment * alignment
    if config["num_experts"] <= top_k:
        raise ValueError("Capacity fraction/alignment leaves too few experts")
    return config


def make_family(granularity_shift=0, expert_capacity_fraction=1.0, expert_alignment=1):
    if isinstance(granularity_shift, bool) or not isinstance(granularity_shift, int) or granularity_shift not in (-1, 0, 1):
        raise ValueError("Granularity shift must be -1, 0 or 1")
    if (isinstance(expert_capacity_fraction, bool) or not isinstance(expert_capacity_fraction, (int, float))
            or not math.isfinite(expert_capacity_fraction) or not 0 < expert_capacity_fraction <= 1):
        raise ValueError("Expert capacity fraction must lie in (0, 1]")
    if isinstance(expert_alignment, bool) or expert_alignment not in (1, 4, 8, 16):
        raise ValueError("Expert alignment must be 1, 4, 8 or 16")
    family = {
        "schema": "qwen3-moe-scaling-v2",
        "hidden_by_tier": dict(HIDDEN_BY_TIER),
        "granularity_shift": granularity_shift,
        "expert_capacity_fraction": float(expert_capacity_fraction),
        "expert_alignment": expert_alignment,
        "generation_rules": {
            "layers": "hidden_size // 32",
            "attention_heads": "hidden_size // 64",
            "kv_heads": "2",
            "head_dim": "64",
            "active_ffn_width": "4 * hidden_size",
            "top_k": "max(2, 2**(hidden_size // 256 + granularity_shift))",
            "expert_width": "4 * hidden_size // top_k",
            "expert_count": "floor(E_max * expert_capacity_fraction / expert_alignment) * expert_alignment",
            "E_max": "floor((9*A + 10*K*P)/(P - 9*R)); A=always-active non-router/non-vocab params, P=one expert pool, R=one router row across layers",
        },
        "fixed_mechanisms": {
            "layer_type": "stock Qwen3Moe decoder in every layer",
            "connections": "pre-norm residual attention followed by residual routed FFN",
            "activation": "SwiGLU",
            "position_encoding": "RoPE, theta=1000000",
            "routing": "dropless token-choice softmax/top-k with renormalized selected weights; K follows the declared size rule",
            "shared_experts": 0,
        },
        "transformers_version": "5.18.0",
        "model_code_sha256": hashlib.sha256((ROOT / "training/src/qwen3_moe_training.py").read_bytes()).hexdigest(),
        "acceleration_code_sha256": hashlib.sha256((ROOT / "training/src/h800_acceleration.py").read_bytes()).hexdigest(),
        "base_config_sha256": hashlib.sha256((ROOT / "qwen3_moe_a20m/config.json").read_bytes()).hexdigest(),
        "generation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for hidden in HIDDEN_BY_TIER.values():
        validate_moe_activation_ratio(architecture_for_hidden(hidden, family))
    family["family_id"] = hashlib.sha256(json.dumps(family, sort_keys=True).encode()).hexdigest()
    return family


def validate_family_settings(settings, *, check_runtime=False):
    family = settings.get("architecture_family")
    if not isinstance(family, dict):
        raise ValueError("Missing predeclared architecture_family")
    expected = make_family(family.get("granularity_shift"), family.get("expert_capacity_fraction"), family.get("expert_alignment"))
    if family != expected:
        raise ValueError("Architecture family, generation rules or model source version changed")
    if settings.get("budget", {}).get("flops_protocol") != PROTOCOL:
        raise ValueError("This family requires the competition-reference MoE FLOPs protocol")
    tier = settings["submission"]["budget_tier"]
    if tier not in HIDDEN_BY_TIER:
        raise ValueError("Unknown architecture scale")
    config = settings["model"]["hf_config"]
    if config != architecture_for_hidden(HIDDEN_BY_TIER[tier], expected):
        raise ValueError("HF architecture differs from its predeclared scaling rule")
    if check_runtime and version("transformers") != family["transformers_version"]:
        raise ValueError("Runtime Transformers version differs from the declared model code version")
    return validate_moe_activation_ratio(config)
