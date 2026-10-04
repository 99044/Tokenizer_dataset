"""Shared OLMo-core model construction for the SCICOMP26 baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType


def load_settings(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def build_model_config(
    config_path: Path, vocab_size: int
) -> tuple[TransformerConfig, int, dict[str, Any]]:
    settings = load_settings(config_path)
    model = settings["model"]
    training = settings["training"]

    if model["activation"] != "swiglu" or training["dtype"] != "bfloat16":
        raise ValueError("the OLMo-core runner expects SwiGLU and bfloat16")
    if model["hidden_size"] != model["num_attention_heads"] * model["head_dim"]:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim")

    olmo_config = TransformerConfig.llama_like(
        d_model=model["hidden_size"],
        vocab_size=vocab_size,
        n_layers=model["num_layers"],
        n_heads=model["num_attention_heads"],
        n_kv_heads=model["num_attention_heads"],
        rope_theta=model["rope_theta"],
        layer_norm_eps=model["rms_norm_eps"],
        block_name=TransformerBlockType.default,
        qk_norm=False,
        dtype=DType.bfloat16,
        hidden_size_multiple_of=64,
        hidden_size_multiplier=model["intermediate_size"]
        / int(8 * model["hidden_size"] / 3),
    )
    return olmo_config, model["context_length"], settings


def tie_output_weights(model: Any) -> None:
    """Tie the output projection to embeddings across OLMo-core API versions."""

    if hasattr(model, "w_out"):
        output_layer = model.w_out
    elif hasattr(model, "lm_head") and hasattr(model.lm_head, "w_out"):
        output_layer = model.lm_head.w_out
    else:
        raise AttributeError("could not find the OLMo-core output projection")

    if model.embeddings.weight.shape != output_layer.weight.shape:
        raise ValueError("embedding and output projection shapes do not match")
    output_layer.weight = model.embeddings.weight


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
