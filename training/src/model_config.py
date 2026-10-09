"""Shared OLMo-core model construction for the SCICOMP26 model variants."""

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
    head_dim = model.get("head_dim", model.get("attention_head_dim"))
    if model["hidden_size"] != model["num_attention_heads"] * head_dim:
        if model["type"] != "olmo_hybrid_1b":
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

    if model["type"] == "olmo_hybrid_1b":
        return _build_olmo_hybrid_1b(model, settings, vocab_size)

    if model["type"] == "olmo3_1b":
        expected = {
            "num_layers": 16,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 128,
            "intermediate_size": 8192,
            "normalization_style": "reordered_norm",
            "rms_norm_eps": 1.0e-6,
            "qk_norm": True,
            "rope_theta": 500000,
            "sliding_window_pattern": [4096, 4096, 4096, -1],
            "tie_word_embeddings": True,
        }
        for key, value in expected.items():
            if model.get(key) != value:
                raise ValueError(
                    f"OLMo3-1B setting {key}={model.get(key)!r} does not match "
                    f"the official preset value {value!r}"
                )

        factory = getattr(TransformerConfig, "olmo3_1B", None)
        if factory is None:
            raise RuntimeError(
                "the installed OLMo-core does not provide TransformerConfig.olmo3_1B"
            )
        olmo_config = factory(vocab_size=vocab_size, dtype=DType.bfloat16)
        return olmo_config, model["context_length"], settings

    if model["type"] != "dense_decoder_only":
        raise ValueError(f"unsupported model type: {model['type']!r}")

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


def _build_olmo_hybrid_1b(
    model: dict[str, Any], settings: dict[str, Any], vocab_size: int
) -> tuple[TransformerConfig, int, dict[str, Any]]:
    """Build a parameter-matched 1B OLMo Hybrid configuration.

    The sequence mixer pattern is three GDN blocks followed by one global
    attention block. Dense SwiGLU feed-forward layers remain in every block.
    """

    expected = {
        "num_layers": 16,
        "hidden_size": 1792,
        "num_attention_heads": 14,
        "num_key_value_heads": 14,
        "attention_head_dim": 128,
        "intermediate_size": 7168,
        "normalization_style": "reordered_norm",
        "rms_norm_eps": 1.0e-6,
        "qk_norm": True,
        "rope_theta": 500000,
        "attention_window_pattern": [4096, 4096, 4096, -1],
        "block_pattern": ["gdn", "gdn", "gdn", "attn"],
        "gdn_num_heads": 14,
        "gdn_num_value_heads": 14,
        "gdn_head_dim": 96,
        "gdn_value_expand": 2.0,
        "gdn_allow_neg_eigval": True,
        "gdn_conv_size": 4,
        "gdn_norm_eps": 1.0e-5,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(
                f"OLMo Hybrid 1B setting {key}={model.get(key)!r} does not match "
                f"the parameter-matched preset value {value!r}"
            )

    from olmo_core.nn.attention import AttentionBackendName, SlidingWindowAttentionConfig
    from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
    from olmo_core.nn.transformer.config import TransformerBlockConfig

    attention_window = SlidingWindowAttentionConfig(
        pattern=model["attention_window_pattern"],
        force_full_attention_on_first_layer=model["force_full_attention_on_first_layer"],
        force_full_attention_on_last_layer=model["force_full_attention_on_last_layer"],
    )
    attention_config = TransformerConfig.llama_like(
        d_model=model["hidden_size"],
        vocab_size=vocab_size,
        n_layers=model["num_layers"],
        n_heads=model["num_attention_heads"],
        n_kv_heads=model["num_key_value_heads"],
        head_dim=model["attention_head_dim"],
        rope_theta=model["rope_theta"],
        layer_norm_eps=model["rms_norm_eps"],
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=model["qk_norm"],
        dtype=DType.bfloat16,
        attn_backend=AttentionBackendName.flash_2,
        sliding_window=attention_window,
        hidden_size_multiple_of=64,
        hidden_size_multiplier=model["intermediate_size"]
        / int(8 * model["hidden_size"] / 3),
        tie_word_embeddings=model["tie_word_embeddings"],
    )
    if not isinstance(attention_config.block, TransformerBlockConfig):
        raise TypeError("the OLMo Hybrid attention block must be a TransformerBlockConfig")

    gdn_block = attention_config.block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=model["gdn_num_heads"],
            n_v_heads=model["gdn_num_value_heads"],
            head_dim=model["gdn_head_dim"],
            expand_v=model["gdn_value_expand"],
            allow_neg_eigval=model["gdn_allow_neg_eigval"],
            conv_size=model["gdn_conv_size"],
            norm_eps=model["gdn_norm_eps"],
            dtype=DType.bfloat16,
        )
    )
    attention_config.block = {"gdn": gdn_block, "attn": attention_config.block}
    attention_config.block_pattern = model["block_pattern"]

    return attention_config, model["context_length"], settings


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
