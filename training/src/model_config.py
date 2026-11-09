"""Shared OLMo-core model construction for the SCICOMP26 model variants."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from olmo_core.nn.transformer import TransformerConfig
    from qwen3_moe_training import Qwen3MoeTrainingConfig

from dual_residual import validate_dual_residual_spec
from value_residual import validate_value_residual_spec


def load_settings(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def build_model_config(
    config_path: Path, vocab_size: int
) -> tuple[TransformerConfig | Qwen3MoeTrainingConfig, int, dict[str, Any]]:
    settings = load_settings(config_path)
    model = settings["model"]
    training = settings["training"]
    if model["type"] == "qwen3_moe":
        from qwen3_moe_training import Qwen3MoeTrainingConfig
        from qwen3_flops import validate_moe_activation_ratio

        hf = model["hf_config"]
        validate_moe_activation_ratio(hf)
        if "architecture_family" in settings:
            from qwen3_moe_active_ladder import validate_qwen3_moe_settings

            validate_qwen3_moe_settings(settings, check_runtime=True)
        if hf["vocab_size"] != vocab_size or hf["max_position_embeddings"] != model["context_length"]:
            raise ValueError("Qwen3 MoE vocabulary or context disagrees with the training configuration")
        if training["dtype"] != "bfloat16" or training.get("gradient_checkpointing", False):
            raise ValueError("This Qwen3 MoE ledger requires BF16 without activation recomputation")
        return Qwen3MoeTrainingConfig(hf), model["context_length"], settings
    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerBlockType

    dual_residual = validate_dual_residual_spec(
        model.get("dual_residual"), num_layers=model["num_layers"]
    )
    value_residual = validate_value_residual_spec(model.get("value_residual"))
    if (dual_residual is not None or value_residual is not None) and model["type"] not in {
        "olmo_hybrid_1b", "olmo_hybrid_scaling"
    }:
        raise ValueError("residual extensions currently support only OLMo hybrid models")
    configured_vocab = model.get("vocab_size")
    if configured_vocab is not None and configured_vocab != vocab_size:
        raise ValueError(
            f"configured vocab_size={configured_vocab} does not match data vocab_size={vocab_size}"
        )

    if model["activation"] != "swiglu" or training["dtype"] != "bfloat16":
        raise ValueError("the OLMo-core runner expects SwiGLU and bfloat16")
    head_dim = model.get("head_dim", model.get("attention_head_dim"))
    hybrid_types = {"olmo_hybrid_1b", "olmo_hybrid_scaling"}
    if model["hidden_size"] != model["num_attention_heads"] * head_dim:
        if model["type"] not in hybrid_types:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")

    if model["type"] in hybrid_types:
        return _build_olmo_hybrid(
            model,
            settings,
            vocab_size,
            strict_1b=model["type"] == "olmo_hybrid_1b",
        )

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


def _build_olmo_hybrid(
    model: dict[str, Any],
    settings: dict[str, Any],
    vocab_size: int,
    *,
    strict_1b: bool,
) -> tuple[TransformerConfig, int, dict[str, Any]]:
    """Build the fixed 1B preset or a member of its scaling family.

    The sequence mixer pattern is three GDN blocks followed by one global
    attention block. Dense SwiGLU feed-forward layers remain in every block.
    """

    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerBlockType

    fixed_architecture = {
        "activation": "swiglu",
        "normalization": "rmsnorm",
        "normalization_style": "reordered_norm",
        "rms_norm_eps": 1.0e-6,
        "qk_norm": True,
        "position_encoding": "rope",
        "rope_theta": 500000,
        "attention_window_pattern": [4096, 4096, 4096, -1],
        "force_full_attention_on_first_layer": False,
        "force_full_attention_on_last_layer": True,
        "block_pattern": ["gdn", "gdn", "gdn", "attn"],
        "gdn_head_dim": 96,
        "gdn_value_expand": 2.0,
        "gdn_allow_neg_eigval": True,
        "gdn_conv_size": 4,
        "gdn_norm_eps": 1.0e-5,
        "context_length": 8192,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "dropout": 0.0,
    }
    for key, value in fixed_architecture.items():
        if model.get(key) != value:
            raise ValueError(
                f"OLMo Hybrid setting {key}={model.get(key)!r} does not match "
                f"the scaling-family value {value!r}"
            )

    if strict_1b:
        expected_dimensions = {
            "num_layers": 16,
            "hidden_size": 1792,
            "num_attention_heads": 14,
            "num_key_value_heads": 14,
            "attention_head_dim": 128,
            "intermediate_size": 7168,
            "gdn_num_heads": 14,
            "gdn_num_value_heads": 14,
        }
        for key, value in expected_dimensions.items():
            if model.get(key) != value:
                raise ValueError(
                    f"OLMo Hybrid 1B setting {key}={model.get(key)!r} does not match "
                    f"the parameter-matched preset value {value!r}"
                )
    else:
        integer_fields = (
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "attention_head_dim",
            "intermediate_size",
            "gdn_num_heads",
            "gdn_num_value_heads",
        )
        for key in integer_fields:
            value = model.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"OLMo Hybrid scaling setting {key} must be a positive integer")
        if model["num_layers"] % len(model["block_pattern"]):
            raise ValueError("num_layers must contain complete gdn/gdn/gdn/attn groups")
        if model["hidden_size"] != model["num_attention_heads"] * model["attention_head_dim"]:
            raise ValueError("hidden_size must equal num_attention_heads * attention_head_dim")
        if model["num_key_value_heads"] != model["num_attention_heads"]:
            raise ValueError("the reviewed scaling family uses equal attention and KV head counts")
        if not model.get("compressed_attention") and model["intermediate_size"] != 4 * model["hidden_size"]:
            raise ValueError("the reviewed scaling family uses intermediate_size = 4 * hidden_size")
        if model["gdn_num_heads"] != model["num_attention_heads"]:
            raise ValueError("the reviewed scaling family keeps GDN and attention head counts equal")
        if model["gdn_num_value_heads"] != model["gdn_num_heads"]:
            raise ValueError("the reviewed scaling family uses equal GDN key and value head counts")

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
