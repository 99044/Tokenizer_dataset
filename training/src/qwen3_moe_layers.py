"""Dense/MoE placement matching the stock Qwen3-MoE decoder."""


from collections.abc import Mapping


def qwen3_moe_layer_pattern(config):
    """Return one 1 (MoE) or 0 (dense FFN) per Transformer layer."""
    get = config.get if isinstance(config, Mapping) else lambda key, default=None: getattr(config, key, default)
    layers = get("num_hidden_layers")
    stride = get("decoder_sparse_step", 1)
    dense_layers = get("mlp_only_layers", [])
    if type(layers) is not int or layers < 1 or type(stride) is not int or stride < 1:
        raise ValueError("Qwen3 MoE layer count and sparse stride must be positive integers")
    if (not isinstance(dense_layers, (list, tuple))
            or any(type(index) is not int or not 0 <= index < layers for index in dense_layers)):
        raise ValueError("mlp_only_layers must contain valid zero-based layer indices")
    dense = set(dense_layers)
    if len(dense) != len(dense_layers):
        raise ValueError("mlp_only_layers must not contain duplicate indices")
    has_experts = get("num_experts", 0) > 0
    return [
        int(has_experts and index not in dense and (index + 1) % stride == 0)
        for index in range(layers)
    ]
