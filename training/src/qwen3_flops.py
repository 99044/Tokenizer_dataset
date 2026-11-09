"""Logical prediction-graph FLOPs using the historical Hybrid conventions."""

from graph_accounting import gemm_operations, operation_flops
from hybrid_flops import Operations


def parameters(model):
    d, f, h = (model[k] for k in ("hidden_size", "intermediate_size", "attention_head_dim"))
    q, kv = (model[k] * h for k in ("num_attention_heads", "num_key_value_heads"))
    vocab = model["vocab_size"]
    return {
        "embedding_and_tied_lm_head": vocab * d,
        "attention_projections": model["num_layers"] * d * (2 * q + 2 * kv),
        "swiglu_projections": model["num_layers"] * 3 * d * f,
        "rmsnorm": model["num_layers"] * (2 * d + 2 * h) + d,
    }


def qwen3_ledger(model, micro_batch_size, *, attention_window=None):
    d, f, h = (model[k] for k in ("hidden_size", "intermediate_size", "attention_head_dim"))
    q_heads, kv_heads = (model[k] for k in ("num_attention_heads", "num_key_value_heads"))
    vocab, seq = model["vocab_size"], model["context_length"]
    tokens = micro_batch_size * seq
    q, kv = q_heads * h, kv_heads * h
    components = {}

    def projections(name, shapes):
        components[name] = [
            op for width_in, width_out in shapes
            for op in gemm_operations(tokens, width_out, width_in,
                                      input_gradient=True, weight_gradient=True)
        ]

    projections("layer.attention.projections", [(d, q), (d, kv), (d, kv), (q, d)])
    projections("layer.swiglu.projections", [(d, f), (d, f), (f, d)])
    projections("output.lm_head", [(d, vocab)])

    # Global layers keep the historical S²/2 convention. A window includes
    # the query token and has a shorter prefix before the band is filled.
    pairs = micro_batch_size * q_heads * seq * seq // 2
    if attention_window is not None:
        window = min(attention_window, seq)
        pairs_per_head = seq * window - window * (window - 1) // 2
        pairs = micro_batch_size * q_heads * pairs_per_head
    attention = Operations(1)
    for phase, calls in (("forward", 2), ("backward", 4)):
        if attention_window is None:
            attention.mm("causal_qk_av", seq, h, seq, phase,
                         calls=calls * micro_batch_size * q_heads, causal_reference_half=True)
        else:
            attention.mm("windowed_qk_av", pairs_per_head, h, 1, phase,
                         calls=calls * micro_batch_size * q_heads)
    components["layer.attention.core"] = attention.items

    elem = Operations(1)
    elem.rms("qk_norm", tokens * (q_heads + kv_heads), h, torch_autograd=True)
    width = tokens * (q + kv)
    for phase in ("forward", "backward"):
        elem.op("rope.multiply", "multiply", 2 * width, phase)
        elem.op("rope.add", "add", width, phase)
        elem.op("rope.negate", "multiply", width // 2, phase)
    rows = tokens * q_heads
    elem.op("attention.scale", "multiply", pairs)
    elem.op("softmax.subtract_max", "subtract", pairs)
    elem.scalar("softmax.exp", pairs)
    elem.op("softmax.sum", "add", pairs - rows)
    elem.scalar("softmax.divide", pairs)
    elem.op("softmax.gradient_products", "multiply", pairs, "backward", 2)
    elem.op("softmax.gradient_sum", "add", pairs - rows, "backward")
    elem.op("softmax.gradient_subtract", "subtract", pairs, "backward")
    elem.op("attention.gradient_scale", "multiply", pairs, "backward")
    elem.op("gqa.kv_gradient_reduction", "add", 2 * tokens * (q - kv), "backward")
    elem.op("qkv.input_gradient_add", "add", 2 * tokens * d, "backward")
    components["layer.attention.elementwise"] = elem.items

    ffn = Operations(1)
    ffn.silu("swiglu.silu", tokens * f)
    ffn.op("swiglu.product", "multiply", tokens * f)
    ffn.op("swiglu.product_gradients", "multiply", tokens * f, "backward", 2)
    ffn.op("swiglu.input_gradient_add", "add", tokens * d, "backward")
    components["layer.swiglu.elementwise"] = ffn.items

    norm = Operations(1)
    norm.rms("pre_norm", tokens, d, copies=2, torch_autograd=True)
    norm.op("residual.add", "add", tokens * d, calls=2)
    norm.op("residual.gradient_add", "add", tokens * d, "backward", calls=2)
    components["layer.norm_residual"] = norm.items
    out = Operations(1)
    out.rms("final_norm", tokens, d, torch_autograd=True)
    out.op("embedding.gradient_scatter_add", "add", tokens * d, "backward")
    out.op("tied_embedding_output_gradient_add", "add", vocab * d, "backward")
    components["output.norm_and_embedding_gradients"] = out.items

    result = {}
    for name, operations in components.items():
        copies = model["num_layers"] if name.startswith("layer.") else 1
        result[name] = {
            "layers": copies,
            "operations_per_layer": operations,
            **{phase: copies * sum(operation_flops(op) for op in operations if op["phase"] == phase)
               for phase in ("forward", "backward")},
        }
    return result


def qwen3_moe_windows(config):
    """Return the local width and the zero-based global layer indices."""
    window = config.get("local_attention_window")
    global_layers = config.get("global_attention_layers")
    if window is None and global_layers is None:
        return None, ()
    layers = config["num_hidden_layers"]
    if (
        type(window) is not int or window < 1 or config.get("use_sliding_window", False)
        or not isinstance(global_layers, (list, tuple)) or not 0 < len(global_layers) < layers
        or any(type(index) is not int or not 0 <= index < layers for index in global_layers)
        or len(set(global_layers)) != len(global_layers)
    ):
        raise ValueError("Expected a positive local window and unique in-range global MoE layers")
    return window, tuple(global_layers)


def qwen3_moe_cost(config, sequence):
    """Prediction-graph microbatch FLOPs for one sequence of a routed Qwen3."""
    layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    shape = dict(
        num_layers=layers, hidden_size=hidden, num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"], attention_head_dim=config["head_dim"],
        context_length=sequence, vocab_size=config["vocab_size"],
        intermediate_size=top_k * config["moe_intermediate_size"],
    )
    ledger = qwen3_ledger(shape, micro_batch_size=1)
    dense = sum(group["forward"] + group["backward"] for group in ledger.values())
    if not config["tie_word_embeddings"]:
        dense -= config["vocab_size"] * hidden
    router_projection = 6 * layers * hidden * experts * sequence
    router_combine = layers * sequence * (32 * (experts + top_k) + (6 * top_k - 2) * hidden)
    attention = sum(
        ledger[name]["forward"] + ledger[name]["backward"]
        for name in ("layer.attention.core", "layer.attention.elementwise")
    )
    window, global_layers = qwen3_moe_windows(config)
    if window is not None:
        local_layers = layers - len(global_layers)
        local_ledger = qwen3_ledger(
            {**shape, "num_layers": local_layers}, micro_batch_size=1, attention_window=window,
        )
        local_attention = sum(
            local_ledger[name]["forward"] + local_ledger[name]["backward"]
            for name in ("layer.attention.core", "layer.attention.elementwise")
        )
        delta = local_attention - attention * local_layers // layers
        dense += delta
        attention += delta
    return dense + router_projection + router_combine, attention
