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


def qwen3_moe_parameters(config):
    """Exact counts for homogeneous, fixed-top-k MoE in every decoder layer."""
    if config.get("decoder_sparse_step", 1) != 1 or config.get("mlp_only_layers", []):
        raise ValueError("Parameter accounting requires an MoE in every layer")
    layers, hidden = config["num_hidden_layers"], config["hidden_size"]
    experts, top_k = config["num_experts"], config["num_experts_per_tok"]
    width = config["moe_intermediate_size"]
    if not 1 <= top_k <= experts:
        raise ValueError("Invalid fixed-top-k expert configuration")
    active = parameters(dict(
        num_layers=layers, hidden_size=hidden,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"], attention_head_dim=config["head_dim"],
        vocab_size=config["vocab_size"], intermediate_size=top_k * width,
    ))
    embedding = active["embedding_and_tied_lm_head"]
    tied = config["tie_word_embeddings"]
    if not tied:
        del active["embedding_and_tied_lm_head"]
        active.update(token_embedding=embedding, lm_head=embedding)
    active["router"] = layers * hidden * experts
    total = dict(active, swiglu_projections=layers * experts * 3 * hidden * width)
    vocabulary = embedding * (1 if tied else 2)
    including = "including_tied_vocab" if tied else "including_input_and_output_vocab"
    excluding = "excluding_tied_vocab" if tied else "excluding_input_and_output_vocab"
    return {
        "total_parameters": sum(total.values()),
        "parameters_by_component": total,
        f"active_parameters_{including}": sum(active.values()),
        f"active_parameters_{excluding}": sum(active.values()) - vocabulary,
        "total_non_vocabulary_parameters": sum(total.values()) - vocabulary,
        "active_non_vocabulary_parameters": sum(active.values()) - vocabulary,
    }


def validate_moe_activation_ratio(config):
    """Apply the supplied >=10% rule, excluding both vocabulary matrices."""
    report = qwen3_moe_parameters(config)
    numerator = report["active_non_vocabulary_parameters"]
    denominator = report["total_non_vocabulary_parameters"]
    # Exact integer comparison avoids rounding a sub-threshold model up to 10%.
    if denominator <= 0:
        raise ValueError("Non-vocabulary parameter count must be positive")
    if 10 * numerator < denominator:
        raise ValueError(
            f"Non-vocabulary MoE activation ratio {numerator}/{denominator} "
            f"({numerator / denominator:.4%}) is below the required 10%"
        )
    return dict(report, active_non_vocabulary_fraction=numerator / denominator)


def qwen3_ledger(model, micro_batch_size):
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

    attention = Operations(1)
    for phase, calls in (("forward", 2), ("backward", 4)):
        attention.mm("causal_qk_av", seq, h, seq, phase,
                     calls=calls * micro_batch_size * q_heads, causal_reference_half=True)
    components["layer.attention.core"] = attention.items

    elem = Operations(1)
    elem.rms("qk_norm", tokens * (q_heads + kv_heads), h, torch_autograd=True)
    width = tokens * (q + kv)
    for phase in ("forward", "backward"):
        elem.op("rope.multiply", "multiply", 2 * width, phase)
        elem.op("rope.add", "add", width, phase)
        elem.op("rope.negate", "multiply", width // 2, phase)
    rows = tokens * q_heads
    pairs = micro_batch_size * q_heads * seq * seq // 2
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


def qwen3_moe_cost_breakdown(config, sequence, *, protocol=None, micro_batch_size=1):
    """Expose the legacy estimate's exact GEMMs and still-pending proxy terms."""
    if protocol is not None:
        from qwen3_moe_flops import PROTOCOL, cost_breakdown

        if protocol != PROTOCOL:
            raise ValueError(f"Unknown Qwen3 MoE FLOPs protocol: {protocol}")
        return cost_breakdown(config, sequence, micro_batch_size)
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
    gemm_names = ("layer.attention.projections", "layer.attention.core",
                  "layer.swiglu.projections", "output.lm_head")
    components = {name: group["forward"] + group["backward"] for name, group in ledger.items()}
    if not config["tie_word_embeddings"]:
        components["untied_embedding_gradient_adjustment"] = -config["vocab_size"] * hidden
    components["router_projection_gemm"] = router_projection
    components["router_softmax_and_combine_proxy"] = router_combine
    return {
        "estimated_total": dense + router_projection + router_combine,
        "attention_core_and_elementwise": attention,
        "standard_gemm_components": {name: components[name] for name in gemm_names},
        "router_projection_gemm": router_projection,
        "components": components,
        "official_ledger": False,
        "pending_review": [
            "Router softmax/combine uses the inherited 32*(experts+top_k) proxy, not a fully audited operator graph.",
            "Elementwise exp/divide/sqrt use scalar cost 1; the supplied rule does not specify these costs.",
            "Top-k/dispatch, LM loss and router auxiliary-loss graphs are not included in this estimate.",
        ],
        "excluded_by_supplied_rules": ["optimizer updates", "activation recomputation", "communication"],
    }


def qwen3_moe_cost(config, sequence, *, protocol=None, micro_batch_size=1):
    """Prediction-graph microbatch estimate; see breakdown for review status."""
    report = qwen3_moe_cost_breakdown(config, sequence, protocol=protocol, micro_batch_size=micro_batch_size)
    factor = micro_batch_size if protocol is None else 1
    return factor * report["estimated_total"], factor * report["attention_core_and_elementwise"]
