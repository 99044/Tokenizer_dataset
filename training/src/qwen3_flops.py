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
