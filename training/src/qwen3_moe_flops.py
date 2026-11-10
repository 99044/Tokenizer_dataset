"""Competition Megatron reference plus a source-derived Qwen3 routing ledger.

GEMMs use 2mnk, with separate forward/input-gradient/weight-gradient entries.
Ordinary Transformer elementwise/loss terms follow the reference exclusions.
Routing exp/log/divide use an explicit one-operation convention: the supplied
rules do not publish their weights. This is not organizer certification.
"""

import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from graph_accounting import gemm_operations, operation_flops
from qwen3_moe_layers import qwen3_moe_layer_pattern

PROTOCOL = "scicomp26-megatron-qwen3-moe-router-zloss-v2"
REFERENCE = Path(__file__).with_name("flops_reference")
REFERENCE_SHA256 = "4d6e6daf89926f83f4e5009f337540cf1f16850e1b18d9b1c6cb6fb74b93fc97"
SCALAR_COST = 1


@lru_cache(maxsize=1)
def load_reference():
    path = REFERENCE / "megatron_training.py"
    source = path.read_text()
    manifest = json.loads((REFERENCE / "megatron_manifest.json").read_text())
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "num_floating_point_operations")
    function = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
    if (hashlib.sha256(function.encode()).hexdigest() != REFERENCE_SHA256
            or hashlib.sha256(source.encode()).hexdigest() != manifest["file_sha256"]):
        raise ValueError("Megatron reference source differs from the reviewed snapshot")
    # This adapter supports the standard Transformer branch only, not hybrid/LA.
    namespace = {"is_hybrid_model": lambda _: False, "is_linear_attention_variant": lambda _: False,
                 "is_gated_delta_net_variant": lambda _: False}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["num_floating_point_operations"], manifest


def reference_flops(config, sequence, micro_batch_size=1):
    function, _ = load_reference()
    args = SimpleNamespace(
        decoder_seq_length=None, seq_length=sequence, hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"], num_attention_heads=config["num_attention_heads"],
        num_query_groups=config["num_key_value_heads"], kv_channels=config["head_dim"],
        group_query_attention=True, multi_latent_attention=False, attention_output_gate=False,
        num_experts=config["num_experts"], moe_router_topk=config["num_experts_per_tok"],
        moe_layer_freq=qwen3_moe_layer_pattern(config), moe_ffn_hidden_size=config["moe_intermediate_size"],
        ffn_hidden_size=config["intermediate_size"], moe_shared_expert_intermediate_size=None,
        moe_latent_size=None, mtp_num_layers=None, experimental_attention_variant=None,
        swiglu=True, padded_vocab_size=config["vocab_size"],
    )
    return function(args, micro_batch_size)


def routing_ledger(config, tokens, *, include_input_scatter=True, include_loss_weight=True):
    """Logical sparse routing DAG, including the runner's accumulated auxiliary loss.

    The normalization derivative follows PyTorch 2.6 FunctionsManual.cpp:
    grad/self divisor and -grad*((self/other)/other), including broadcast sums.
    Dense zero-buffer materialization, indexing, comparisons and casts are not
    extra logical floating-point arithmetic. Active-path scatter additions are.
    """
    layers = sum(qwen3_moe_layer_pattern(config))
    if not layers:
        return []
    hidden, experts, top_k = (config[k] for k in (
        "hidden_size", "num_experts", "num_experts_per_tok"))
    if not config["norm_topk_prob"]:
        raise ValueError("This routing ledger requires normalized top-k probabilities")
    operations = []

    def add(name, kind, elements, phase="forward", copies=layers):
        if elements <= 0:
            return
        if kind in {"exp", "log", "divide"}:
            operations.append(dict(name=name, kind="audited_scalar", operation=kind,
                                   elements=elements, calls=copies, phase=phase,
                                   flops_per_element=SCALAR_COST,
                                   counting_basis="Declared exp/log/divide cost 1; not specified by supplied rule"))
        else:
            operations.append(dict(name=name, kind=kind, elements=elements, calls=copies, phase=phase))

    for operation in gemm_operations(tokens, experts, hidden, input_gradient=True, weight_gradient=True):
        operations.append(dict(operation, name="router.projection", calls=layers))
    for prefix in ("router.softmax", "auxiliary.softmax"):
        add(prefix + ".subtract_max", "subtract", tokens * experts)
        add(prefix + ".exp", "exp", tokens * experts)
        add(prefix + ".sum", "add", tokens * (experts - 1))
        add(prefix + ".normalize", "divide", tokens * experts)
        add(prefix + ".gradient_product", "multiply", 2 * tokens * experts, "backward")
        add(prefix + ".gradient_sum", "add", tokens * (experts - 1), "backward")
        add(prefix + ".gradient_subtract", "subtract", tokens * experts, "backward")
    add("topk_normalize.sum", "add", tokens * (top_k - 1))
    add("topk_normalize.divide", "divide", tokens * top_k)
    add("topk_normalize.gradient_divisions", "divide", 3 * tokens * top_k, "backward")
    add("topk_normalize.gradient_negate", "multiply", tokens * top_k, "backward")
    add("topk_normalize.gradient_product", "multiply", tokens * top_k, "backward")
    add("topk_normalize.broadcast_gradient_sum", "add", tokens * (top_k - 1), "backward")
    add("topk_normalize.gradient_add", "add", tokens * top_k, "backward")
    add("combine.weight", "multiply", tokens * top_k * hidden)
    add("combine.scatter_add", "add", tokens * top_k * hidden)
    add("combine.gradient_products", "multiply", 2 * tokens * top_k * hidden, "backward")
    add("combine.weight_gradient_sum", "add", tokens * top_k * (hidden - 1), "backward")
    if include_input_scatter:
        add("dispatch.input_gradient_scatter", "add", tokens * top_k * hidden, "backward")
        add("router.input_gradient_merge", "add", tokens * hidden, "backward")
    add("auxiliary.probability_mean_sum", "add", experts * (tokens - 1))
    add("auxiliary.probability_mean_divide", "divide", experts)
    add("auxiliary.frequency_divide", "divide", experts)
    add("auxiliary.frequency_probability_product", "multiply", experts)
    add("auxiliary.expert_sum", "add", experts - 1)
    add("auxiliary.layer_sum", "add", layers - 1, copies=1)
    add("auxiliary.layer_mean", "divide", 1, copies=1)
    add("auxiliary.expert_scale", "multiply", 1, copies=1)
    add("auxiliary.expert_scale_gradient", "multiply", 1, "backward", copies=1)
    add("auxiliary.layer_mean_gradient", "divide", 1, "backward")
    add("auxiliary.probability_gradient_product", "multiply", experts, "backward")
    add("auxiliary.probability_mean_gradient", "divide", tokens * experts, "backward")
    add("auxiliary.router_gradient_merge", "add", tokens * experts, "backward")
    if include_loss_weight:
        add("auxiliary.loss_coefficient", "multiply", 1, copies=1)
        add("auxiliary.loss_add", "add", 1, copies=1)
        add("auxiliary.loss_coefficient_gradient", "multiply", 1, "backward", copies=1)
    if config.get("router_z_loss_coef", 0):
        # logsumexp: max comparisons are not FLOPs. The stable arithmetic is
        # subtract, exp, reduce-sum, log, and restore max; backward is
        # exp(logits - logsumexp) * upstream gradient. square' uses 2*x*grad.
        add("z_loss.logsumexp.subtract_max", "subtract", tokens * experts)
        add("z_loss.logsumexp.exp", "exp", tokens * experts)
        add("z_loss.logsumexp.sum", "add", tokens * (experts - 1))
        add("z_loss.logsumexp.log", "log", tokens)
        add("z_loss.logsumexp.restore_max", "add", tokens)
        add("z_loss.square", "multiply", tokens)
        add("z_loss.token_mean_sum", "add", tokens - 1)
        add("z_loss.token_mean_divide", "divide", 1)
        add("z_loss.layer_mean_sum", "add", layers - 1, copies=1)
        add("z_loss.layer_mean_divide", "divide", 1, copies=1)
        add("z_loss.layer_mean_gradient", "divide", 1, "backward")
        add("z_loss.token_mean_gradient", "divide", tokens, "backward")
        add("z_loss.square_gradient", "multiply", 2 * tokens, "backward")
        add("z_loss.logsumexp.gradient_subtract", "subtract", tokens * experts, "backward")
        add("z_loss.logsumexp.gradient_exp", "exp", tokens * experts, "backward")
        add("z_loss.logsumexp.gradient_product", "multiply", tokens * experts, "backward")
        add("z_loss.router_gradient_merge", "add", tokens * experts, "backward")
        if include_loss_weight:
            add("z_loss.loss_coefficient", "multiply", 1, copies=1)
            add("z_loss.loss_add", "add", 1, copies=1)
            add("z_loss.loss_coefficient_gradient", "multiply", 1, "backward", copies=1)
    if config.get("hi_moe"):
        # The grouped Top-K comparisons are integer selection; the added FP32
        # probability and regularizer arithmetic is charged explicitly.
        add("himoe.bias_scale", "multiply", tokens * experts)
        add("himoe.bias_subtract", "subtract", tokens * experts)
        add("himoe.temperature_divide", "divide", tokens * experts)
        add("himoe.ema_token_sum", "add", (tokens - 1) * experts)
        add("himoe.ema_token_mean", "divide", experts)
        add("himoe.ema_update", "multiply", 2 * experts)
        add("himoe.ema_accumulate", "add", experts)
        add("himoe.regularizer_softmax_subtract", "subtract", tokens * experts)
        add("himoe.regularizer_softmax_exp", "exp", tokens * experts)
        add("himoe.regularizer_softmax_sum", "add", tokens * (experts - 1))
        add("himoe.regularizer_softmax_divide", "divide", tokens * experts)
        add("himoe.inter_square", "multiply", tokens * top_k)
        add("himoe.inter_sum", "add", tokens * (top_k - 1))
        add("himoe.intra_square", "multiply", tokens * experts)
        add("himoe.intra_sum", "add", tokens * (experts - 1))
        add("himoe.regularizer_softmax_gradient", "multiply", 2 * tokens * experts, "backward")
        add("himoe.regularizer_softmax_gradient_sum", "add", tokens * (experts - 1), "backward")
        add("himoe.regularizer_softmax_gradient_subtract", "subtract", tokens * experts, "backward")
        add("himoe.inter_square_gradient", "multiply", 2 * tokens * top_k, "backward")
        add("himoe.intra_square_gradient", "multiply", 2 * tokens * experts, "backward")
        add("himoe.router_gradient_merge", "add", 2 * tokens * experts, "backward")
    return operations


def cost_breakdown(config, sequence, micro_batch_size=1):
    if (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 2
            or isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size < 1):
        raise ValueError("Sequence length >=2 and a positive integer micro-batch are required")
    tokens = micro_batch_size * sequence
    pattern = qwen3_moe_layer_pattern(config)
    layers, routed_layers, hidden = len(pattern), sum(pattern), config["hidden_size"]
    query = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    base = {
        "attention_projections": 6 * tokens * layers * hidden * (2 * query + 2 * kv),
        "attention_core": 6 * micro_batch_size * sequence**2 * layers * query,
        "active_expert_projections": 18 * tokens * routed_layers * hidden * config["num_experts_per_tok"] * config["moe_intermediate_size"],
        "lm_head": 6 * tokens * hidden * config["vocab_size"],
    }
    if routed_layers < layers:
        base["dense_ffn_projections"] = 18 * tokens * (layers - routed_layers) * hidden * config["intermediate_size"]
    reference = reference_flops(config, sequence, micro_batch_size)
    if sum(base.values()) != reference:
        raise ValueError("Matrix decomposition disagrees with the pinned Megatron reference")
    operations = routing_ledger(config, tokens)
    phases = {phase: sum(operation_flops(op) for op in operations if op["phase"] == phase)
              for phase in ("forward", "backward")}
    projection = 6 * tokens * routed_layers * hidden * config["num_experts"]
    _, manifest = load_reference()
    return {
        "protocol": PROTOCOL, "estimated_total": reference + sum(phases.values()),
        "attention_core_and_elementwise": base["attention_core"],
        "reference_matched": True, "router_graph_audited": True, "official_ledger": False,
        "reference_function_sha256": REFERENCE_SHA256,
        "counter_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_source": manifest["upstream_url"],
        "megatron_reference_total": reference, "megatron_components": base,
        "router_projection_gemm": projection,
        "router_forward": phases["forward"], "router_backward": phases["backward"],
        "router_operations": operations,
        "components": dict(base, router_supplement=sum(phases.values())),
        "scalar_costs": {"exp": SCALAR_COST, "log": SCALAR_COST, "divide": SCALAR_COST},
        "scalar_costs_status": "Explicit convention; the supplied rules do not assign exp/log/divide weights",
        "non_flop_operations": ["top-k comparisons", "integer routing counts", "indexing", "copies and casts"],
        "reference_exclusions": ["ordinary Transformer normalization/activation/RoPE/attention-softmax/LM-loss elementwise terms"],
        "excluded_by_supplied_rules": ["optimizer updates", "activation recomputation", "communication"],
        "sources": [
            "https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
            "https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/autograd/FunctionsManual.cpp",
            "training/src/qwen3_moe_training.py:AccumulatedRouterBalance",
            "training/src/qwen3_moe_training.py:router_z_loss",
        ],
    }
