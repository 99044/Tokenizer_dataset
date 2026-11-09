"""Estimate the 3e17 plan with the existing Qwen3 Dense FLOPs ledger."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "training" / "src"))
from qwen3_flops import parameters, qwen3_ledger

TARGET_FLOPS = 3 * 10**17


def make_plan(config, world_size=4, grad_accumulation=16):
    experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    expert_width = config["moe_intermediate_size"]
    tied_vocab = config["tie_word_embeddings"]
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    sequence = config["max_position_embeddings"]
    ledger_shape = {
        "num_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "attention_head_dim": config["head_dim"],
        "context_length": sequence,
        "vocab_size": config["vocab_size"],
        "intermediate_size": top_k * expert_width,
    }
    # The selected experts contribute the same projection FLOPs as one
    # SwiGLU with their combined width. Router work is accounted separately.
    active_breakdown = parameters(ledger_shape)
    embedding_parameters = active_breakdown["embedding_and_tied_lm_head"]
    if not tied_vocab:
        del active_breakdown["embedding_and_tied_lm_head"]
        active_breakdown["token_embedding"] = embedding_parameters
        active_breakdown["lm_head"] = embedding_parameters
    router_parameters = layers * hidden * experts
    active_breakdown["router"] = router_parameters
    total_breakdown = dict(active_breakdown)
    total_breakdown["swiglu_projections"] = layers * experts * 3 * hidden * expert_width

    ledger = qwen3_ledger(ledger_shape, micro_batch_size=1)
    dense_ledger_cost = sum(group["forward"] + group["backward"] for group in ledger.values())
    if not tied_vocab:
        # The existing Dense ledger includes the tied weights' gradient add.
        dense_ledger_cost -= embedding_parameters
    router_projection_cost = 6 * router_parameters * sequence
    # Explicit planning allowance for softmax, selected-weight normalization,
    # and weighted combination (forward and backward). Dispatch is runtime work.
    router_combine_cost = layers * sequence * (32 * (experts + top_k) + (6 * top_k - 2) * hidden)
    flops_per_microbatch = dense_ledger_cost + router_projection_cost + router_combine_cost
    replicas_per_update = world_size * grad_accumulation
    flops_per_update = flops_per_microbatch * replicas_per_update
    tokens_per_update = sequence * replicas_per_update
    steps = TARGET_FLOPS // flops_per_update
    active = sum(active_breakdown.values())
    vocab_parameters = embedding_parameters * (1 if tied_vocab else 2)
    including_vocab = "including_tied_vocab" if tied_vocab else "including_input_and_output_vocab"
    excluding_vocab = "excluding_tied_vocab" if tied_vocab else "excluding_input_and_output_vocab"
    return {
        "total_parameters": sum(total_breakdown.values()),
        "parameters_by_component": total_breakdown,
        f"active_parameters_{including_vocab}": active,
        f"active_parameters_{excluding_vocab}": active - vocab_parameters,
        "world_size": world_size,
        "micro_batch_size": 1,
        "grad_accumulation_steps": grad_accumulation,
        "sequence_length": sequence,
        "tokens_per_update": tokens_per_update,
        "flops_per_microbatch": flops_per_microbatch,
        "flops_per_token": flops_per_microbatch / sequence,
        "flops_per_update": flops_per_update,
        "target_flops": TARGET_FLOPS,
        "optimizer_updates": steps,
        "planned_training_tokens": steps * tokens_per_update,
        "estimated_training_flops": steps * flops_per_update,
        "attention_core_and_elementwise_flops_per_microbatch": sum(
            ledger[name]["forward"] + ledger[name]["backward"]
            for name in ("layer.attention.core", "layer.attention.elementwise")
        ),
        "accounting": "Logical prediction graph with the existing causal S^2/2 convention",
        "excluded": [
            "expert dispatch and top-k selection",
            "language-model and router auxiliary losses",
            "optimizer, DDP communication and activation recomputation",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=16)
    args = parser.parse_args()
    if min(args.world_size, args.grad_accumulation) < 1:
        parser.error("world size and gradient accumulation must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(make_plan(config, args.world_size, args.grad_accumulation), indent=2))
