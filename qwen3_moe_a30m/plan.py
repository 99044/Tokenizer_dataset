"""Estimate the 3e17 plan with the existing Qwen3 Dense FLOPs ledger."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "training" / "src"))
from qwen3_flops import parameters, qwen3_moe_cost

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

    flops_per_microbatch, attention_cost = qwen3_moe_cost(config, sequence)
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
        "attention_core_and_elementwise_flops_per_microbatch": attention_cost,
        "accounting": "Logical prediction graph with the existing causal S^2/2 convention",
        "excluded": [
            "expert dispatch and top-k selection",
            "language-model and router auxiliary losses",
            "optimizer, DDP communication and activation recomputation",
        ],
    }


def make_training_settings(config, baseline, target_tokens=None):
    from context_curriculum import apply_length_curriculum

    seq = config["max_position_embeddings"]
    per_token = Fraction(baseline["flops_per_microbatch"], seq)
    full_steps = baseline["optimizer_updates"]
    settings = {
        "run": {"name": "qwen3_moe_a30m_3e17_soap", "seed": 42},
        "model": {"type": "qwen3_moe", "context_length": seq, "hf_config": config},
        "training": {
            "dtype": "bfloat16", "max_steps": full_steps,
            "micro_batch_size": baseline["micro_batch_size"],
            "gradient_accumulation_steps": baseline["grad_accumulation_steps"],
            "gradient_checkpointing": False,
        },
        "optimizer": {
            "name": "soap", "learning_rate": 0.003, "betas": [0.95, 0.95],
            "eps": 1e-8, "weight_decay": 0.1,
            "precondition_frequency": 10, "max_precond_dim": 1024,
        },
        "scheduler": {
            "name": "linear_cooldown", "warmup_steps": 20,
            "cooldown_steps": round(full_steps * 432 / 2570), "final_lr_ratio": 0.0,
        },
        "data": {
            "format": "npy_packed_tokens",
            "tokenizer_json_sha256": "6f1812b3fbf8758220be646974072ef08c66df2d2d5ef9d398f0e158b9ee947c",
        },
        "scaling": {
            "expected_total_parameters": baseline["total_parameters"],
            "tokens_per_update": baseline["tokens_per_update"],
        },
        "budget": {
            "total_training_flops": TARGET_FLOPS,
            "planned_training_flops": baseline["estimated_training_flops"],
            "planned_training_tokens": baseline["planned_training_tokens"],
            "flops_per_token_numerator": per_token.numerator,
            "flops_per_token_denominator": per_token.denominator,
            "flops_per_update": baseline["flops_per_update"],
        },
        "submission": {"budget_tier": "3e17", "data_parallel_size": baseline["world_size"]},
        "logging": {"eval_batches": 8, "final_eval_batches": 64, "checkpoint_keep_last": 2},
    }
    if target_tokens is not None:
        active_key = (
            "active_parameters_including_tied_vocab" if config["tie_word_embeddings"]
            else "active_parameters_including_input_and_output_vocab"
        )
        active_millions = round(baseline[active_key] / 1_000_000)
        settings["run"]["name"] = f"qwen3_moe_a{active_millions}m_3e17_soap"
        settings["budget"]["target_training_tokens"] = target_tokens
    apply_length_curriculum(settings)
    steps = settings["training"]["max_steps"]
    tokens = settings["budget"]["planned_training_tokens"]
    if target_tokens is None:
        if steps <= 1500 or not 800_000_000 <= tokens <= 1_000_000_000:
            raise ValueError("Qwen3 MoE plan must exceed 1500 steps and process 0.8–1B tokens")
    elif not target_tokens <= tokens < target_tokens + baseline["tokens_per_update"]:
        raise ValueError("Qwen3 MoE token budget differs from the requested whole-update target")
    milestones = list(range(800, steps, 800))
    settings["logging"].update(checkpoint_steps=milestones, eval_steps=milestones)
    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=16)
    parser.add_argument("--write-training-config", type=Path)
    parser.add_argument("--target-tokens", type=int)
    args = parser.parse_args()
    if min(args.world_size, args.grad_accumulation) < 1:
        parser.error("world size and gradient accumulation must be positive")
    if args.target_tokens is not None and (args.target_tokens < 1 or not args.write_training_config):
        parser.error("--target-tokens requires a positive value and --write-training-config")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline = make_plan(config, args.world_size, args.grad_accumulation)
    if args.write_training_config:
        import yaml

        settings = make_training_settings(config, baseline, args.target_tokens)
        args.write_training_config.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        print(json.dumps(settings["training"]["length_curriculum"], indent=2))
    else:
        print(json.dumps(baseline, indent=2))
