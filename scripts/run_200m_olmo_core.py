#!/usr/bin/env python3
"""Build and run one training step for the SCICOMP26 200M baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType


def load_model_config(config_path: Path, vocab_size: int) -> tuple[TransformerConfig, int]:
    settings = yaml.safe_load(config_path.read_text())
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
        hidden_size_multiplier=model["intermediate_size"] / int(8 * model["hidden_size"] / 3),
    )
    return olmo_config, model["context_length"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/200m_dense.yaml"),
        help="Path to the versioned model YAML.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50304,
        help="Tokenizer vocabulary size; replace with the selected tokenizer value.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config, context_length = load_model_config(args.config, args.vocab_size)
    print(f"context_length={context_length}")

    device = torch.device(args.device)
    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=context_length, device=torch.device("cpu"))
    model.to(device)
    if model.embeddings.weight.shape != model.w_out.weight.shape:
        raise ValueError("embedding and output projection shapes do not match")
    model.w_out.weight = model.embeddings.weight
    model.train()

    print(f"total_params={sum(parameter.numel() for parameter in model.parameters()):,}")

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, args.vocab_size),
        labels[:, :-1].reshape(-1),
    )
    loss.backward()
    optimizer.step()
    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
