#!/usr/bin/env python3
"""Build and run one training step for the SCICOMP26 200M baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model_config import build_model_config, count_parameters, tie_output_weights


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

    config, context_length, _ = build_model_config(args.config, args.vocab_size)
    print(f"context_length={context_length}")

    device = torch.device(args.device)
    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=context_length, device=torch.device("cpu"))
    model.to(device)
    tie_output_weights(model)
    model.train()

    print(f"total_params={count_parameters(model):,}")

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
