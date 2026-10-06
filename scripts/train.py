#!/usr/bin/env python3
"""Train the SCICOMP26 baseline on packed token arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from comp.scicomp26.scripts.model_config import build_model_config, count_parameters, tie_output_weights


class PackedTokenSampler:
    def __init__(self, path: Path, sequence_length: int, vocab_size: int, seed: int) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        if self.tokens.ndim != 1:
            raise ValueError(f"{path} must contain a one-dimensional token array")
        if not np.issubdtype(self.tokens.dtype, np.integer):
            raise ValueError(f"{path} must contain integer token IDs")
        if self.tokens.size and int(self.tokens.max()) >= vocab_size:
            raise ValueError(f"{path} contains a token ID outside vocab_size={vocab_size}")
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)
        self.max_start = len(self.tokens) - sequence_length - 1
        if self.max_start < 0:
            raise ValueError(f"{path} is shorter than sequence_length + 1")

    def batch(
        self, batch_size: int, device: torch.device, starts: Optional[np.ndarray] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if starts is None:
            starts = self.rng.integers(0, self.max_start + 1, size=batch_size)
        inputs = np.empty((len(starts), self.sequence_length), dtype=np.int64)
        labels = np.empty_like(inputs)
        for row, start in enumerate(starts):
            inputs[row] = self.tokens[start : start + self.sequence_length]
            labels[row] = self.tokens[start + 1 : start + self.sequence_length + 1]
        return (
            torch.from_numpy(inputs).to(device=device),
            torch.from_numpy(labels).to(device=device),
        )


def evaluate(
    model: torch.nn.Module,
    sampler: PackedTokenSampler,
    batch_size: int,
    batches: int,
    vocab_size: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, labels = sampler.batch(batch_size, device)
            logits = model(inputs)
            losses.append(
                F.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, vocab_size),
                    labels[:, :-1].reshape(-1),
                ).item()
            )
    model.train()
    return float(np.mean(losses))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> None:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/200m_dense.yaml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dense_200m_baseline"))
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metadata = json.loads((args.data_dir / "metadata.json").read_text())
    if metadata["vocab_size"] != args.vocab_size:
        raise ValueError("metadata vocab_size does not match --vocab-size")
    config, context_length, settings = build_model_config(args.config, args.vocab_size)
    if metadata["context_length"] != context_length:
        raise ValueError("metadata context_length does not match the model config")
    device = torch.device(args.device)
    torch.manual_seed(settings["run"]["seed"])

    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=context_length, device=torch.device("cpu"))
    model.to(device)
    tie_output_weights(model)
    model.train()
    print(f"total_params={count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"resumed_step={start_step}")

    train_sampler = PackedTokenSampler(
        args.data_dir / "train.npy", context_length, args.vocab_size, 42
    )
    validation_sampler = PackedTokenSampler(
        args.data_dir / "validation.npy", context_length, args.vocab_size, 43
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb

        logging = settings["logging"]
        wandb_run = wandb.init(
            entity=logging["entity"],
            project=logging["project"],
            name=settings["run"]["name"],
            config={"model": settings["model"], "data": metadata, "max_steps": args.max_steps},
        )

    for step in range(start_step + 1, args.max_steps + 1):
        inputs, labels = train_sampler.batch(args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, args.vocab_size),
            labels[:, :-1].reshape(-1),
        )
        loss.backward()
        optimizer.step()

        scheduler.step()
        metrics = {
            "step": step,
            "train/loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
        }
        if step % args.eval_every == 0:
            metrics["validation/loss"] = evaluate(
                model,
                validation_sampler,
                args.batch_size,
                args.eval_batches,
                args.vocab_size,
                device,
            )
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                args.output_dir / f"step-{step}.pt", model, optimizer, scheduler, step
            )
        if step % args.log_every == 0:
            print(
                " ".join(
                    f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in metrics.items()
                )
            )
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
