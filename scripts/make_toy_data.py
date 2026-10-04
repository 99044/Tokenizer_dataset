#!/usr/bin/env python3
"""Create a deterministic packed-token dataset for pipeline validation."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np


def make_tokens(length: int, period: int, offset: int) -> np.ndarray:
    return ((np.arange(length, dtype=np.uint64) + offset) % period).astype(np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--train-tokens", type=int, default=65536)
    parser.add_argument("--validation-tokens", type=int, default=16384)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--period", type=int, default=64)
    args = parser.parse_args()

    if args.period <= 1 or args.period >= args.vocab_size:
        raise ValueError("period must be greater than 1 and smaller than vocab_size")
    if args.train_tokens <= args.context_length or args.validation_tokens <= args.context_length:
        raise ValueError("each split must contain more than context_length tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tokenizer").mkdir(exist_ok=True)
    np.save(args.output_dir / "train.npy", make_tokens(args.train_tokens, args.period, 0))
    np.save(
        args.output_dir / "validation.npy",
        make_tokens(args.validation_tokens, args.period, args.period // 2),
    )
    metadata = {
        "data_version": "synthetic-v1",
        "source": "deterministic periodic token stream",
        "tokenizer": "synthetic-pattern",
        "vocab_size": args.vocab_size,
        "eos_token_id": 0,
        "train_tokens": args.train_tokens,
        "validation_tokens": args.validation_tokens,
        "context_length": args.context_length,
        "created_at": date.today().isoformat(),
        "notes": f"Pipeline validation only; period={args.period}.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
