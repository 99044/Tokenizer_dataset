"""Sample shifted sequences, reusing verification of unchanged token caches."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from token_cache_validation import ensure_array


class PackedTokenSampler:
    def __init__(self, path: Path, sequence_length: int, vocab_size: int, seed: int,
                 metadata: Optional[dict] = None) -> None:
        path = Path(path)
        self.range_verified_from_cache = ensure_array(path, metadata, vocab_size)
        # Reload after a possible validation wait so an old mapping is not
        # retained if another process replaced the file before validation.
        self.tokens = np.load(path, mmap_mode='r', allow_pickle=False)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.max_start = len(self.tokens) - sequence_length - 1
        if self.max_start < 0:
            raise ValueError(f'{path} is shorter than sequence_length + 1')

    def batch(self, batch_size: int, device: torch.device,
              starts: Optional[np.ndarray] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if starts is None:
            starts = self.rng.integers(0, self.max_start + 1, size=batch_size)
        inputs = np.empty((len(starts), self.sequence_length), dtype=np.int64)
        labels = np.empty_like(inputs)
        for row, start in enumerate(starts):
            inputs[row] = self.tokens[start:start + self.sequence_length]
            labels[row] = self.tokens[start + 1:start + self.sequence_length + 1]
        # Small CPU batches are always checked before the embedding lookup,
        # including when the startup range scan was reused from preprocessing.
        if min(inputs.min(), labels.min()) < 0 or max(inputs.max(), labels.max()) >= self.vocab_size:
            raise ValueError(f'Sampled token ID is outside [0, {self.vocab_size})')
        return torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)
