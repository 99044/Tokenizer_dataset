"""Causal 128-token K/V compression beside one existing attention layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompressedKVBranch(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, block_size: int,
        intermediate_size: int, *, device: torch.device, dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.k_compress = nn.Linear(
            block_size * hidden_size, hidden_size, bias=False, device=device, dtype=dtype
        )
        self.v_compress = nn.Linear(
            block_size * hidden_size, hidden_size, bias=False, device=device, dtype=dtype
        )
        self.k_up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.k_down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.v_down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        if width != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {width}")
        complete_blocks = length // self.block_size
        if complete_blocks == 0:
            return torch.zeros_like(x)

        packed = x[:, :complete_blocks * self.block_size, :].reshape(
            batch, complete_blocks, self.block_size * width
        )
        key = self.k_compress(packed)
        value = self.v_compress(packed)
        key = key + self.k_down(F.silu(self.k_up(key)))
        value = value + self.v_down(F.silu(self.v_up(value)))

        head_dim = width // self.num_heads
        query = x.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, complete_blocks, self.num_heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, complete_blocks, self.num_heads, head_dim).transpose(1, 2)
        key = torch.cat((torch.zeros_like(key[:, :, :1]), key), dim=2)
        value = torch.cat((torch.zeros_like(value[:, :, :1]), value), dim=2)
        token_end = torch.arange(length, device=x.device)[:, None] + 1
        block_end = torch.arange(complete_blocks + 1, device=x.device)[None, :] * self.block_size
        allowed = block_end <= token_end
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False
        )
        return output.transpose(1, 2).contiguous().reshape(batch, length, width)

    def num_train_flops_per_token_upper_bound(self, sequence_length: int) -> int:
        """Conservative 3x-forward estimate for the added branch."""
        blocks = sequence_length // self.block_size
        if not blocks:
            return 0
        d, h, c = self.hidden_size, self.k_up.out_features, self.block_size
        # Two packed projections and four post-compression projections.
        linear_forward = (4 * blocks * c * d * d + 8 * blocks * d * h) / sequence_length
        # QK and AV each use one multiply-add per query and completed block.
        attention_forward = 4 * d * blocks
        return math.ceil(3 * (linear_forward + attention_forward))


class AugmentedAttention(nn.Module):
    def __init__(self, base: nn.Module, branch: CompressedKVBranch, scale: float) -> None:
        super().__init__()
        if not 0.0 < scale <= 1.0:
            raise ValueError("branch scale must be in (0, 1]")
        self.base = base
        self.compressed = branch
        self.scale = scale

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base(x, **kwargs) + self.scale * self.compressed(x)

    def num_flops_per_token(self, seq_len: int) -> int:
        # OLMo-core's model-level method uses forward FLOPs; the training
        # budget below uses a separate 3x-forward conservative estimate.
        return self.base.num_flops_per_token(seq_len) + math.ceil(
            self.compressed.num_train_flops_per_token_upper_bound(seq_len) / 3
        )


def attach_compressed_attention(model: nn.Module, settings: dict, device: torch.device) -> int:
    spec = settings["model"].get("compressed_attention")
    if not spec:
        return 0
    index = int(spec["layer_index"])
    block = model.blocks[str(index)]
    if isinstance(block.attention, AugmentedAttention):
        raise RuntimeError("compressed attention already attached")
    if (index + 1) % 4 != 0:
        raise ValueError("the pilot must attach to an attention, not a GDN, layer")
    branch = CompressedKVBranch(
        settings["model"]["hidden_size"], settings["model"]["num_attention_heads"],
        int(spec["block_size"]), int(spec["intermediate_size"]),
        device=device, dtype=next(block.attention.parameters()).dtype,
    )
    block.attention = AugmentedAttention(block.attention, branch, float(spec["scale"]))
    return sum(parameter.numel() for parameter in branch.parameters())
