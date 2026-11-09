"""Inference-only Triton kernels adapted from the H200 fusion experiment."""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_packed_kernel(output, gate_up, output_stride, gate_up_stride,
                            width: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    gate_values = tl.load(gate_up + row * gate_up_stride + offsets,
                          mask=mask, other=0.0).to(tl.float32)
    up_values = tl.load(gate_up + row * gate_up_stride + width + offsets,
                        mask=mask, other=0.0)
    activated = (gate_values * tl.sigmoid(gate_values)).to(gate_up.dtype.element_ty)
    tl.store(output + row * output_stride + offsets, activated * up_values, mask=mask)


@triton.jit
def _rms_norm_kernel(output, inputs, weight, row_stride, width: tl.constexpr,
                     eps: tl.constexpr, CAST_BEFORE_WEIGHT: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    values = tl.load(inputs + row * row_stride + offsets, mask=mask,
                     other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / width
    normalized = values * tl.rsqrt(variance + eps)
    if CAST_BEFORE_WEIGHT:
        normalized = normalized.to(inputs.dtype.element_ty).to(tl.float32)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + row * row_stride + offsets, normalized * weights, mask=mask)


def fused_silu_mul_packed(gate_up):
    if not gate_up.is_cuda or not gate_up.is_contiguous():
        raise ValueError('packed fused SwiGLU requires a contiguous CUDA tensor')
    if gate_up.shape[-1] % 2:
        raise ValueError('packed gate/up width must be even')
    width = gate_up.shape[-1] // 2
    rows = gate_up.numel() // gate_up.shape[-1]
    output = torch.empty((*gate_up.shape[:-1], width), device=gate_up.device,
                         dtype=gate_up.dtype)
    block = 256
    _silu_mul_packed_kernel[(rows, triton.cdiv(width, block))](
        output, gate_up, width, gate_up.shape[-1], width=width,
        BLOCK_SIZE=block, num_warps=4)
    return output


def fused_rms_norm(inputs, weight, eps, *, cast_before_weight=False):
    if not inputs.is_cuda or inputs.stride(-1) != 1:
        raise ValueError('fused RMSNorm requires CUDA input contiguous in its last axis')
    if weight.ndim != 1 or weight.numel() != inputs.shape[-1]:
        raise ValueError('RMSNorm weight shape does not match the input width')
    contiguous = inputs.contiguous()
    output = torch.empty_like(contiguous)
    width = contiguous.shape[-1]
    rows = contiguous.numel() // width
    block = triton.next_power_of_2(width)
    warps = 4 if block <= 2048 else 8
    _rms_norm_kernel[(rows,)](output, contiguous, weight, width, width=width,
                              eps=eps, CAST_BEFORE_WEIGHT=cast_before_weight,
                              BLOCK_SIZE=block, num_warps=warps)
    return output.view_as(inputs)
