"""Inference-only FFN and RMSNorm fusion for exported HF Hybrid models."""

from types import MethodType

import torch
from torch import nn


def _fused_mlp_forward(self, hidden_states):
    from fusion_kernels import fused_silu_mul_packed

    return self.down_proj(fused_silu_mul_packed(self._fusion_gate_up(hidden_states)))


def _fused_rms_forward(self, hidden_states):
    from fusion_kernels import fused_rms_norm

    if not hidden_states.is_cuda or hidden_states.stride(-1) != 1:
        return self._fusion_original_forward(hidden_states)
    return fused_rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _validate_linear_parts(parts):
    if any(not isinstance(part, nn.Linear) or part.bias is not None for part in parts):
        raise ValueError("FFN fusion requires bias-free Linear projections")
    width = parts[0].in_features
    dtype, device = parts[0].weight.dtype, parts[0].weight.device
    if any(part.in_features != width or part.weight.dtype != dtype
           or part.weight.device != device for part in parts):
        raise ValueError("FFN projections must share input width, dtype and device")
    if dtype != torch.bfloat16 or device.type != "cuda":
        raise ValueError("FFN fusion requires BF16 CUDA projections")


def _packed_linear(parts):
    """Pack gate and up into one projection without changing parameter count."""
    _validate_linear_parts(parts)
    packed = nn.Linear(
        parts[0].in_features, sum(part.out_features for part in parts),
        bias=False, device=parts[0].weight.device, dtype=parts[0].weight.dtype,
    )
    with torch.no_grad():
        packed.weight.copy_(torch.cat([part.weight for part in parts], dim=0))
    packed.requires_grad_(False)
    return packed


def install_operator_fusion(model, mode="ffn_rmsnorm"):
    """Install fusion after loading an HF checkpoint and before evaluation."""
    if mode not in ("none", "ffn_rmsnorm"):
        raise ValueError(f"Unknown fusion mode: {mode}")
    if mode == "none":
        return {"mode": mode, "packed_gate_up": 0, "rms_norm": 0}
    expected = int(getattr(model.config, "num_hidden_layers", 0))
    if getattr(model.config, "model_type", None) != "scicomp26_olmo_hybrid" or expected < 1:
        raise ValueError("Fusion only supports the submission Hybrid HF exports")
    modules = [module for module in model.modules() if type(module).__name__ == "OlmoHybridMLP"]
    if len(modules) != expected:
        raise ValueError(f"Expected {expected} HF MLP blocks, found {len(modules)}")
    for module in modules:
        gate, up, down = module.gate_proj, module.up_proj, module.down_proj
        _validate_linear_parts((gate, up))
        if not isinstance(down, nn.Linear) or down.bias is not None:
            raise ValueError("FFN fusion requires a bias-free down projection")
        if gate.out_features != up.out_features or down.in_features != gate.out_features:
            raise ValueError("FFN projection dimensions do not match")
        if getattr(module.config, "hidden_act", None) != "silu":
            raise ValueError("FFN fusion requires SiLU activation")
    norms = [module for module in model.modules()
             if type(module).__name__ == "SubmissionRMSNorm"]
    if not norms or any(module.weight is None or
                        not hasattr(module, "variance_epsilon") for module in norms):
        raise ValueError("Fusion requires affine SubmissionRMSNorm modules")

    before = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        for module in modules:
            gate, up = module.gate_proj, module.up_proj
            module._fusion_gate_up = _packed_linear((gate, up))
            module.gate_proj = nn.Identity()
            module.up_proj = nn.Identity()
            module.forward = MethodType(_fused_mlp_forward, module)
        for module in norms:
            module._fusion_original_forward = module.forward
            module.forward = MethodType(_fused_rms_forward, module)
    after = sum(parameter.numel() for parameter in model.parameters())
    if after != before:
        raise RuntimeError(f"Fusion changed parameter count: {before} -> {after}")
    return {"mode": mode, "packed_gate_up": len(modules), "rms_norm": len(norms)}
