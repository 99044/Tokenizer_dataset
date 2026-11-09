"""Optional Hopper kernels for stock packed Qwen3 MoE parameters.

No parameter replacement, token dropping, CPU routing-count transfer, or FP8.
Auto modes fall back before training starts and report the reason. Explicit
kernel requests fail if unavailable. Exported checkpoints remain ordinary HF.
"""

from importlib.metadata import version
import os
from types import MethodType

import torch
import torch.nn.functional as F


def swiglu(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def grouped_experts(module, hidden, indices, weights, gmm, activation=swiglu):
    """Sort assignments once, do two grouped GEMMs, then restore token order."""
    flat = indices.reshape(-1)
    permutation = torch.argsort(flat, stable=True)
    token_ids = torch.div(permutation, indices.shape[-1], rounding_mode="floor")
    # Fixed output size; unlike bincount this does not discover a size on CPU.
    counts = torch.zeros(module.num_experts, device=flat.device, dtype=torch.int64)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    dispatched = hidden.index_select(0, token_ids)
    gate_up = gmm(dispatched, module.gate_up_proj, counts, trans_b=True)
    output = gmm(activation(gate_up), module.down_proj, counts, trans_b=True)
    output = output * weights.reshape(-1).index_select(0, permutation).unsqueeze(-1)
    return torch.zeros_like(hidden).index_add_(0, token_ids, output.to(hidden.dtype))


def _grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    return grouped_experts(self, hidden_states, top_k_index, top_k_weights,
                           self._h800_gmm, self._h800_activation)


def check_grouped_gemm(gmm, device, activation=swiglu):
    """Exercise BF16 forward and both gradients, including an unused expert."""
    with torch.random.fork_rng(devices=[device]), torch.enable_grad():
        torch.manual_seed(7)
        a = (torch.randn(11, 64, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
        b = (torch.randn(4, 128, 64, device=device, dtype=torch.bfloat16) * 0.1).requires_grad_()
        counts = torch.tensor([3, 0, 5, 3], device=device, dtype=torch.int64)
        actual = activation(gmm(a, b, counts, trans_b=True))
        expected = swiglu(torch.cat([F.linear(x, w) for x, w in zip(a.split([3, 0, 5, 3]), b)]))
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (a, b), gradient, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (a, b), gradient)
        for got, ref in zip((actual, *actual_grads), (expected, *expected_grads)):
            torch.testing.assert_close(got, ref, rtol=0.04, atol=0.004)
        if torch.count_nonzero(actual_grads[1][1]).item():
            raise RuntimeError("Grouped GEMM did not zero the unused expert gradient")


def check_flash_attention_3(device):
    from flash_attn_interface import flash_attn_func

    with torch.random.fork_rng(devices=[device]), torch.enable_grad():
        torch.manual_seed(8)
        q = torch.randn(2, 32, 4, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        k, v = [torch.randn(2, 32, 2, 64, device=device, dtype=torch.bfloat16,
                            requires_grad=True) for _ in range(2)]
        actual = flash_attn_func(q, k, v, causal=True)
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.repeat_interleave(2, dim=2).transpose(1, 2),
            v.repeat_interleave(2, dim=2).transpose(1, 2), is_causal=True,
        ).transpose(1, 2)
        gradient = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (q, k, v), gradient, retain_graph=True)
        refs = torch.autograd.grad(expected, (q, k, v), gradient)
        for got, ref in zip((actual, *grads), (expected, *refs)):
            torch.testing.assert_close(got, ref, rtol=0.04, atol=0.03)


def runtime_options(settings=None):
    """Environment overrides are also recorded in the exact-resume plan."""
    options = dict(experts="auto", attention="auto", compile_swiglu=True)
    options.update(settings or {})
    for key, env in (("experts", "QWEN_MOE_EXPERTS"), ("attention", "QWEN_MOE_ATTENTION")):
        if env in os.environ:
            options[key] = os.environ[env]
    if "QWEN_MOE_COMPILE_SWIGLU" in os.environ:
        value = os.environ["QWEN_MOE_COMPILE_SWIGLU"]
        if value not in {"0", "1"}:
            raise ValueError("QWEN_MOE_COMPILE_SWIGLU must be 0 or 1")
        options["compile_swiglu"] = value == "1"
    if options["experts"] not in {"auto", "eager", "grouped_gemm"}:
        raise ValueError("experts must be auto, eager or grouped_gemm")
    if options["attention"] not in {"auto", "sdpa", "flash_attention_3"}:
        raise ValueError("attention must be auto, sdpa or flash_attention_3")
    if type(options["compile_swiglu"]) is not bool or set(options) != {"experts", "attention", "compile_swiglu"}:
        raise ValueError("Invalid Hopper acceleration options")
    return options


def install_h800_acceleration(model, options=None):
    """Install on a loaded CUDA model, before DDP/optimizer or timed evaluation."""
    options = runtime_options(options)
    if model.config.model_type != "qwen3_moe":
        raise ValueError("Hopper acceleration currently supports Qwen3 MoE only")
    parameter = next(model.parameters())
    device = parameter.device
    hopper = device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)
    report = dict(requested=options, experts="eager", attention="sdpa",
                  compile_swiglu=False, grouped_layers=0, fallback_reasons=[],
                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
    before = tuple((name, id(p), tuple(p.shape)) for name, p in model.named_parameters())
    eligible = hopper and parameter.dtype == torch.bfloat16

    def unavailable(feature, requested, error):
        if requested != "auto":
            raise RuntimeError(f"Requested {feature} unavailable: {error}") from error
        report["fallback_reasons"].append(f"{feature}: {type(error).__name__}: {error}")

    if options["experts"] != "eager":
        try:
            if not eligible:
                raise RuntimeError("requires SM90 CUDA and BF16 weights")
            if model.config.hidden_act != "silu":
                raise RuntimeError("grouped SwiGLU requires hidden_act=silu")
            if (model.config.hidden_size % 8 or model.config.moe_intermediate_size % 8
                    or model.config.num_experts > 1024):
                raise RuntimeError("CUTLASS requires dimensions aligned to 8 and <=1024 experts")
            if version("grouped_gemm") != "0.2.0":
                raise RuntimeError("requires grouped_gemm 0.2.0 built for this PyTorch/CUDA ABI")
            from grouped_gemm.ops import gmm

            check_grouped_gemm(gmm, device)
            activation = swiglu
            if options["compile_swiglu"]:
                try:
                    compiled = torch.compile(swiglu, fullgraph=True, dynamic=True)
                    check_grouped_gemm(gmm, device, compiled)
                    activation = compiled
                    report["compile_swiglu"] = True
                except Exception as error:
                    report["fallback_reasons"].append(f"SwiGLU compilation: {type(error).__name__}: {error}")
            modules = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeExperts"]
            if len(modules) != model.config.num_hidden_layers:
                raise RuntimeError("Expected homogeneous stock Qwen3 MoE experts")
            for module in modules:
                module._h800_gmm = gmm
                module._h800_activation = activation
                module.forward = MethodType(_grouped_forward, module)
            report.update(experts="grouped_gemm", grouped_layers=len(modules),
                          grouped_gemm_version=version("grouped_gemm"))
        except Exception as error:
            unavailable("grouped_gemm", options["experts"], error)

    if options["attention"] != "sdpa":
        try:
            if not eligible:
                raise RuntimeError("requires SM90 CUDA and BF16 weights")
            # Do not invoke Transformers' kernel-hub download fallback.
            from transformers.utils import is_flash_attn_3_available

            if not is_flash_attn_3_available():
                raise ImportError("local flash-attn-3 package not installed")
            fa3_version = version("flash-attn-3")
            check_flash_attention_3(device)
            model.set_attn_implementation("flash_attention_3")
            report["attention"] = "flash_attention_3"
            report["flash_attention_3_version"] = fa3_version
        except Exception as error:
            model.set_attn_implementation("sdpa")
            unavailable("flash_attention_3", options["attention"], error)

    after = tuple((name, id(p), tuple(p.shape)) for name, p in model.named_parameters())
    if before != after:
        raise RuntimeError("Kernel installation changed model parameters")
    return report
