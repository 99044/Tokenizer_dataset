#!/usr/bin/env python3
"""Native H800 correctness and isolated expert forward/backward benchmark.

Uses each tier's real expert dimensions, not a full-training throughput claim.
No downloads, datasets, optimizer updates, or saved checkpoints are involved.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeRMSNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/src"))
sys.path.insert(0, str(ROOT / "evaluation"))
from h800_acceleration import install_h800_acceleration
from fusion import _fused_rms_forward


def benchmark(call, iterations, device):
    for _ in range(5):
        call()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        call()
    stop.record()
    torch.cuda.synchronize(device)
    return dict(cuda_ms_per_call=start.elapsed_time(stop) / iterations,
                wall_ms_per_call=(time.perf_counter() - wall) * 1000 / iterations,
                peak_extra_allocated_bytes=torch.cuda.max_memory_allocated(device) - baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("3e17", "1e18", "3e18", "1e19"), default="3e17")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens < 16 or args.iterations < 1:
        parser.error("tokens >=16 and iterations >=1 required")
    if args.output and args.output.exists():
        parser.error("Output already exists")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise SystemExit("Native verification requires an H800/SM90 CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    config = Qwen3MoeConfig.from_dict(json.loads((ROOT / "qwen3_moe/configs" / args.tier / "config.json").read_text()))
    # A one-layer shell allows the same installer used in training/evaluation.
    config.num_hidden_layers = 1
    config._attn_implementation = "sdpa"
    stock = Qwen3MoeExperts(config).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in stock.parameters():
            parameter.normal_(0, 0.02)
    fast = deepcopy(stock)

    class Shell(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.config, self.experts = config, experts

    report = install_h800_acceleration(Shell(fast), {"experts": "grouped_gemm", "attention": "sdpa"})
    tokens, hidden, top_k = args.tokens, config.hidden_size, config.num_experts_per_tok
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
    # Deliberately leave the final expert unused and exercise uneven group sizes.
    indices = torch.randn(tokens, config.num_experts - 1, device=device).topk(top_k, dim=-1).indices
    weights = torch.randn(tokens, top_k, device=device).softmax(-1).to(torch.bfloat16).requires_grad_()
    gradient = torch.randn_like(x) * 0.01

    def run(module, backward):
        module.zero_grad(set_to_none=True)
        x.grad = weights.grad = None
        output = module(x, indices, weights)
        if backward:
            output.backward(gradient)
        return output

    expected = run(stock, True).detach().clone()
    refs = [p.grad.clone() for p in (x, weights, *stock.parameters())]
    actual = run(fast, True).detach()
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.004)
    for got, ref in zip((x, weights, *fast.parameters()), refs):
        torch.testing.assert_close(got.grad, ref, rtol=0.06, atol=0.004)
    if torch.count_nonzero(fast.gate_up_proj.grad[-1]).item():
        raise RuntimeError("Unused expert received nonzero gradients")
    for width in (config.head_dim, hidden):
        norm = Qwen3MoeRMSNorm(width).to(device=device, dtype=torch.bfloat16).eval()
        norm._fusion_original_forward = norm.forward
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            inputs = torch.randn(2, 13, width, device=device, dtype=torch.bfloat16)
            torch.testing.assert_close(_fused_rms_forward(norm, inputs), norm(inputs), rtol=0.01, atol=0.01)
    timings = {}
    for name, module in (("eager", stock), ("grouped_gemm", fast)):
        with torch.inference_mode():
            timings[name + "_forward"] = benchmark(lambda: run(module, False), args.iterations, device)
        timings[name + "_forward_backward"] = benchmark(lambda: run(module, True), args.iterations, device)
    result = dict(scope="isolated MoE expert block; excludes attention, DDP, SOAP and data loading",
                  gpu=torch.cuda.get_device_name(device), torch=torch.__version__, tier=args.tier,
                  tokens=tokens, hidden=hidden, experts=config.num_experts, top_k=top_k,
                  expert_width=config.moe_intermediate_size, correctness="passed",
                  acceleration=report, timings=timings)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
