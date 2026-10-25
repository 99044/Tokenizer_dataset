"""Keep FP32 master weights and optimizer state for BF16 model updates."""

from __future__ import annotations

from types import MethodType

import torch


class FP32MasterWeights:
    def __init__(self, model):
        self.named = tuple(
            (name, parameter, torch.nn.Parameter(parameter.detach().float().clone()))
            for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        if not self.named or any(parameter.dtype not in {torch.bfloat16, torch.float32} for _, parameter, _ in self.named):
            raise ValueError("FP32 master mode requires BF16/FP32 model parameters")
        self.parameter_map = {id(parameter): master for _, parameter, master in self.named}

    def install(self, optimizer):
        original_step = optimizer.step
        original_zero_grad = optimizer.zero_grad
        named = self.named

        def step(_, *args, **kwargs):
            for _, parameter, master in named:
                master.grad = None if parameter.grad is None else parameter.grad.detach().float()
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                for _, parameter, master in named:
                    parameter.copy_(master)
            return result

        def zero_grad(_, *args, **kwargs):
            for _, parameter, _ in named:
                parameter.grad = None
            return original_zero_grad(*args, **kwargs)

        optimizer.step = MethodType(step, optimizer)
        optimizer.zero_grad = MethodType(zero_grad, optimizer)
        return optimizer

    def state_dict(self):
        return {name: master.detach().cpu() for name, _, master in self.named}

    def load_state_dict(self, state):
        expected = {name for name, _, _ in self.named}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("FP32 master checkpoint parameters differ from the model")
        with torch.no_grad():
            for name, parameter, master in self.named:
                source = state[name]
                if source.shape != master.shape or source.dtype != torch.float32:
                    raise ValueError(f"invalid FP32 master tensor: {name}")
                master.copy_(source)
                parameter.copy_(master)
