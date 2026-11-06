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
        self.aliases = tuple(
            (name, parameter) for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        )

    def install(self, optimizer):
        grouped = [parameter for group in optimizer.param_groups for parameter in group['params']]
        expected = {id(master) for _, _, master in self.named}
        if len(grouped) != len(expected) or {id(parameter) for parameter in grouped} != expected:
            raise ValueError("Optimizer must contain every FP32 master parameter exactly once")
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

    def _validate_state(self, state):
        expected = {name for name, _, _ in self.named}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("FP32 master checkpoint parameters differ from the model")
        for name, _, master in self.named:
            source = state[name]
            if (
                not isinstance(source, torch.Tensor)
                or source.shape != master.shape
                or source.dtype != torch.float32
                or source.layout != torch.strided
            ):
                raise ValueError(f"invalid FP32 master tensor: {name}")
            if not torch.isfinite(source).all().item():
                raise ValueError(f"nonfinite FP32 master tensor: {name}")

    def validate_checkpoint(self, checkpoint):
        state = checkpoint.get('fp32_master')
        self._validate_state(state)
        model_state = checkpoint.get('model')
        if not isinstance(model_state, dict):
            raise ValueError("FP32 master checkpoint is missing model weights")
        primary_names = {id(parameter): name for name, parameter, _ in self.named}
        for name, parameter in self.aliases:
            saved = model_state.get(name)
            if (
                not isinstance(saved, torch.Tensor)
                or saved.shape != parameter.shape
                or saved.dtype != parameter.dtype
                or saved.layout != torch.strided
                or not torch.isfinite(saved).all().item()
            ):
                raise ValueError(f"invalid saved model tensor: {name}")
            rounded = state[primary_names[id(parameter)]].to(device=saved.device, dtype=saved.dtype)
            if not torch.equal(rounded, saved):
                raise ValueError(f"Model and rounded FP32 master disagree for {name}")

    def load_state_dict(self, state):
        self._validate_state(state)
        with torch.no_grad():
            for name, parameter, master in self.named:
                master.copy_(state[name])
                parameter.copy_(master)
