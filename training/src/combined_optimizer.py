"""Expose SOAP and compressed-branch AdamW through one training optimizer."""

from collections import defaultdict

import torch


class CombinedOptimizer(torch.optim.Optimizer):
    """Share parameter groups with SOAP and AdamW for a single LR scheduler."""

    def __init__(self, soap_optimizer, adamw_optimizer):
        if type(soap_optimizer).__name__ != "SOAP" or not isinstance(
            adamw_optimizer, torch.optim.AdamW
        ):
            raise TypeError("CombinedOptimizer requires SOAP followed by AdamW")
        self.optimizers = (soap_optimizer, adamw_optimizer)
        groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]
        parameters = [parameter for group in groups for parameter in group["params"]]
        if len(parameters) != len({id(parameter) for parameter in parameters}):
            raise ValueError("SOAP and AdamW parameter groups must be disjoint")
        super().__init__(groups, {})
        self._refresh_groups_and_state()

    def _refresh_groups_and_state(self):
        # Child loaders replace group dictionaries. Keep scheduler writes shared.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self.state = defaultdict(
            dict,
            {
                parameter: state
                for optimizer in self.optimizers
                for parameter, state in optimizer.state.items()
            },
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        self._refresh_groups_and_state()
        return loss

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "kind": "soap_adamw",
            "version": 1,
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or state_dict.get("kind") != "soap_adamw"
            or state_dict.get("version") != 1
        ):
            raise ValueError(
                "Checkpoint optimizer layout is incompatible with SOAP + compressed-branch "
                "AdamW. Use a checkpoint saved with the same optimizer configuration."
            )
        saved_optimizers = state_dict.get("optimizers")
        if not isinstance(saved_optimizers, list) or len(saved_optimizers) != 2:
            raise ValueError("SOAP + AdamW checkpoint must contain exactly two optimizers")
        # Check both layouts before allowing either child to load its state.
        for optimizer, saved in zip(self.optimizers, saved_optimizers):
            if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
                raise ValueError("Invalid child optimizer checkpoint state")
            saved_groups = saved.get("param_groups")
            if not isinstance(saved_groups, list) or len(saved_groups) != len(optimizer.param_groups):
                raise ValueError("Checkpoint optimizer parameter-group count differs")
            for group, saved_group in zip(optimizer.param_groups, saved_groups):
                if (
                    not isinstance(saved_group, dict)
                    or not isinstance(saved_group.get("params"), list)
                    or len(saved_group["params"]) != len(group["params"])
                ):
                    raise ValueError("Checkpoint optimizer parameter count differs")
        for optimizer, saved in zip(self.optimizers, saved_optimizers):
            # In particular, SOAP restores its BF16 parameters' FP32 statistics.
            optimizer.load_state_dict(saved)
        self._refresh_groups_and_state()
