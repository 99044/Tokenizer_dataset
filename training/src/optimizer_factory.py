"""Build optimizer experiments with explicit, auditable parameter groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from compressed_attention import CompressedKVBranch


class AdamWDtypeBuckets:
    """Keep Dion's fused AdamW calls homogeneous without changing saved groups."""

    def _create_adamw_tasks(self, param_groups):
        for group in param_groups:
            buckets = {}
            for parameter in group['params']:
                key = (parameter.device, parameter.dtype, parameter.layout)
                buckets.setdefault(key, []).append(parameter)
            for parameters in buckets.values():
                yield from super()._create_adamw_tasks([
                    dict(group, params=parameters)
                ])


@dataclass(frozen=True)
class NamedParameterGroup:
    name: str
    algorithm: str
    learning_rate: float
    weight_decay: float
    parameters: tuple[tuple[str, nn.Parameter], ...]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.parameters)


def _positive_float(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"optimizer.{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"optimizer.{field} must be numeric") from None
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"optimizer.{field} must be {'nonnegative' if allow_zero else 'positive'}")
    return result


def _matrix_and_scalar_parameters(
    model: nn.Module,
    excluded_parameter_ids: frozenset[int] = frozenset(),
) -> tuple[tuple[tuple[str, nn.Parameter], ...], tuple[tuple[str, nn.Parameter], ...]]:
    """Separate internal matrices from embedding, output, convolution and vectors."""

    named = tuple(
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in excluded_parameter_ids
    )
    if not named:
        raise ValueError("model has no trainable parameters")

    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    output_ids: set[int] = set()
    for path in (("w_out",), ("lm_head", "w_out")):
        module: Any = model
        for attribute in path:
            module = getattr(module, attribute, None)
            if module is None:
                break
        if isinstance(module, nn.Module):
            output_ids.update(id(parameter) for parameter in module.parameters(recurse=False))

    convolution_ids = {
        id(parameter)
        for module in model.modules()
        if type(module).__name__ == "CausalConv1d"
        for parameter in module.parameters(recurse=False)
    }
    protected_ids = embedding_ids | output_ids | convolution_ids
    matrix = tuple(
        (name, parameter)
        for name, parameter in named
        if parameter.ndim == 2 and id(parameter) not in protected_ids
    )
    matrix_ids = {id(parameter) for _, parameter in matrix}
    scalar = tuple((name, parameter) for name, parameter in named if id(parameter) not in matrix_ids)

    grouped_ids = [id(parameter) for group in (matrix, scalar) for _, parameter in group]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(p) for _, p in named}:
        raise RuntimeError("optimizer parameter groups are duplicated or incomplete")
    if not matrix or not scalar:
        raise RuntimeError("matrix-aware optimizer requires nonempty matrix and scalar groups")
    if any(parameter.ndim != 2 for _, parameter in matrix):
        raise RuntimeError("matrix optimizer group contains a non-matrix parameter")
    return matrix, scalar


def _report(name: str, groups: list[NamedParameterGroup]) -> dict[str, Any]:
    total = sum(group.parameter_count for group in groups)
    if total <= 0:
        raise RuntimeError("optimizer report has no parameters")
    return {
        "optimizer": name,
        "total_parameters": total,
        "groups": [
            {
                "name": group.name,
                "algorithm": group.algorithm,
                "learning_rate": group.learning_rate,
                "weight_decay": group.weight_decay,
                "parameter_count": group.parameter_count,
                "parameter_fraction": group.parameter_count / total,
                "parameter_tensors": len(group.parameters),
                "parameters": [name for name, _ in group.parameters],
            }
            for group in groups
        ],
    }


def build_optimizer(
    model: nn.Module,
    settings: dict[str, Any],
    *,
    primary_learning_rate: float | None = None,
    parameter_map: dict[int, nn.Parameter] | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    config = settings["optimizer"]
    name = str(config["name"]).lower()
    primary_lr = _positive_float(
        primary_learning_rate if primary_learning_rate is not None else config["learning_rate"],
        "learning_rate",
    )
    weight_decay = _positive_float(config["weight_decay"], "weight_decay", allow_zero=True)
    betas = tuple(float(value) for value in config["betas"])
    if len(betas) != 2 or any(not 0 <= value < 1 for value in betas):
        raise ValueError("optimizer.betas must contain two values in [0, 1)")
    eps = _positive_float(config["eps"], "eps")
    compressed_algorithm = config.get("compressed_branch")
    if compressed_algorithm not in {None, "adamw"}:
        raise ValueError("optimizer.compressed_branch must be adamw when specified")
    compressed_ids = frozenset()
    if compressed_algorithm == "adamw":
        compressed_ids = frozenset(
            id(parameter)
            for module in model.modules() if isinstance(module, CompressedKVBranch)
            for parameter in module.parameters() if parameter.requires_grad
        )
    named = tuple(
        (name, parameter_map[id(parameter)] if parameter_map is not None else parameter)
        for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    compressed = tuple(
        (parameter_name, parameter_map[id(parameter)] if parameter_map is not None else parameter)
        for parameter_name, parameter in model.named_parameters()
        if id(parameter) in compressed_ids
    )
    compressed_group = NamedParameterGroup(
        "compressed", "adamw", primary_lr, weight_decay, compressed
    )

    if name == "adamw":
        group = NamedParameterGroup("all", "adamw", primary_lr, weight_decay, named)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=primary_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        return optimizer, _report(name, [group])

    if name in {"muon", "normuon"}:
        matrix, scalar = _matrix_and_scalar_parameters(model, compressed_ids)
        if parameter_map is not None:
            matrix = tuple((name, parameter_map[id(parameter)]) for name, parameter in matrix)
            scalar = tuple((name, parameter_map[id(parameter)]) for name, parameter in scalar)
        scalar_lr = _positive_float(config["scalar_learning_rate"], "scalar_learning_rate")
        adjust_lr = config.get("adjust_lr")
        if adjust_lr != "rms_norm":
            raise ValueError("Muon/NorMuon experiments require optimizer.adjust_lr=rms_norm")
        groups = [
            NamedParameterGroup("matrix", name, primary_lr, weight_decay, matrix),
            NamedParameterGroup("scalar", "adamw", scalar_lr, weight_decay, scalar),
        ]
        try:
            from dion import Muon, NorMuon
        except ImportError as error:
            raise RuntimeError(
                "Muon/NorMuon requires the pinned optimizer experiment image with dion installed"
            ) from error
        optimizer_class = Muon if name == "muon" else NorMuon
        parameter_groups = [
            {
                "params": [parameter for _, parameter in matrix],
                "algorithm": "muon" if name == "muon" else "normuon",
                "lr": primary_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [parameter for _, parameter in scalar],
                "algorithm": "adamw",
                "lr": scalar_lr,
                "weight_decay": weight_decay,
            },
        ]
        if compressed:
            groups.append(compressed_group)
            parameter_groups.append({
                "params": [parameter for _, parameter in compressed],
                "algorithm": "adamw",
                "lr": primary_lr,
                "weight_decay": weight_decay,
            })
        optimizer_kwargs = {
            "distributed_mesh": dist.group.WORLD if dist.is_initialized() else None,
            "lr": primary_lr,
            "mu": float(config.get("momentum", 0.95)),
            "betas": betas,
            "weight_decay": weight_decay,
            "epsilon": eps,
            "nesterov": bool(config.get("nesterov", False)),
            "adjust_lr": adjust_lr,
            "flatten": False,
            "use_triton": bool(config.get("use_triton", False)),
        }
        if name == "normuon":
            optimizer_kwargs["muon_beta2"] = _positive_float(
                config.get("muon_beta2", 0.95), "muon_beta2"
            )
        class CompatibleOptimizer(AdamWDtypeBuckets, optimizer_class):
            pass

        optimizer = CompatibleOptimizer(parameter_groups, **optimizer_kwargs)
        return optimizer, _report(name, groups)

    if name == "soap":
        from third_party.soap import SOAP

        precondition_frequency = int(config.get("precondition_frequency", 10))
        max_precond_dim = int(config.get("max_precond_dim", 1024))
        if precondition_frequency <= 0 or max_precond_dim <= 0:
            raise ValueError("SOAP precondition_frequency and max_precond_dim must be positive")
        compressed_optimizer_ids = {id(parameter) for _, parameter in compressed}
        ordinary = tuple(
            (parameter_name, parameter) for parameter_name, parameter in named
            if id(parameter) not in compressed_optimizer_ids
        )
        group = NamedParameterGroup(
            "ordinary" if compressed else "all", "soap", primary_lr, weight_decay, ordinary
        )
        optimizer = SOAP(
            [parameter for _, parameter in ordinary],
            lr=primary_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=False,
            precondition_1d=False,
            normalize_grads=False,
            correct_bias=True,
        )
        if compressed:
            from combined_optimizer import CombinedOptimizer

            compressed_optimizer = torch.optim.AdamW(
                [parameter for _, parameter in compressed],
                lr=primary_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            optimizer = CombinedOptimizer(optimizer, compressed_optimizer)
            return optimizer, _report(name, [group, compressed_group])
        return optimizer, _report(name, [group])

    raise ValueError(f"unsupported optimizer: {name!r}")
