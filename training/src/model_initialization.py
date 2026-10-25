"""Keep sampled GDN decay rates finite before optimizer weight decay."""

import math

import torch


@torch.no_grad()
def repair_zero_gdn_decay_rates(model):
    """Replace log(0) from BF16 initialization without resampling any weights.

    OLMo-core initializes A_log with log(uniform(0, 16)). BF16 uniform
    sampling can return exactly zero. Its log is -inf, whose AdamW/SOAP
    weight decay produces NaN. Represent only these zero rates by the
    smallest positive normal rate supported by the parameter dtype.
    """
    repairs = []
    for name, module in model.named_modules():
        if type(module).__name__ != "GatedDeltaNet":
            continue
        parameter = module.A_log
        if torch.isnan(parameter).any().item() or torch.isposinf(parameter).any().item():
            raise ValueError(f"Unexpected nonfinite GDN initialization: {name}.A_log")
        zero_rates = torch.isneginf(parameter)
        if not zero_rates.any().item():
            continue
        indices = zero_rates.nonzero(as_tuple=False).cpu().tolist()
        floor = math.log(torch.finfo(parameter.dtype).tiny)
        parameter.masked_fill_(zero_rates, floor)
        repairs.append({
            "parameter": f"{name}.A_log",
            "indices": indices,
            "reason": "zero_sampled_decay_rate",
            "replacement_log_rate": float(parameter[zero_rates][0].item()),
        })
    return repairs
