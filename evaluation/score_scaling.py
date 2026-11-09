#!/usr/bin/env python3
"""Reproduce the supplied four-tier BPB scaling score for one A/B track.

Local fitting assumption, pending official parameter bounds: E >= 0, k > 0,
p > 0. Optimize E + a * (C / C_ref)**(-p) to avoid enormous raw k values.
Use reviewed actual training FLOPs as inputs, never nominal tier budgets.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

TIERS = ("3e17", "1e18", "3e18", "1e19")
C_MIN = 3 * 10**17
C_MAX = 10**19


def positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return value


def validate_points(document):
    if not isinstance(document, dict) or document.get("track") not in ("A", "B"):
        raise ValueError("Specify one track, A or B; fit the two tracks separately")
    points = document.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("Exactly four measured tier points are required")
    by_tier = {}
    for point in points:
        if not isinstance(point, dict) or point.get("tier") not in TIERS or point["tier"] in by_tier:
            raise ValueError("Supply each of 3e17, 1e18, 3e18 and 1e19 exactly once")
        if point.get("track", document["track"]) != document["track"]:
            raise ValueError("A and B observations cannot be mixed")
        by_tier[point["tier"]] = dict(
            tier=point["tier"],
            actual_training_flops=positive_number(point.get("actual_training_flops"), "actual_training_flops"),
            bpb=positive_number(point.get("bpb"), "bpb"),
        )
        family_id = point.get("architecture_family_id")
        if family_id is None:
            family_id = document.get("architecture_family_id")
        if family_id is not None:
            if (not isinstance(family_id, str) or len(family_id) != 64
                    or any(character not in "0123456789abcdef" for character in family_id)):
                raise ValueError("architecture_family_id must be the SHA-256 declaration ID")
            if document.get("architecture_family_id") not in (None, family_id):
                raise ValueError("Point architecture family differs from the document declaration")
            by_tier[point["tier"]]["architecture_family_id"] = family_id
    ordered = [by_tier[tier] for tier in TIERS]
    family_ids = [point.get("architecture_family_id") for point in ordered]
    if any(family_ids) and (any(family is None for family in family_ids) or len(set(family_ids)) != 1):
        raise ValueError("All four points must use the same declared architecture family")
    if any(a["actual_training_flops"] >= b["actual_training_flops"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Actual FLOPs must increase across the four tiers")
    return ordered


def fit_curve(compute, bpb):
    """Equal-weight nonlinear least squares in ln(BPB), including E=0 boundary."""
    reference = float(compute[0])
    x = np.log(compute / reference)
    y = np.log(bpb)

    def predictions(parameters):
        floor, amplitude, exponent = parameters
        decay = np.exp(-exponent * x)
        return floor + amplitude * decay, decay

    def residual(parameters):
        fitted, _ = predictions(parameters)
        return np.log(np.maximum(fitted, np.finfo(float).tiny)) - y

    def jacobian(parameters):
        fitted, decay = predictions(parameters)
        floor, amplitude, exponent = parameters
        fitted = np.maximum(fitted, np.finfo(float).tiny)
        return np.column_stack((1 / fitted, decay / fitted, -x * amplitude * decay / fitted))

    candidates = []
    # An exact log-linear fit covers the valid E=0 boundary explicitly.
    slope = np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean())
    if slope < 0:
        exponent = -float(slope)
        amplitude = math.exp(float(np.mean(y + exponent * x)))
        parameters = np.array([0.0, amplitude, exponent])
        candidates.append((float(np.dot(residual(parameters), residual(parameters))), parameters, "E=0 boundary"))
    for floor_fraction in (0.0, 0.5, 0.9):
        floor = floor_fraction * float(bpb.min())
        for exponent in (0.03, 0.1, 0.3, 1.0):
            initial = [floor, max(float(bpb[0]) - floor, 0.01 * float(bpb[0])), exponent]
            result = least_squares(
                residual, initial, jac=jacobian, bounds=(0, np.inf),
                loss="linear", x_scale="jac", max_nfev=4000,
                ftol=1e-12, xtol=1e-12, gtol=1e-12,
            )
            if result.success and np.all(np.isfinite(result.x)):
                candidates.append((float(np.dot(result.fun, result.fun)), result.x, "bounded multistart least squares"))
    if not candidates:
        raise ValueError("Scaling fit did not converge")
    _, parameters, method = min(candidates, key=lambda candidate: candidate[0])
    return parameters, predictions(parameters)[0], reference, method


def frontier_area(floor, amplitude, exponent, reference):
    """Mean fitted BPB under d ln C, using expm1 for small positive p."""
    width = math.log(C_MAX / C_MIN)
    if exponent == 0:
        return floor + amplitude
    mean_decay = -math.expm1(-exponent * width) / (exponent * width)
    at_min = amplitude * math.exp(-exponent * math.log(C_MIN / reference))
    return floor + at_min * mean_decay


def score(document):
    points = validate_points(document)
    compute = np.array([point["actual_training_flops"] for point in points], dtype=np.float64)
    measured = np.array([point["bpb"] for point in points], dtype=np.float64)
    log_measured = np.log(measured)
    sst = float(np.sum((log_measured - log_measured.mean()) ** 2))
    if sst == 0:
        raise ValueError("Identical BPB observations have undefined log-space R^2")
    parameters, fitted, reference, method = fit_curve(compute, measured)
    floor, amplitude, exponent = map(float, parameters)
    residuals = np.log(fitted) - log_measured
    sse = float(np.dot(residuals, residuals))
    r_squared = 1 - sse / sst
    try:
        area = frontier_area(floor, amplitude, exponent, reference)
    except OverflowError:
        raise ValueError("Fitted frontier overflows over the scoring interval") from None
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Fitted frontier must be finite and positive")
    log_k = math.log(amplitude) + exponent * math.log(reference) if amplitude > 0 else None
    valid = r_squared > 0 and amplitude > 0 and exponent > 0
    performance = area / r_squared if valid else None
    if performance is not None and not math.isfinite(performance):
        raise ValueError("Performance is not finite")
    return {
        "status": "local_reproduction" if valid else "undefined_nonpositive_r_squared_or_nondecaying_fit",
        "track": document["track"],
        "architecture_family_id": points[0].get("architecture_family_id"),
        "family_declaration_check": "matched_declared_ids" if points[0].get("architecture_family_id") else "not_supplied",
        "lower_is_better": True,
        "fit": {
            "model": "E + k / C**p = E + a * (C / C_ref)**(-p)",
            "objective": "sum((ln(fitted_bpb_i) - ln(measured_bpb_i))**2), equal weights",
            "parameter_bounds_assumption": "E >= 0, k > 0, p > 0; pending official bounds/code",
            "method": method, "E": floor, "a": amplitude, "p": exponent, "C_ref": reference,
            "k": math.exp(log_k) if log_k is not None and -700 < log_k < 700 else None,
            "ln_k": log_k,
            "log_space_sse": sse, "log_space_sst": sst,
        },
        "R_squared_log_bpb": r_squared,
        "integration_interval": {"C_min": C_MIN, "C_max": C_MAX},
        "A_front_bpb": area,
        "Performance": performance,
        "points": [dict(point, fitted_bpb=float(prediction), log_residual=float(residual))
                   for point, prediction, residual in zip(points, fitted, residuals)],
        "limitations": [
            "Local analysis, not an official score; parameter bounds were not shown in the supplied rule image.",
            "Use reviewed actual FLOPs and measured BPB for one held-out dataset and one track.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        result = score(json.loads(args.points.read_text()))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
