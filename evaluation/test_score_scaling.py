"""Independent synthetic-curve and quadrature checks for the supplied rules."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_scaling import C_MAX, C_MIN, TIERS, frontier_area, score


def observations(floor=0.75, amplitude=1.2, exponent=0.25):
    compute = (3e17, 1e18, 3e18, 1e19)
    return {"track": "A", "points": [
        {"tier": tier, "actual_training_flops": value,
         "bpb": floor + amplitude * (value / C_MIN) ** (-exponent)}
        for tier, value in zip(TIERS, compute)
    ]}


class ScalingScoreTests(unittest.TestCase):
    def test_known_curve_recovers_parameters_and_score(self):
        result = score(observations())
        self.assertAlmostEqual(result["fit"]["E"], 0.75, places=7)
        self.assertAlmostEqual(result["fit"]["a"], 1.2, places=7)
        self.assertAlmostEqual(result["fit"]["p"], 0.25, places=7)
        self.assertAlmostEqual(result["R_squared_log_bpb"], 1, places=12)
        expected = 0.75 + 1.2 * (1 - (C_MAX / C_MIN)**(-0.25)) / (0.25 * math.log(C_MAX / C_MIN))
        self.assertAlmostEqual(result["Performance"], expected, places=10)

    def test_area_matches_independent_log_compute_quadrature(self):
        for exponent in (0.0, 1e-10, 0.2, 2.0):
            width = math.log(C_MAX / C_MIN)
            expected = quad(lambda t: 0.7 + 1.4 * math.exp(-exponent * t), 0, width)[0] / width
            self.assertAlmostEqual(frontier_area(0.7, 1.4, exponent, C_MIN), expected, places=12)

    def test_r_squared_and_penalty_use_log_bpb(self):
        points = observations()
        points["points"][1]["bpb"] *= 1.02
        result = score(points)
        measured = np.log([point["bpb"] for point in result["points"]])
        fitted = np.log([point["fitted_bpb"] for point in result["points"]])
        expected = 1 - np.sum((measured - fitted)**2) / np.sum((measured - measured.mean())**2)
        self.assertAlmostEqual(result["R_squared_log_bpb"], expected, places=12)
        self.assertGreater(result["Performance"], result["A_front_bpb"])
        self.assertAlmostEqual(result["Performance"], result["A_front_bpb"] / expected, places=12)

    def test_actual_compute_and_zero_floor_boundary(self):
        document = observations(floor=0, exponent=0.4)
        for point in document["points"]:
            point["actual_training_flops"] *= 0.995
            point["bpb"] = 1.2 * (point["actual_training_flops"] / C_MIN) ** -0.4
        result = score(document)
        self.assertLess(result["fit"]["E"], 1e-7)
        self.assertAlmostEqual(result["fit"]["p"], 0.4, places=7)
        expected = frontier_area(0, 1.2, 0.4, C_MIN)
        self.assertAlmostEqual(result["Performance"], expected, places=9)

    def test_different_architecture_families_cannot_be_combined(self):
        document = observations()
        for point in document["points"]:
            point["architecture_family_id"] = "a" * 64
        result = score(document)
        self.assertEqual(result["family_declaration_check"], "matched_declared_ids")
        document["points"][-1]["architecture_family_id"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "same declared architecture family"):
            score(document)

    def test_rejects_flat_bpb_mixed_tracks_and_missing_actual_flops(self):
        flat = observations()
        for point in flat["points"]:
            point["bpb"] = 1.2
        mixed = observations()
        mixed["points"][0]["track"] = "B"
        planned = observations()
        planned["points"][0]["planned_training_flops"] = planned["points"][0].pop("actual_training_flops")
        duplicate = observations()
        duplicate["points"][1]["tier"] = "3e17"
        for document in (flat, mixed, planned, duplicate):
            with self.assertRaises(ValueError):
                score(document)


if __name__ == "__main__":
    unittest.main()
