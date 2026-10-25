import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_initialization import repair_zero_gdn_decay_rates


class GatedDeltaNet(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.A_log = torch.nn.Parameter(torch.tensor(values, dtype=torch.bfloat16))


class InitializationTest(unittest.TestCase):
    def test_zero_rate_stays_finite_after_weight_decay(self):
        model = GatedDeltaNet([2.71875, -math.inf, 0.921875])
        original = model.A_log.detach().clone()
        rng = torch.get_rng_state().clone()
        repairs = repair_zero_gdn_decay_rates(model)
        self.assertEqual(len(repairs), 1)
        self.assertTrue(torch.equal(model.A_log[[0, 2]], original[[0, 2]]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(repair_zero_gdn_decay_rates(model), [])
        with torch.no_grad():
            original.add_(original, alpha=-0.0003 * 0.1)
            model.A_log.add_(model.A_log, alpha=-0.0003 * 0.1)
        self.assertTrue(torch.isnan(original[1]).item())
        self.assertTrue(torch.isfinite(model.A_log).all().item())
        self.assertTrue((model.A_log.float().exp() > 0).all().item())

    def test_finite_initialization_is_unchanged(self):
        model = GatedDeltaNet([2.71875, 2.265625, 0.921875])
        before = model.A_log.detach().clone()
        self.assertEqual(repair_zero_gdn_decay_rates(model), [])
        self.assertTrue(torch.equal(model.A_log, before))

    def test_other_invalid_values_are_rejected(self):
        for value in [math.nan, math.inf]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'Unexpected nonfinite'):
                    repair_zero_gdn_decay_rates(GatedDeltaNet([value]))


if __name__ == '__main__':
    unittest.main()
