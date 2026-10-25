import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_party.soap import SOAP


class SOAPResumeTests(unittest.TestCase):
    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_bf16_resume_preserves_exact_fp32_moments_and_eigenbases(self):
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.randn(4, 5, dtype=torch.bfloat16))
        optimizer = SOAP([parameter], lr=0.003, max_precond_dim=16)
        for _ in range(3):
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        saved_before = copy.deepcopy(saved)
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = SOAP([restored_parameter], lr=0.1, max_precond_dim=16)
        restored.load_state_dict(saved)
        self.assert_state_equal(optimizer.state[parameter], restored.state[restored_parameter])
        self.assertEqual(restored.param_groups[0]['lr'], 0.003)
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient.clone()
        restored_parameter.grad = gradient.clone()
        optimizer.step()
        restored.step()
        self.assertTrue(torch.equal(parameter, restored_parameter))
        self.assert_state_equal(optimizer.state[parameter], restored.state[restored_parameter])
        # Loading/updating the restored optimizer must not mutate the checkpoint.
        self.assert_state_equal(saved_before, saved)


if __name__ == '__main__':
    unittest.main()
