import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdn_stability import GDNStabilityGuard


class GDNStabilityTests(unittest.TestCase):
    def test_fp32_recurrence_preserves_bf16_interface_and_gradient(self):
        dtypes = []

        def operator(q, k, v, g, beta, **kwargs):
            dtypes.append((q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype))
            return q * k + v + g + beta, None

        with tempfile.TemporaryDirectory() as directory:
            guard = GDNStabilityGuard(operator, directory, rank=0)
            values = [torch.tensor([0.25], dtype=torch.bfloat16, requires_grad=True)
                      for _ in range(5)]
            normal, _ = guard(*values)
            self.assertEqual(normal.dtype, torch.bfloat16)
            self.assertEqual(dtypes[-1], (torch.bfloat16,) * 5)
            guard.force_fp32 = True
            recomputed, _ = guard(*values)
            self.assertEqual(recomputed.dtype, torch.bfloat16)
            self.assertEqual(dtypes[-1], (torch.float32,) * 5)
            recomputed.sum().backward()
            self.assertTrue(all(torch.isfinite(value.grad).all() for value in values))

    def test_nonfinite_bf16_forward_recomputes_in_fp32(self):
        def operator(q, k, v, g, beta, **kwargs):
            if q.dtype == torch.bfloat16:
                return torch.full_like(q, float('nan')), None
            return q + k + v + g + beta, None

        with tempfile.TemporaryDirectory() as directory:
            guard = GDNStabilityGuard(operator, directory, rank=0)
            values = [torch.tensor([0.25], dtype=torch.bfloat16, requires_grad=True)
                      for _ in range(5)]
            result, _ = guard(*values)
            result.sum().backward()
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(all(torch.isfinite(value.grad).all() for value in values))
            self.assertEqual(guard.fallback_count, 1)
            self.assertTrue((Path(directory) / 'numerical-fallbacks-rank-0.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
