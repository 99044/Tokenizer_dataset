import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizer_factory import AdamWDtypeBuckets, build_optimizer


class FusedAdamWTasks(torch.optim.Optimizer):
    """Exercise the fused operator that Dion calls for each AdamW task."""

    def __init__(self, parameters):
        super().__init__(parameters, dict(lr=0.003))

    def _create_adamw_tasks(self, param_groups):
        for group in param_groups:
            parameters = group['params']
            for parameter in parameters:
                state = self.state[parameter]
                if not state:
                    state.update(momentum=torch.zeros_like(parameter),
                                 variance=torch.zeros_like(parameter),
                                 step=torch.tensor(0.0))
                state['step'].add_(1)
            torch._fused_adamw_(
                parameters, [p.grad for p in parameters],
                [self.state[p]['momentum'] for p in parameters],
                [self.state[p]['variance'] for p in parameters], [],
                [self.state[p]['step'] for p in parameters],
                lr=group['lr'], beta1=0.9, beta2=0.95, weight_decay=0.1,
                eps=1e-8, amsgrad=False, maximize=False,
            )
            yield None

    @torch.no_grad()
    def step(self):
        list(self._create_adamw_tasks(self.param_groups))


class CompatibleAdamWTasks(AdamWDtypeBuckets, FusedAdamWTasks):
    pass


class DtypeBucketTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('dion'), 'requires the pinned Dion environment')
    def test_real_muon_factory_steps_and_restores_mixed_dtype_state(self):
        model = torch.nn.Module()
        model.embeddings = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        model.matrix = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        model.norm = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        settings = dict(optimizer=dict(
            name='muon', learning_rate=.003, scalar_learning_rate=.000265,
            weight_decay=.1, betas=[.9, .95], eps=1e-8,
            adjust_lr='rms_norm', use_triton=False,
        ))
        optimizer, report = build_optimizer(model, settings)
        for step in range(5):
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, .25 + step * .1)
            optimizer.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertEqual(len(optimizer.param_groups), len(report['groups']))
            if step == 2:
                state = copy.deepcopy(optimizer.state_dict())
                optimizer, _ = build_optimizer(model, settings)
                optimizer.load_state_dict(state)
        self.assertEqual(model.embeddings.weight.dtype, torch.bfloat16)
        self.assertEqual(model.norm.dtype, torch.float32)

    def test_mixed_dtype_fused_update_matches_separate_reference_and_resume(self):
        parameters = [torch.nn.Parameter(torch.ones(4, dtype=dtype))
                      for dtype in (torch.bfloat16, torch.float32)]
        expected = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        optimizer = CompatibleAdamWTasks(parameters)
        reference = FusedAdamWTasks([dict(params=[p]) for p in expected])
        for step in range(5):
            for p in parameters + expected:
                p.grad = torch.full_like(p, 0.2 + step * 0.1)
            optimizer.step()
            reference.step()
            for actual, target in zip(parameters, expected):
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
            self.assertEqual(len(optimizer.param_groups), 1)
            if step == 2:
                state = copy.deepcopy(optimizer.state_dict())
                optimizer = CompatibleAdamWTasks(parameters)
                optimizer.load_state_dict(state)
        self.assertEqual([p.dtype for p in parameters],
                         [torch.bfloat16, torch.float32])


if __name__ == '__main__':
    unittest.main()
