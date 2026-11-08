"""CPU-only precision, checkpoint and weight-selection tests; no LM training."""

import copy
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp32_master import FP32MasterWeights
from optimizer_factory import build_optimizer
from weight_ema import WeightEMA, resolve_ema_config, select_ema_weights


class TinyTiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.output = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.output.weight = self.embeddings.weight
        self.projection = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.norm = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.register_buffer('counter', torch.tensor(7))


def build(master=True):
    model = TinyTiedModel()
    weights = FP32MasterWeights(model) if master else None
    settings = {'optimizer': dict(name='soap', learning_rate=0.0003, betas=[0.95, 0.95],
                                 eps=1e-8, weight_decay=0.1, precondition_frequency=2,
                                 max_precond_dim=16)}
    optimizer, report = build_optimizer(model, settings,
                                       parameter_map=weights.parameter_map if weights else None)
    if weights:
        weights.install(optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 100)
    specification = resolve_ema_config(dict(enabled=True, start_fraction=0.2,
                                            half_life_tokens=[200, 800]), 10, 100)
    return model, weights, optimizer, scheduler, WeightEMA(model, specification, weights), report


def step(bundle, step_number):
    model, _, optimizer, scheduler, ema, _ = bundle
    optimizer.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(1000 + step_number)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator).to(parameter.dtype)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema.update(step_number)
    scheduler.step()


class PrecisionTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_unique_masters_and_tied_weights(self):
        model, masters, optimizer, _, _, report = build()
        self.assertEqual(len(masters.named), len(list(model.parameters())))
        self.assertIs(model.output.weight, model.embeddings.weight)
        self.assertEqual(report['total_parameters'], sum(p.numel() for p in model.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 for g in optimizer.param_groups for p in g['params']))
        with self.assertRaises(ValueError):
            FP32MasterWeights(model).install(torch.optim.SGD(model.parameters(), lr=0.1))

    def test_master_accumulates_updates_below_bf16_rounding(self):
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.fill_(1)
        masters = FP32MasterWeights(model)
        optimizer = torch.optim.SGD(masters.parameter_map.values(), lr=0.0001)
        masters.install(optimizer)
        for _ in range(100):
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertLess(model.weight.item(), 0.995)
        self.assertAlmostEqual(masters.named[0][2].item(), 0.99, places=4)
        self.assertIsNone(model.weight.grad)

    def test_soap_fp32_update_matches_float_reference(self):
        model, masters, optimizer, _, _, _ = build()
        from third_party.soap import SOAP
        refs = [torch.nn.Parameter(master.detach().clone()) for _, _, master in masters.named]
        reference = SOAP(refs, lr=0.0003, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.1,
                         precondition_frequency=2, max_precond_dim=16)
        for _ in range(5):
            for (_, parameter, _), ref in zip(masters.named, refs):
                parameter.grad = torch.randn_like(parameter)
                ref.grad = parameter.grad.float().clone()
            optimizer.step()
            reference.step()
            for (_, parameter, master), ref in zip(masters.named, refs):
                self.assertTrue(torch.equal(master, ref))
                self.assertTrue(torch.equal(parameter, master.to(parameter.dtype)))

    def test_exact_serialized_resume_including_ema(self):
        torch.manual_seed(1)
        original = build()
        for number in range(1, 6):
            step(original, number)
        model, masters, optimizer, scheduler, ema, _ = original
        checkpoint = dict(step=5, model=model.state_dict(), fp32_master=masters.state_dict(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), ema=ema.state_dict())
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        resumed = build()
        resumed[0].load_state_dict(checkpoint['model'])
        resumed[2].load_state_dict(checkpoint['optimizer'])
        resumed[1].load_state_dict(checkpoint['fp32_master'])
        resumed[3].load_state_dict(checkpoint['scheduler'])
        resumed[4].load_state_dict(checkpoint['ema'], 5)
        for number in range(6, 11):
            step(original, number)
            step(resumed, number)
        for i in (0, 1, 2, 3, 4):
            self.assert_nested_equal(original[i].state_dict(), resumed[i].state_dict())

    def test_ema_initialization_and_export_preserve_aliases_dtypes_and_raw(self):
        bundle = build()
        model, masters, _, _, ema, _ = bundle
        for number in range(1, 4):
            step(bundle, number)
        for values in ema.shadows.values():
            for name, _, master in masters.named:
                self.assertTrue(torch.equal(values[name], master))
        step(bundle, 4)
        checkpoint = dict(step=4, model=copy.deepcopy(model.state_dict()), ema=ema.state_dict())
        before = copy.deepcopy(checkpoint)
        selected = select_ema_weights(checkpoint, 'ema_200')
        self.assertEqual(selected['embeddings.weight'].data_ptr(), selected['output.weight'].data_ptr())
        self.assertEqual(selected['embeddings.weight'].dtype, torch.bfloat16)
        self.assertEqual(selected['norm'].dtype, torch.float32)
        self.assertEqual(selected['counter'].item(), 7)
        self.assert_nested_equal(before, checkpoint)
        with self.assertRaises(ValueError):
            select_ema_weights(checkpoint, 'not_a_variant')
        checkpoint['step'] = 5
        with self.assertRaises(ValueError):
            select_ema_weights(checkpoint, 'ema_200')

    def test_token_half_life_and_ema_does_not_mutate_training(self):
        torch.manual_seed(99)
        original = build()
        torch.manual_seed(99)
        control = build()
        spec = resolve_ema_config(dict(enabled=True, start_fraction=0, half_life_tokens=[200]), 10, 100)
        ema = WeightEMA(original[0], spec, original[1])
        ema.update(1)
        expected = {name: p.clone() for name, p in ema.shadows['ema_200'].items()}
        with torch.no_grad():
            for p in ema.sources.values():
                p.add_(1)
        ema.update(2)
        ema.update(3)
        for name, p in ema.shadows['ema_200'].items():
            torch.testing.assert_close(p, expected[name] + 0.5)
        # Separate identical SOAP paths, one also maintaining EMA.
        torch.manual_seed(99)
        original = build()
        for number in range(1, 8):
            step(original, number)
            model, _, optimizer, scheduler, _, _ = control
            generator = torch.Generator().manual_seed(1000 + number)
            optimizer.zero_grad(set_to_none=True)
            for p in model.parameters():
                p.grad = torch.randn(p.shape, generator=generator).to(p.dtype)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        self.assert_nested_equal(original[0].state_dict(), control[0].state_dict())
        self.assert_nested_equal(original[1].state_dict(), control[1].state_dict())

    def test_resume_before_start_and_reject_incompatible_state(self):
        bundle = build()
        step(bundle, 1)
        ema = bundle[4]
        resumed = build()[4]
        resumed.load_state_dict(ema.state_dict(), 1)
        self.assertEqual(resumed.shadows, {})
        with self.assertRaises(ValueError):
            resumed.load_state_dict(ema.state_dict(), 2)
        with self.assertRaises(ValueError):
            resumed.update(3)
        for bad in (-0.1, 1, float('nan')):
            with self.assertRaises(ValueError):
                resolve_ema_config(dict(enabled=True, start_fraction=bad), 10, 100)


if __name__ == '__main__':
    unittest.main()
