"""CPU semantic/gradient checks; native Hopper kernels have a separate verifier."""

from copy import deepcopy
from importlib.metadata import version
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import torch
import torch.nn.functional as F

from h800_acceleration import grouped_experts, install_h800_acceleration, runtime_options
from qwen3_moe_training import router_z_loss


def reference_gmm(a, b, counts, trans_b=False):
    assert trans_b
    return torch.cat([F.linear(x, w) for x, w in zip(a.split(counts.tolist()), b)])


class RouterZTests(unittest.TestCase):
    def test_stable_value_and_analytic_gradient_over_layers_and_tokens(self):
        torch.manual_seed(3)
        for shift in (0, 10000, -10000):
            values = [((torch.randn(5, 8) + shift).requires_grad_()) for _ in range(2)]
            result = router_z_loss(values)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, torch.cat(values).logsumexp(-1).square().mean())
            result.backward()
            for logits in values:
                expected = 2 * logits.logsumexp(-1, keepdim=True) * logits.softmax(-1) / 10
                # Stable logsumexp backward is exp(x-logsumexp), with FP32
                # cancellation at large shifts; permit that rounding error.
                torch.testing.assert_close(logits.grad, expected, rtol=0.002, atol=1e-5)
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bf16_logits_use_fp32_and_empty_input_is_rejected(self):
        values = torch.randn(4, 5, dtype=torch.bfloat16, requires_grad=True)
        result = router_z_loss((values,))
        torch.testing.assert_close(result, values.float().logsumexp(-1).square().mean())
        result.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        with self.assertRaises(ValueError):
            router_z_loss(())


@unittest.skipUnless(version("transformers") == "5.18.0", "requires pinned Transformers")
class GroupedExpertTests(unittest.TestCase):
    def test_outputs_and_all_gradients_match_stock_including_unused_experts(self):
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

        torch.set_num_threads(2)
        for top_k in (2, 4, 8):
            torch.manual_seed(top_k)
            config = Qwen3MoeConfig(hidden_size=16, num_experts=12, num_experts_per_tok=top_k,
                                    moe_intermediate_size=8)
            stock = Qwen3MoeExperts(config).double()
            with torch.no_grad():
                for parameter in stock.parameters():
                    parameter.normal_(0, 0.1)
            fast = deepcopy(stock)
            hidden = torch.randn(11, 16, dtype=torch.float64, requires_grad=True)
            other_hidden = hidden.detach().clone().requires_grad_()
            ids = torch.stack([torch.randperm(9)[:top_k] for _ in range(11)])
            weights = torch.randn(11, top_k, dtype=torch.float64, requires_grad=True)
            other_weights = weights.detach().clone().requires_grad_()
            expected = stock(hidden, ids, weights)
            actual = grouped_experts(fast, other_hidden, ids, other_weights, reference_gmm)
            torch.testing.assert_close(actual, expected)
            gradient = torch.randn_like(actual)
            expected.backward(gradient)
            actual.backward(gradient)
            for got, ref in zip((other_hidden, other_weights, *fast.parameters()),
                                (hidden, weights, *stock.parameters())):
                torch.testing.assert_close(got.grad, ref.grad)
            self.assertEqual(torch.count_nonzero(fast.gate_up_proj.grad[9:]).item(), 0)
            self.assertEqual(stock.state_dict().keys(), fast.state_dict().keys())

    def test_cpu_auto_fallback_preserves_parameters_and_strict_request_fails(self):
        from transformers import AutoModelForCausalLM, Qwen3MoeConfig

        model = AutoModelForCausalLM.from_config(Qwen3MoeConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, vocab_size=32, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        ), attn_implementation="sdpa")
        before = {name: id(p) for name, p in model.named_parameters()}
        with patch.dict("os.environ", {}, clear=True):
            report = install_h800_acceleration(model)
            self.assertEqual(report["experts"], "eager")
            self.assertEqual(report["attention"], "sdpa")
            self.assertEqual(len(report["fallback_reasons"]), 2)
            # Exercise the same safe-load mode as train_ddp --resume, with the
            # actual runtime report nested in the saved training plan.
            checkpoint = dict(model=model.state_dict(), plan={"acceleration": report})
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, map_location="cpu", weights_only=True)
            self.assertEqual(type(loaded["plan"]["acceleration"]["torch_version"]), str)
            self.assertEqual(loaded["plan"]["acceleration"], report)
            for name, value in loaded["model"].items():
                torch.testing.assert_close(value, model.state_dict()[name])
            self.assertEqual(before, {name: id(p) for name, p in model.named_parameters()})
            with self.assertRaisesRegex(RuntimeError, "Requested grouped_gemm unavailable"):
                install_h800_acceleration(model, {"experts": "grouped_gemm"})

    def test_inference_norm_fusion_retains_cpu_forward_gradients_and_state(self):
        from transformers import AutoModelForCausalLM, Qwen3MoeConfig
        from fusion import install_operator_fusion

        model = AutoModelForCausalLM.from_config(Qwen3MoeConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, vocab_size=32, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        ), attn_implementation="sdpa")
        stock = deepcopy(model)
        info = install_operator_fusion(model, "rmsnorm")
        self.assertEqual(info["rms_norm"], 5)
        self.assertEqual(stock.state_dict().keys(), model.state_dict().keys())
        inputs = torch.randint(0, 32, (1, 8))
        expected, actual = stock(inputs).logits, model(inputs).logits
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        for got, ref in zip(model.parameters(), stock.parameters()):
            torch.testing.assert_close(got.grad, ref.grad)


class RuntimeOptionTests(unittest.TestCase):
    def test_overrides_and_invalid_values(self):
        with patch.dict("os.environ", {"QWEN_MOE_EXPERTS": "eager", "QWEN_MOE_COMPILE_SWIGLU": "0"}):
            options = runtime_options()
            self.assertEqual(options["experts"], "eager")
            self.assertFalse(options["compile_swiglu"])
        for options in ({"experts": "bad"}, {"attention": "fp8"}, {"compile_swiglu": "true"}):
            with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
                runtime_options(options)


if __name__ == "__main__":
    unittest.main()
