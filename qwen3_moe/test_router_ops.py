"""Compare routing arithmetic to actual pinned-HF eager forward/backward ops."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/src"))

try:
    PINNED_TRANSFORMERS = version("transformers") == "5.18.0"
except PackageNotFoundError:
    PINNED_TRANSFORMERS = False


@unittest.skipUnless(PINNED_TRANSFORMERS, "requires the pinned Transformers 5.18.0 environment")
class RouterGraphTests(unittest.TestCase):
    def test_router_and_auxiliary_graph_match_actual_eager_arithmetic(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
        from qwen3_moe_training import AccumulatedRouterBalance, router_z_loss
        from qwen3_moe_flops import routing_ledger
        from graph_accounting import operation_flops

        class Counter(TorchDispatchMode):
            def __init__(self):
                self.phase = "forward"
                self.totals = dict(forward=0, backward=0)

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if not isinstance(out, torch.Tensor) or not out.is_floating_point():
                    return out
                name = str(func)
                cost = 0
                if name == "aten.mm.default":
                    cost = 2 * args[0].shape[0] * args[0].shape[1] * args[1].shape[1]
                elif name in {"aten._softmax.default", "aten._softmax_backward_data.default"}:
                    cost = 4 * out.numel() - out.numel() // out.shape[-1]
                elif name == "aten.logsumexp.default":
                    cost = 3 * args[0].numel() + out.numel()
                elif name == "aten.pow.Tensor_Scalar":
                    if args[1] not in {1, 2}:
                        raise AssertionError(f"Unexpected power {args[1]}")
                    cost = out.numel() if args[1] == 2 else 0
                elif name.startswith(("aten.mul.", "aten.div.", "aten.div_.", "aten.add.", "aten.add_.", "aten.neg.", "aten.sub.", "aten.exp.")):
                    cost = out.numel()
                elif name.startswith("aten.sum."):
                    cost = args[0].numel() - out.numel()
                elif name.startswith("aten.mean."):
                    cost = args[0].numel()  # reduction plus one divide per result
                elif name == "aten.index_add_.default":
                    cost = args[3].numel()
                elif not name.startswith((
                    "aten.view.", "aten.t.", "aten.detach.", "aten.clone.", "aten.zeros.",
                    "aten.zeros_like.", "aten.ones.", "aten.slice.", "aten.slice_backward.",
                    "aten.unsqueeze.", "aten.squeeze.", "aten.expand.", "aten.select.",
                    "aten._to_copy.", "aten.stack.", "aten.index_select.", "aten.scatter.",
                )):
                    raise AssertionError(f"Unaccounted floating-point operator: {name}")
                self.totals[self.phase] += cost
                return out

        torch.set_num_threads(2)
        for tokens, hidden, experts, top_k, layers, use_z, weighted in (
            (3, 8, 12, 2, 2, False, False), (3, 8, 12, 2, 2, True, False),
            (3, 8, 12, 2, 2, True, True), (5, 16, 20, 4, 1, True, True),
        ):
            config = Qwen3MoeConfig(hidden_size=hidden, num_experts=experts,
                                    num_experts_per_tok=top_k, num_hidden_layers=layers, norm_topk_prob=True)
            routers = [Qwen3MoeTopKRouter(config) for _ in range(layers)]
            inputs = [torch.randn(tokens, hidden, requires_grad=True) for _ in range(layers)]
            values = [torch.randn(tokens, top_k, hidden, requires_grad=True) for _ in range(layers)]
            gradients = [torch.randn(tokens, hidden) for _ in range(layers)]
            ids = torch.arange(tokens).repeat_interleave(top_k)
            counter = Counter()
            with counter:
                outputs, all_logits = [], []
                for router, inputs_i, values_i in zip(routers, inputs, values):
                    logits, weights, _ = router(inputs_i)
                    all_logits.append(logits)
                    output = torch.zeros(tokens, hidden)
                    output.index_add_(0, ids, (values_i * weights[:, :, None]).reshape(-1, hidden))
                    outputs.append(output)
                auxiliary = AccumulatedRouterBalance(config, torch.device("cpu"))(all_logits)
                losses = [auxiliary]
                if use_z:
                    losses.append(router_z_loss(all_logits))
                if weighted:
                    # Two scalar coefficients and two additions to token CE.
                    # The actual token-CE derivative is supplied via outputs.
                    penalty = losses[0] * 0.001 + torch.zeros(())
                    if use_z:
                        penalty = penalty + losses[1] * 0.001
                    losses = [penalty]
                counter.phase = "backward"
                torch.autograd.backward(outputs + losses, gradients + [torch.ones(()) for _ in losses])
            declared = dict(num_hidden_layers=layers, hidden_size=hidden, num_experts=experts,
                            num_experts_per_tok=top_k, norm_topk_prob=True, router_z_loss_coef=0.001 if use_z else 0)
            ledger = routing_ledger(declared, tokens, include_input_scatter=False, include_loss_weight=weighted)
            for phase in ("forward", "backward"):
                expected = sum(operation_flops(op) for op in ledger if op["phase"] == phase)
                self.assertEqual(counter.totals[phase], expected, (tokens, experts, phase))


if __name__ == "__main__":
    unittest.main()
