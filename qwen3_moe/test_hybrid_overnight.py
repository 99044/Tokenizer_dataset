"""Regression coverage for mixed layer accounting and the overnight queue."""

import json
from pathlib import Path
import sys
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen3_moe import hybrid_overnight
from qwen3_moe.plan import HYBRID_VARIANTS, make_recipe
from qwen3_flops import qwen3_moe_parameters
from qwen3_moe_flops import cost_breakdown
from qwen3_moe_layers import qwen3_moe_layer_pattern


class MixedMoETests(unittest.TestCase):
    def test_plans_match_actual_model_parameters_and_reference_flops(self):
        import torch
        from transformers import AutoModelForCausalLM, Qwen3MoeConfig
        from eval import active_parameter_count

        _, _, baseline = make_recipe("3e17")
        full_ffn = baseline["reference_8k"]["parameters_by_component"]["swiglu_projections"]
        for name, (dense, experts) in HYBRID_VARIANTS.items():
            with self.subTest(name=name):
                config, settings, budget = make_recipe("3e17", hybrid_variant=name)
                pattern = qwen3_moe_layer_pattern(config)
                self.assertEqual([i for i, sparse in enumerate(pattern) if not sparse], list(dense))
                self.assertEqual(config["num_experts"], experts)
                self.assertEqual(config["moe_intermediate_size"] * config["num_experts_per_tok"], 1024)
                counts = qwen3_moe_parameters(config)
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(Qwen3MoeConfig.from_dict(config))
                self.assertEqual(sum(p.numel() for p in model.parameters()), counts["total_parameters"])
                self.assertEqual(active_parameter_count(model), counts["active_non_vocabulary_parameters"])
                self.assertGreaterEqual(budget["active_non_vocabulary_fraction"], 0.10)
                if experts > 24:
                    self.assertEqual(counts["parameters_by_component"]["swiglu_projections"], full_ffn)
                for length in (2048, 4096, 8192):
                    report = cost_breakdown(config, length, 8192 // length)
                    parts = report["megatron_components"]
                    self.assertEqual(sum(parts.values()), report["megatron_reference_total"])
                    self.assertEqual(parts["dense_ffn_projections"], 18 * 8192 * len(dense) * 256 * 1024)
                    self.assertEqual(report["router_projection_gemm"], 6 * 8192 * sum(pattern) * 256 * experts)
                self.assertLessEqual(budget["estimated_training_flops"], 3 * 10**17)
                self.assertEqual(settings["submission"]["data_parallel_size"], 8)

    def test_mixed_decoder_returns_router_logits_only_for_sparse_layers(self):
        import torch
        import torch.nn.functional as F
        from qwen3_moe_training import AccumulatedRouterBalance, Qwen3MoeTrainingConfig, router_z_loss

        config, _, _ = make_recipe("3e17", hybrid_variant="04-alternate-e24")
        config.update(hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8, vocab_size=32, intermediate_size=64,
                      mlp_only_layers=[0, 2], moe_intermediate_size=32, num_experts=4)
        device = torch.device("cpu")
        model = Qwen3MoeTrainingConfig(config).build()
        model.init_weights(8192, device)
        inputs = torch.randint(4, 32, (1, 8))
        labels = inputs.clone()
        labels[:, 0] = -100
        logits, routers = model(inputs, route=True)
        self.assertEqual(len(routers), 2)
        balance = AccumulatedRouterBalance(model.lm.config, device)
        ce = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
        objective = ce + 0.001 * balance(routers) + 0.001 * router_z_loss(routers)
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertEqual(balance.counts.sum(-1).tolist(), [16, 16])
        self.assertTrue(torch.isfinite(model.lm.model.layers[0].mlp.gate_proj.weight.grad).all())
        self.assertTrue(torch.isfinite(model.lm.model.layers[1].mlp.gate.weight.grad).all())


class QueueTests(unittest.TestCase):
    def test_all_training_finishes_before_eval_and_failures_do_not_stop_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plans = hybrid_overnight.make_plans(output, 15 * 10**16)
            names = list(plans)
            training_order = []
            eval_start = Barrier(7)

            def train(_, directory, settings, env, train_seconds, grace):
                training_order.append(directory.name)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                self.assertEqual((train_seconds, grace), (3600, 600))
                if directory.name == names[1]:
                    raise RuntimeError("simulated train failure")
                return directory / "step-1.pt", dict(
                    status="passed_time_limited_prefix_audit", completed_optimizer_steps=1,
                    actual_training_flops=1, training_tokens=262144,
                    final_context_length=2048, elapsed_seconds=3600,
                )

            def evaluate(_, directory, checkpoint, device, env, batch_size, timeout):
                self.assertEqual(training_order, names)
                self.assertEqual(device, str(names.index(directory.name)))
                self.assertEqual((batch_size, timeout), (64, 1800))
                eval_start.wait(timeout=10)
                if directory.name == names[3]:
                    raise RuntimeError("simulated eval failure")
                return dict(bpb=1.2, evaltime=1, forward_seconds=2, active_parameters=3, scored_tokens=4)

            with patch.object(hybrid_overnight, "train_variant", side_effect=train), \
                 patch.object(hybrid_overnight, "evaluate_variant", side_effect=evaluate):
                result = hybrid_overnight.run_queue(
                    None, output, plans, list("01234567"), 3600, 600, 64, 1800,
                )
            self.assertEqual(result, 1)
            report = json.loads((output / "summary.json").read_text())["variants"]
            self.assertEqual([entry["variant"] for entry in report], names)
            self.assertEqual(report[1]["train_status"], "failed")
            self.assertEqual(report[1]["eval_status"], "skipped")
            self.assertEqual(report[3]["eval_status"], "failed")
            self.assertEqual(sum(row["eval_status"] == "passed" for row in report), 6)
            self.assertEqual(report[-1]["train_status"], "time_limit")


if __name__ == "__main__":
    unittest.main()
