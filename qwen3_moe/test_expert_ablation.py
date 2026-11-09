import json
from pathlib import Path
import tempfile
import unittest

import yaml

from expert_ablation import BASELINE, ROUTER_METRIC, VARIANTS, compare, make_expert_recipe


class ExpertAblationTests(unittest.TestCase):
    def test_variants_keep_the_controlled_recipe_and_fit_budget(self):
        reference, settings, budget = make_expert_recipe(BASELINE)
        baseline_yaml = Path(__file__).resolve().parent / "configs/3e17/training.yaml"
        self.assertEqual(settings, yaml.safe_load(baseline_yaml.read_text()))
        for variant in VARIANTS:
            model, candidate, planned = make_expert_recipe(variant)
            self.assertEqual(model["moe_intermediate_size"] * model["num_experts_per_tok"], 1024)
            self.assertEqual(candidate["optimizer"], settings["optimizer"])
            self.assertEqual(candidate["training"]["adaptive_token_loss"], settings["training"]["adaptive_token_loss"])
            self.assertEqual(model["router_aux_loss_coef"], reference["router_aux_loss_coef"])
            self.assertEqual(planned["tokens_per_update"], budget["tokens_per_update"])
            self.assertGreaterEqual(planned["active_non_vocabulary_fraction"], 0.10)
            self.assertLessEqual(planned["estimated_training_flops"], 3 * 10**17)
        fine, _, _ = make_expert_recipe("e48-k4-w256")
        self.assertEqual(fine["num_experts"] * fine["moe_intermediate_size"],
                         reference["num_experts"] * reference["moe_intermediate_size"])

    def write_run(self, root, variant, bpb, batch=64):
        path = root / variant
        (path / "train").mkdir(parents=True)
        config, settings, budget = make_expert_recipe(variant)
        def write(filename, value):
            (path / filename).write_text(json.dumps(value), encoding="utf-8")

        write("train/run-plan.json", {"settings": settings, "data": {"tokenizer_sha256": "same"}})
        write("flops-report.json", {
            "status": "passed_log_path_audit", "actual_training_flops": budget["estimated_training_flops"],
            "training_tokens": budget["planned_training_tokens"], "completed_optimizer_steps": budget["optimizer_updates"],
        })
        write("eval-result.json", {
            "bpb": bpb, "forward_seconds": 40, "evaltime": 50, "batch_size_per_gpu": batch,
            "sequence_length": 8192, "distributed_world_size": 1, "scored_tokens": 1000,
            "active_parameters": budget["active_non_vocabulary_parameters"],
            "fusion": {"mode": "rmsnorm"},
            "acceleration": {"experts": "grouped_gemm", "attention": "flash_attention_3", "compile_swiglu": True,
                             "torch_version": "2.6.0", "cuda_version": "12.4"},
            "data_source": {"kind": "packed", "path": "/cache", "total_bytes": 3000,
                            "identity": {"cache_tokenizer_sha256": "same"}},
        })
        final = budget["optimizer_updates"]
        with (path / "train/metrics.jsonl").open("w", encoding="utf-8") as out:
            for step in range(final - 199, final + 1):
                row = {"step": step, ROUTER_METRIC: [1.5] * config["num_hidden_layers"]}
                out.write(json.dumps(row) + "\n")
        return path

    def test_compare_reports_router_load_and_rejects_different_eval_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.write_run(root, BASELINE, 1.15)
            fine = self.write_run(root, "e48-k4-w256", 1.14)
            capacity = self.write_run(root, "e20-k2-w512", 1.16, batch=32)
            result = compare(baseline, [fine])["candidates"][0]
            self.assertAlmostEqual(result["delta_bpb"], -0.01)
            self.assertEqual(result["router_layer_max_to_mean"][-1], {"layer": 8, "mean": 1.5, "peak": 1.5})
            with self.assertRaisesRegex(ValueError, "batch 64"):
                compare(baseline, [capacity])


if __name__ == "__main__":
    unittest.main()
