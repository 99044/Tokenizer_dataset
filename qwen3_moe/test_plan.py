"""Check budget safety, runtime schedules and reproducible recipe artifacts."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from qwen3_moe.plan import TIERS, main, make_recipe
from qwen3_moe_a30m.plan import make_plan, make_training_settings
from qwen3_flops import qwen3_moe_parameters, validate_moe_activation_ratio
from qwen3_moe_rules import maximum_expert_count, validate_family_settings
from context_curriculum import (
    curriculum_completed_flops, curriculum_lr_factor, curriculum_stage, validate_curriculum,
)


class RecipeTests(unittest.TestCase):
    def test_committed_recipes_fit_caps_and_runtime(self):
        for tier, preset in TIERS.items():
            with self.subTest(tier=tier):
                config, settings, budget = make_recipe(tier)
                folder = ROOT / "configs" / tier
                self.assertEqual(config, json.loads((folder / "config.json").read_text()))
                self.assertEqual(settings, yaml.safe_load((folder / "training.yaml").read_text()))
                self.assertEqual(budget, json.loads((folder / "budget.json").read_text()))
                training = settings["training"]
                activation = validate_family_settings(settings)
                self.assertGreaterEqual(10 * activation["active_non_vocabulary_parameters"],
                                        activation["total_non_vocabulary_parameters"])
                self.assertEqual(budget["active_non_vocabulary_fraction"], activation["active_non_vocabulary_fraction"])
                validate_curriculum(settings, 8, 1, preset["accumulation"], training["max_steps"])
                spent = curriculum_completed_flops(settings, training["max_steps"])
                self.assertLessEqual(spent, preset["flops"])
                self.assertGreater(spent, preset["flops"] * 0.99)
                stages = training["length_curriculum"]["stages"]
                self.assertGreater(spent + stages[-1]["flops_per_update"], preset["flops"])
                self.assertEqual(sum(s["steps"] for s in stages), training["max_steps"])
                for stage in stages:
                    for step in (stage["first_step"], stage["last_step"]):
                        self.assertEqual(curriculum_stage(settings, step), stage)
                        self.assertTrue(0 <= curriculum_lr_factor(settings, step - 1) <= 1)
                self.assertAlmostEqual(curriculum_lr_factor(settings, training["max_steps"]), 0.01)
                self.assertGreater(budget["training_tokens_per_active_parameter"], 20)

    def test_granularity_candidates_preserve_active_ffn_compute(self):
        for tier in TIERS:
            configs = [make_recipe(tier, granularity_shift=shift)[0] for shift in (-1, 0, 1)]
            self.assertEqual(len({c["moe_intermediate_size"] * c["num_experts_per_tok"] for c in configs}), 1)
            for config in configs:
                validate_moe_activation_ratio(config)

    def test_larger_batches_keep_microbatch_and_recompute_budget_and_runtime(self):
        for tier, preset in TIERS.items():
            old_config, old_settings, old_budget = make_recipe(tier)
            for tokens in (2 * old_budget["tokens_per_update"], 4 * old_budget["tokens_per_update"]):
                with self.subTest(tier=tier, tokens=tokens):
                    config, settings, budget = make_recipe(tier, tokens_per_update=tokens)
                    training = settings["training"]
                    accumulation = tokens // (8 * 8192)
                    self.assertEqual(config, old_config)
                    self.assertEqual(settings["architecture_family"], old_settings["architecture_family"])
                    self.assertEqual(settings["optimizer"], old_settings["optimizer"])
                    self.assertEqual(training["gradient_accumulation_steps"], accumulation)
                    self.assertEqual(budget["tokens_per_update"], tokens)
                    self.assertLess(training["max_steps"], old_budget["optimizer_updates"])
                    self.assertEqual(budget["planned_training_tokens"], tokens * training["max_steps"])
                    validate_curriculum(settings, 8, 1, accumulation, training["max_steps"])
                    validate_family_settings(settings)
                    self.assertLessEqual(budget["estimated_training_flops"], preset["flops"])
                    self.assertGreaterEqual(100 * budget["estimated_training_flops"], 99 * preset["flops"])
                    for stage, old in zip(training["length_curriculum"]["stages"],
                                          old_settings["training"]["length_curriculum"]["stages"]):
                        self.assertEqual(stage["micro_batch_size"], old["micro_batch_size"])
                        self.assertEqual(stage["grad_accum_steps"], accumulation)
                        self.assertEqual(stage["flops_per_update"] * old_budget["tokens_per_update"],
                                         old["flops_per_update"] * tokens)
                    self.assertAlmostEqual(curriculum_lr_factor(settings, training["max_steps"]), 0.01)

    def test_batch_cli_writes_a_directly_usable_tier(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as output:
            main(["--tier", "3e17", "--tokens-per-update", "1048576", "--output-dir", temp])
            summary = json.loads(output.getvalue())[0]
            settings = yaml.safe_load((Path(temp) / "3e17/training.yaml").read_text())
            self.assertEqual(summary["tokens_per_update"], 1048576)
            self.assertEqual(summary["optimizer_updates"], settings["training"]["max_steps"])
            self.assertEqual(summary["optimizer_updates"], 2178)
            validate_curriculum(settings, 8, 1, 16, settings["training"]["max_steps"])

    def test_more_experts_add_capacity_and_recompute_the_router_cost(self):
        config, _, budget = make_recipe("3e17", expert_capacity_fraction=0.8)
        larger, _, more = make_recipe("3e17")
        self.assertEqual(config["moe_intermediate_size"], larger["moe_intermediate_size"])
        self.assertGreater(more["total_parameters"], budget["total_parameters"])
        self.assertGreater(more["reference_8k"]["flops_per_update"], budget["reference_8k"]["flops_per_update"])
        self.assertLessEqual(more["estimated_training_flops"], more["target_flops"])

    def test_tampered_curriculum_is_rejected_before_training(self):
        _, settings, _ = make_recipe("1e19")
        for change in ("cost", "topology"):
            broken = copy.deepcopy(settings)
            if change == "cost":
                broken["training"]["length_curriculum"]["stages"][0]["flops_per_update"] += 1
            else:
                broken["submission"]["data_parallel_size"] = 4
            with self.assertRaises(ValueError):
                validate_curriculum(broken, 8, 1, settings["training"]["gradient_accumulation_steps"],
                                    broken["training"]["max_steps"])

    def test_invalid_candidates_are_rejected(self):
        for options in ({"granularity_shift": 2}, {"granularity_shift": True},
                        {"expert_capacity_fraction": 1.1}, {"expert_alignment": 3},
                        {"learning_rate": float("nan")}, {"learning_rate": 0},
                        *({"tokens_per_update": value} for value in
                          (True, 0, -65536, 65535, 65537, 1048576.0, 65536 * 10**9))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_recipe("3e17", **options)

    def test_one_scaling_rule_and_family_id_for_all_four_tiers(self):
        for shift in (-1, 0, 1):
            recipes = [make_recipe(tier, granularity_shift=shift) for tier in TIERS]
            self.assertEqual(len({s["architecture_family"]["family_id"] for _, s, _ in recipes}), 1)
            for config, settings, _ in recipes:
                self.assertEqual(config["num_hidden_layers"], config["hidden_size"] // 32)
                self.assertEqual(config["num_attention_heads"], config["hidden_size"] // 64)
                self.assertEqual(config["num_key_value_heads"], 2)
                self.assertEqual(config["num_experts_per_tok"], max(2, 2**(config["hidden_size"] // 256 + shift)))
                self.assertEqual(config["num_experts"], maximum_expert_count(config))
                with self.assertRaisesRegex(ValueError, "below the required 10%"):
                    validate_moe_activation_ratio(dict(config, num_experts=config["num_experts"] + 1))
                validate_family_settings(settings)

    def test_per_tier_architecture_change_is_rejected_but_lr_can_vary(self):
        _, settings, _ = make_recipe("1e18")
        altered = copy.deepcopy(settings)
        altered["model"]["hf_config"]["rope_theta"] *= 2
        with self.assertRaises(ValueError):
            validate_family_settings(altered)
        _, changed_lr, _ = make_recipe("1e18", learning_rate=0.003)
        self.assertEqual(settings["architecture_family"], changed_lr["architecture_family"])
        validate_family_settings(changed_lr)
        altered = copy.deepcopy(settings)
        altered["budget"].pop("flops_protocol")
        with self.assertRaisesRegex(ValueError, "competition-reference"):
            validate_family_settings(altered)

    def test_vocabulary_cannot_mask_invalid_activation_fraction(self):
        config, _, _ = make_recipe("3e17")
        config.update(num_experts=256, vocab_size=10_000_000)
        counts = qwen3_moe_parameters(config)
        self.assertGreater(counts["active_parameters_including_input_and_output_vocab"] / counts["total_parameters"], 0.1)
        with self.assertRaisesRegex(ValueError, "below the required 10%"):
            validate_moe_activation_ratio(config)

    def test_legacy_presets_regenerate_unchanged(self):
        for name, world, accumulation, tokens in (("a20m", 8, 8, 2_000_000_000), ("a30m", 4, 16, None)):
            folder = ROOT.parent / f"qwen3_moe_{name}"
            config = json.loads((folder / "config.json").read_text())
            settings = make_training_settings(config, make_plan(config, world, accumulation), tokens)
            self.assertEqual(settings, yaml.safe_load((folder / "training.yaml").read_text()))


if __name__ == "__main__":
    unittest.main()
