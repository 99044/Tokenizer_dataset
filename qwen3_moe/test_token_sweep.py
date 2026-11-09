"""Protect the token sweep's single-variable control and compute budget."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_moe.plan import make_recipe
from context_curriculum import validate_curriculum
from qwen3_moe_token_rules import EXPERT_WIDTHS, TIER, validate_qwen3_moe_settings


class TokenSweepTests(unittest.TestCase):
    def test_eight_widths_trade_active_parameters_for_tokens_at_fixed_flops(self):
        baseline, baseline_settings, previous = make_recipe(TIER)
        self.assertEqual(len(EXPERT_WIDTHS), 8)
        for width in EXPERT_WIDTHS:
            with self.subTest(width=width):
                config, settings, budget = make_recipe(TIER, expert_width=width)
                self.assertEqual(config, dict(baseline, moe_intermediate_size=width))
                self.assertEqual(settings["optimizer"], baseline_settings["optimizer"])
                self.assertEqual(settings["data"], baseline_settings["data"])
                self.assertEqual(budget["tokens_per_update"], 262144)
                self.assertGreater(budget["planned_training_tokens"], previous["planned_training_tokens"])
                self.assertLess(budget["active_non_vocabulary_parameters"], previous["active_non_vocabulary_parameters"])
                self.assertLess(budget["active_parameters_including_input_and_output_vocab"],
                                previous["active_parameters_including_input_and_output_vocab"])
                self.assertEqual(budget["planned_training_tokens"], 262144 * budget["optimizer_updates"])
                activation = validate_qwen3_moe_settings(settings)
                self.assertGreater(10 * activation["active_non_vocabulary_parameters"],
                                   activation["total_non_vocabulary_parameters"])
                stages = settings["training"]["length_curriculum"]["stages"]
                validate_curriculum(settings, 8, 1, 4, budget["optimizer_updates"])
                self.assertEqual([stage["length"] for stage in stages], [2048, 4096, 8192])
                spent = budget["estimated_training_flops"]
                self.assertLessEqual(spent, 3 * 10**17)
                self.assertGreater(spent + stages[-1]["flops_per_update"], 3 * 10**17)
                previous = budget

    def test_non_sweep_configurations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one of"):
            make_recipe(TIER, expert_width=512)
        with self.assertRaisesRegex(ValueError, "3e17"):
            make_recipe("1e18", expert_width=64)
        _, settings, _ = make_recipe(TIER, expert_width=64)
        broken = deepcopy(settings)
        broken["model"]["hf_config"]["moe_intermediate_size"] = 128
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_qwen3_moe_settings(broken)


if __name__ == "__main__":
    unittest.main()
