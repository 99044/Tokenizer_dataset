"""Check the token ladder's spacing, model controls and compute budget."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_moe.plan import make_recipe
from context_curriculum import validate_curriculum
from qwen3_moe_active_ladder import ACTIVE_VARIANTS, validate_qwen3_moe_settings
from qwen3_moe_token_ladder import COMPACT_WIDTH, LADDER_WIDTHS
from qwen3_moe_token_rules import TIER, make_token_architecture


class TokenSweepTests(unittest.TestCase):
    def test_active_ladder_spans_22m_to_12m_in_even_steps(self):
        baseline, baseline_settings, _ = make_recipe(TIER)
        self.assertEqual(len(ACTIVE_VARIANTS), 8)
        counts = []
        previous = None
        for variant, (hidden, width) in ACTIVE_VARIANTS.items():
            with self.subTest(variant=variant):
                config, settings, budget = make_recipe(TIER, active_variant=variant)
                self.assertEqual(config, dict(baseline, hidden_size=hidden, intermediate_size=4 * hidden,
                                              moe_intermediate_size=width))
                self.assertEqual(settings["optimizer"], baseline_settings["optimizer"])
                self.assertEqual(settings["data"], baseline_settings["data"])
                counts.append(budget["active_parameters_including_input_and_output_vocab"])
                activation = validate_qwen3_moe_settings(settings)
                self.assertGreater(10 * activation["active_non_vocabulary_parameters"],
                                   activation["total_non_vocabulary_parameters"])
                validate_curriculum(settings, 8, 1, 4, budget["optimizer_updates"])
                self.assertEqual(budget["planned_training_tokens"], 262144 * budget["optimizer_updates"])
                stages = settings["training"]["length_curriculum"]["stages"]
                spent = budget["estimated_training_flops"]
                self.assertLessEqual(spent, 3 * 10**17)
                self.assertGreater(spent + stages[-1]["flops_per_update"], 3 * 10**17)
                if previous is not None:
                    self.assertGreater(budget["planned_training_tokens"], previous["planned_training_tokens"])
                    self.assertLess(budget["active_non_vocabulary_parameters"], previous["active_non_vocabulary_parameters"])
                previous = budget
        self.assertLess(abs(counts[0] - 22_000_000), 1_000)
        self.assertLess(abs(counts[-1] - 12_000_000), 1_000)
        for left, right in zip(counts, counts[1:]):
            self.assertTrue(1_350_000 <= left - right <= 1_510_000)

    def test_active_variant_rejects_combined_selection_and_modified_shape(self):
        with self.assertRaisesRegex(ValueError, "Select an active-parameter variant"):
            make_recipe(TIER, expert_width=48, active_variant="a12m")
        with self.assertRaisesRegex(ValueError, "3e17"):
            make_recipe("1e18", active_variant="a12m")
        _, settings, _ = make_recipe(TIER, active_variant="a12m")
        settings["model"]["hf_config"]["moe_intermediate_size"] += 8
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_qwen3_moe_settings(settings)

    def test_eight_sizes_trade_active_parameters_for_tokens_at_fixed_flops(self):
        baseline, baseline_settings, previous = make_recipe(TIER)
        self.assertEqual(len(LADDER_WIDTHS), 8)
        gaps = []
        for width in LADDER_WIDTHS:
            with self.subTest(width=width):
                config, settings, budget = make_recipe(TIER, expert_width=width)
                expected = dict(baseline, moe_intermediate_size=width)
                if width == COMPACT_WIDTH:
                    expected.update(hidden_size=144, intermediate_size=576)
                self.assertEqual(config, expected)
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
                gap = previous["total_parameters"] - budget["total_parameters"]
                self.assertGreater(gap, 0)
                gaps.append(gap)
                previous = budget
        self.assertLess(max(gaps) - min(gaps), 2_000_000)
        self.assertEqual(previous["total_parameters"], 11_975_056)

    def test_previous_width_recipe_is_still_valid(self):
        config, family = make_token_architecture(64)
        recipe, settings, budget = make_recipe(TIER, expert_width=64)
        self.assertEqual(recipe, config)
        self.assertEqual(settings["architecture_family"], family)
        self.assertEqual(budget["planned_training_tokens"], 3_111_387_136)
        validate_qwen3_moe_settings(settings)

    def test_non_sweep_configurations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one of"):
            make_recipe(TIER, expert_width=512)
        with self.assertRaisesRegex(ValueError, "3e17"):
            make_recipe("1e18", expert_width=64)
        _, settings, _ = make_recipe(TIER, expert_width=COMPACT_WIDTH)
        broken = deepcopy(settings)
        broken["model"]["hf_config"]["moe_intermediate_size"] = 128
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_qwen3_moe_settings(broken)


if __name__ == "__main__":
    unittest.main()
