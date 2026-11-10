"""Cross-check exact matrix terms with the repository's reviewed Megatron formula."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from qwen3_moe.plan import TIERS, make_recipe
from training.count_flops import DenseTransformerSpec, megatron_dense_flops_per_step
from qwen3_flops import qwen3_moe_cost, qwen3_moe_cost_breakdown
from qwen3_moe_flops import PROTOCOL, load_reference


class FlopsTests(unittest.TestCase):
    def test_standard_and_router_gemms_follow_two_mnk_with_separate_gradients(self):
        for tier in TIERS:
            config, _, _ = make_recipe(tier)
            for length in (2048, 4096, 8192):
                with self.subTest(tier=tier, length=length):
                    spec = DenseTransformerSpec(
                        num_layers=config["num_hidden_layers"], hidden_size=config["hidden_size"],
                        num_attention_heads=config["num_attention_heads"],
                        num_query_groups=config["num_key_value_heads"], head_dim=config["head_dim"],
                        intermediate_size=config["moe_intermediate_size"] * config["num_experts_per_tok"],
                        sequence_length=length, padded_vocab_size=config["vocab_size"], activation="swiglu",
                    )
                    reference = megatron_dense_flops_per_step(spec, 1)
                    audit = qwen3_moe_cost_breakdown(config, length, protocol=PROTOCOL)
                    components = audit["megatron_components"]
                    for name, ref in (("attention_projections", "attention_projection"),
                                      ("attention_core", "attention_core"),
                                      ("active_expert_projections", "mlp"), ("lm_head", "logits")):
                        self.assertEqual(components[name], reference[ref])
                    self.assertEqual(audit["router_projection_gemm"],
                                     6 * length * config["hidden_size"] * config["num_experts"] * config["num_hidden_layers"])
                    self.assertEqual(sum(audit["components"].values()), audit["estimated_total"])
                    self.assertEqual(qwen3_moe_cost(config, length, protocol=PROTOCOL)[0], audit["estimated_total"])
                    self.assertIs(audit["official_ledger"], False)
                    self.assertIs(audit["reference_matched"], True)
                    self.assertIs(audit["router_graph_audited"], True)

    def test_dense_prefix_counts_only_four_himoe_routers(self):
        _, _, budget = make_recipe("3e17", dense_prefix=True)
        _, _, baseline = make_recipe("3e17")
        for length in (2048, 4096, 8192):
            with self.subTest(length=length):
                audit = budget["flops_audit"][str(length)]
                original = baseline["flops_audit"][str(length)]
                components = audit["megatron_components"]
                self.assertEqual(components["dense_ffn_projections"], components["active_expert_projections"])
                self.assertEqual(audit["megatron_reference_total"], original["megatron_reference_total"])
                self.assertEqual(audit["router_projection_gemm"], 6 * 8192 * 4 * 256 * 46)
                self.assertEqual(sum(audit["components"].values()), audit["estimated_total"])
                operations = {op["name"]: op for op in audit["router_operations"]}
                self.assertEqual(operations["himoe.ema_update"]["calls"], 4)
                self.assertEqual(operations["z_loss.logsumexp.exp"]["calls"], 4)

    def test_auxiliary_scalars_are_counted_per_microbatch_not_per_sequence(self):
        config, _, _ = make_recipe("3e17")
        single = qwen3_moe_cost(config, 2048, protocol=PROTOCOL)[0]
        double = qwen3_moe_cost(config, 2048, protocol=PROTOCOL, micro_batch_size=2)[0]
        # The Hi-MoE EMA ledger adds three constant scalar operations per
        # expert, net of the token-sum reduction's -E term.
        standard_per_layer = 4 * config["num_experts"] + 3
        himoe_ema_per_layer = 3 * config["num_experts"]
        constant = config["num_hidden_layers"] * (standard_per_layer + himoe_ema_per_layer) + 8
        self.assertEqual(2 * single - double, constant)

    def test_reference_snapshot_is_the_previously_reviewed_function(self):
        _, manifest = load_reference()
        self.assertEqual(manifest["function_sha256"],
                         "4d6e6daf89926f83f4e5009f337540cf1f16850e1b18d9b1c6cb6fb74b93fc97")


if __name__ == "__main__":
    unittest.main()
