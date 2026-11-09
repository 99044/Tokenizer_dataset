import subprocess
import sys
import unittest
from pathlib import Path


class Qwen3WindowedMoeTests(unittest.TestCase):
    def test_fresh_process_selects_grouped_experts(self):
        # Building the stock model first can populate an inherited cache and
        # hide the fallback, so test the mixed model in a clean interpreter.
        script = """
import torch
from transformers import Qwen3MoeConfig
from modeling_qwen3_windowed_moe import Qwen3WindowedMoeForCausalLM

config = Qwen3MoeConfig(
    vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, intermediate_size=16, moe_intermediate_size=8,
    num_experts=4, num_experts_per_tok=2, local_attention_window=4,
    global_attention_layers=[1],
)
model = Qwen3WindowedMoeForCausalLM._from_config(config, attn_implementation="sdpa")
assert model.config._experts_implementation == "grouped_mm"
assert [layer.self_attn.sliding_window for layer in model.model.layers] == [4, None]
inputs = torch.arange(8).view(1, -1)
grouped = model(inputs, use_cache=False).logits
grouped.square().mean().backward()
grouped_grad = model.model.layers[0].mlp.experts.gate_up_proj.grad.clone()
model.zero_grad(set_to_none=True)
model.set_experts_implementation("eager")
eager = model(inputs, use_cache=False).logits
eager.square().mean().backward()
torch.testing.assert_close(grouped, eager, atol=1e-6, rtol=1e-5)
torch.testing.assert_close(
    grouped_grad, model.model.layers[0].mlp.experts.gate_up_proj.grad,
    atol=1e-6, rtol=1e-5,
)
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1], check=True,
        )


if __name__ == "__main__":
    unittest.main()
