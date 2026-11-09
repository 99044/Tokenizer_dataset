"""Instantiate the A30M preset using the stock Qwen3 MoE implementation."""

import json
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent


def build_model(*, dtype=torch.float32, attn_implementation="sdpa"):
    config = AutoConfig.from_pretrained(ROOT, local_files_only=True)
    return AutoModelForCausalLM.from_config(
        config, dtype=dtype, attn_implementation=attn_implementation
    )


if __name__ == "__main__":
    # A short CPU check exercises the stock router, its auxiliary loss and
    # gradients without occupying a training GPU.
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model()
    config = model.config
    budget = json.loads((ROOT / "budget.json").read_text(encoding="utf-8"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    expert_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".mlp.experts." in name
    )
    active_parameters = (
        parameters - expert_parameters
        + expert_parameters // config.num_experts * config.num_experts_per_tok
    )
    assert parameters == budget["total_parameters"]
    assert active_parameters == budget["active_parameters_including_input_and_output_vocab"]
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    tokens = torch.randint(4, config.vocab_size, (1, 16))
    output = model(input_ids=tokens, labels=tokens, use_cache=False, output_router_logits=True)
    assert output.logits.shape == (1, 16, config.vocab_size)
    assert len(output.router_logits) == config.num_hidden_layers
    assert all(gate.shape == (16, config.num_experts) for gate in output.router_logits)
    assert torch.isfinite(output.loss) and torch.isfinite(output.aux_loss)
    router_weights = [layer.mlp.gate.weight for layer in model.model.layers]
    aux_gradients = torch.autograd.grad(output.aux_loss, router_weights, retain_graph=True)
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in aux_gradients)
    output.loss.backward()
    assert all(weight.grad is not None and torch.isfinite(weight.grad).all() for weight in router_weights)
    print(json.dumps({
        "status": "passed",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "total_parameters": parameters,
        "active_parameters": active_parameters,
        "logits_shape": list(output.logits.shape),
        "router_aux_loss": output.aux_loss.item(),
        "forward_and_backward": "passed",
    }, indent=2))
