"""CPU validation with the pinned training environment; no training GPUs needed."""

import json
from copy import deepcopy
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import transformers
from transformers import Qwen3MoeConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from qwen3_moe.plan import TIERS, make_recipe
from fp32_master import FP32MasterWeights
from optimizer_factory import build_optimizer
from qwen3_moe_training import AccumulatedRouterBalance, Qwen3MoeTrainingConfig, himoe_regularizers, router_z_loss
from hf_qwen3_himoe import Qwen3HiMoeForCausalLM
from qwen3_moe_rules import validate_family_settings


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    reports = []
    for tier in TIERS:
        config, settings, budget = make_recipe(tier)
        activation = validate_family_settings(settings, check_runtime=True)
        # Check the full production architecture without allocating its weights.
        with torch.device("meta"):
            lm = Qwen3HiMoeForCausalLM(Qwen3MoeConfig.from_dict(deepcopy(config)))
        total = sum(p.numel() for p in lm.parameters())
        experts = [p for name, p in lm.named_parameters() if ".mlp.experts." in name]
        if not experts or any(p.ndim != 3 for p in experts):
            raise ValueError("Use the pinned Transformers version with packed 3D experts")
        expert_count = sum(p.numel() for p in experts)
        active = total - expert_count + expert_count // config["num_experts"] * config["num_experts_per_tok"]
        assert total == budget["total_parameters"]
        assert active == budget["active_parameters_including_input_and_output_vocab"]
        vocabulary = 2 * config["vocab_size"] * config["hidden_size"]
        assert total - vocabulary == activation["total_non_vocabulary_parameters"]
        assert active - vocabulary == activation["active_non_vocabulary_parameters"]
        assert 10 * (active - vocabulary) >= total - vocabulary
        del lm, experts

        # Exercise every selected top-k using the real masked-CE/router/FP32 SOAP path.
        tiny = dict(config, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=8, vocab_size=32,
                    intermediate_size=64, moe_intermediate_size=64 // config["num_experts_per_tok"])
        declaration = deepcopy(tiny)
        training_config = Qwen3MoeTrainingConfig(tiny)
        assert tiny == declaration
        model = training_config.build()
        model.init_weights(8192, torch.device("cpu"))
        master = FP32MasterWeights(model)
        optimizer, _ = build_optimizer(model, settings, parameter_map=master.parameter_map)
        optimizer = master.install(optimizer)
        before = model.lm.lm_head.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            balance = AccumulatedRouterBalance(model.lm.config, torch.device("cpu"))
            for _ in range(2):
                inputs = torch.randint(4, 32, (1, 16))
                labels = torch.randint(4, 32, (1, 16))
                labels[:, 0] = -100
                logits, routers = model(inputs, route=True)
                aux = balance(routers)
                ce = F.cross_entropy(logits.float().reshape(-1, 32), labels.flatten())
                inter, intra = himoe_regularizers(routers, config["hi_moe"])
                loss = (ce + config["router_aux_loss_coef"] * aux
                        + config["router_z_loss_coef"] * router_z_loss(routers)
                        + inter + intra) / 2
                assert torch.isfinite(loss)
                loss.backward()
            assert torch.all(balance.counts.sum(-1) == 32 * config["num_experts_per_tok"])
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert all(layer.mlp.gate.weight.grad.abs().sum() > 0 for layer in model.lm.model.layers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        assert not torch.equal(before, model.lm.lm_head.weight)
        master.validate_checkpoint(dict(model=model.state_dict(), fp32_master=master.state_dict()))
        reports.append(dict(tier=tier, total_parameters=total, active_parameters=active,
                            active_non_vocabulary_fraction=activation["active_non_vocabulary_fraction"],
                            top_k=config["num_experts_per_tok"], masked_ce_router_soap="passed"))
    print(json.dumps(dict(status="passed", torch=torch.__version__, transformers=transformers.__version__,
                          full_architecture="meta device", smoke_device="cpu", recipes=reports), indent=2))


if __name__ == "__main__":
    main()
