# Qwen3 MoE A30M

This preset uses the Transformers Qwen3 MoE implementation and keeps the 128-expert, top-8 router, 32 query heads, four KV heads, and separate input and output vocabulary matrices from [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json).

The scaled configuration uses 19 layers, hidden size 256, head dimension 16, expert width 96, the project's 24,576-token vocabulary, and an 8,192-token context. It has 198,125,920 total parameters and 30,026,080 active parameters per token including both vocabulary matrices.

From the repository root:

```sh
python qwen3_moe_a30m/plan.py
python qwen3_moe_a30m/model.py
```

The first command generates the analytical budget. Four ranks with one sequence per rank and 16 gradient accumulation steps give 885 updates and 463,994,880 tokens under the 3e17 prediction-graph estimate. The budget excludes loss, expert dispatch and top-k selection, optimizer and communication costs.

The second command checks initialization, parameter counts, short CPU forward/backward, and router auxiliary gradients. To use the model in code, import `build_model` from `qwen3_moe_a30m.model`. Production training requires integrating the [Qwen3 global-batch routing balance objective](https://arxiv.org/html/2505.09388v1#S2) with the project's masked DDP loss.
