# Qwen3 MoE A30M

The architecture uses the stock Qwen3 MoE implementation with 128 experts and top-8 routing. The 24K vocabulary model has 198,125,920 total parameters and 30,026,080 active parameters per token including both vocabulary matrices.

Run the four-rank SOAP experiment in the existing SCICOMP26 training environment:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash qwen3_moe_a30m/train.sh /path/to/24k-npy-cache /path/to/run
```

The cache must match the 24K tokenizer hash in the training config and include train/validation token arrays and their `is_text` sidecars. Checkpoints at regular 800-step milestones and stage boundaries support `--resume /path/to/checkpoint.pt`. A short run can use `--stop-after 1` to preserve the full schedule.

The curriculum spends 50%, 20%, and the remainder of the 3e17 prediction-graph budget at 2K, 4K, and 8K. Four ranks and 16 accumulation steps keep 524,288 positions per update. The plan contains 1,618 updates and 848,297,984 token positions. The ledger excludes auxiliary/LM losses, expert dispatch, top-k selection, optimizer and communication costs.

The runner uses FP32 masters with BF16 model weights. It sums actual top-k selection counts across ranks per layer, accumulates counts across microsteps within the current optimizer update, and adds the Qwen-style approximate global-batch auxiliary loss to masked CE. Logged load ratios use the per-step mean assignment count as 1.0.

Regenerate the committed training config and check the model:

```sh
python qwen3_moe_a30m/plan.py --write-training-config qwen3_moe_a30m/training.yaml
python qwen3_moe_a30m/model.py
```

`budget.json` is the full-8K architecture reference used by the plan and model checks. The variable-length production plan is recorded in `training.yaml`.
