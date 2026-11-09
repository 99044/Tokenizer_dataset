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

## Train, export and evaluate

Run all stages from the repository root in the SCICOMP26 training environment.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash qwen3_moe_a30m/run_all.sh \
  /path/to/24k-training-cache /path/to/24k-tokenizer \
  /path/to/24k-eval-npy-cache /path/to/new-run
```

The pipeline validates tokenizer identity before training. It uses four GPUs for SOAP training, the CPU for BF16 HF export and tokenizer packaging, then one GPU for full packed-cache BPB and EvalTime at batch size 32 and sequence length 8192. `EVAL_GPU` selects the evaluation GPU, defaulting to the first training device. `EVAL_BATCH_SIZE` overrides 32.

The result directory contains `train/step-1618.pt`, `hf-model/`, `eval-result.json`, and the three stage logs. The evaluation excludes vocabulary matrices from its active parameter count, including 8 of 128 equal-sized experts per sparse layer.

To run or retry individual stages, set `TRAIN_CACHE`, `TOKENIZER_DIR`, `EVAL_CACHE` and `RUN_DIR` to the same four paths:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash qwen3_moe_a30m/train.sh "$TRAIN_CACHE" "$RUN_DIR/train"

CUDA_VISIBLE_DEVICES= python training/export_hf.py \
  --checkpoint "$RUN_DIR/train/step-1618.pt" --tokenizer-dir "$TOKENIZER_DIR" \
  --output-dir "$RUN_DIR/hf-model" --validation-device cpu

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BATCH_SIZE=32 SEQUENCE_LENGTH=8192 FUSION_MODE=none \
  bash evaluation/eval.sh "$EVAL_CACHE" "$RUN_DIR/hf-model" "$RUN_DIR/eval-result.json"
```

Choose a fresh HF output directory or result filename when retrying a completed or partially written stage.
