# Qwen3 MoE A20M, 2B tokens

This preset uses 8 layers, hidden size 256, 32 query heads, 4 KV heads, head dimension 8, and 128 SwiGLU experts of width 120 with top-8 routing. The 24K vocabulary matrices are untied. The model has 108,401,024 total parameters, 19,927,424 active parameters including both vocabulary matrices, and 7,344,512 active parameters excluding them.

The four-rank SOAP plan keeps BF16 model weights, FP32 masters, 16 accumulation microsteps, and 524,288 input positions per update. It runs 1,473 updates at 2K, 763 at 4K, and 1,579 at 8K, giving 3,815 updates and 2,000,158,720 input positions. The last stage sees 827,850,752 positions. The estimated prediction-graph budget is 299,999,377,025,953,792 FLOPs. The 2B target is rounded up to a complete update, and roughly 20% of updates are reserved for 4K before assigning the remaining budget to 8K.

In the existing four-GPU training environment, use the same 24K cache and tokenizer paths as the A30M pipeline.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash qwen3_moe_a20m/run_all.sh \
  /path/to/24k-training-cache /path/to/24k-tokenizer \
  /path/to/24k-eval-npy-cache /path/to/new-run
```

This trains to `train/step-3815.pt`, exports `hf-model/` on CPU, and runs packed-cache evaluation to `eval-result.json`. The evaluation defaults to one GPU, batch size 32, and sequence length 8192.

To run training by itself or resume a checkpoint:

```sh
TRAIN_CONFIG="$PWD/qwen3_moe_a20m/training.yaml" CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash qwen3_moe_a30m/train.sh "$TRAIN_CACHE" "$RUN_DIR/train"
```

Append `--resume /path/to/step-N.pt` for a saved checkpoint from the same plan.

Regenerate the plan from the HF architecture config:

```sh
python qwen3_moe_a30m/plan.py --config qwen3_moe_a20m/config.json \
  --target-tokens 2000000000 --write-training-config qwen3_moe_a20m/training.yaml
```

`budget.json` records the full-8K reference. `training.yaml` records the production curriculum.
