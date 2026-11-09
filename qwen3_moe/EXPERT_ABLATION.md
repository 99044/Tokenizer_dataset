# 3e17 expert ablation

All three recipes use the public paper-scaling baseline at commit `7d5342f`, including adaptive token loss, SOAP, router auxiliary loss and router Z-loss. The planner recalculates each curriculum under the existing 3e17 FLOPs ledger.

- `e24-k2-w512` is the reference with 8,716 steps, 2,284,847,104 input positions and 10.27% non-vocabulary activation.
- `e48-k4-w256` splits the same expert FFN capacity more finely. It keeps the active FFN width at 1,024 and doubles assignments per token. The plan has 8,693 steps, 2,278,817,792 input positions and 10.32% non-vocabulary activation. The larger router changes parameter count and FLOPs. The auxiliary coefficient is shared across runs, and the raw auxiliary loss scale depends on top-k.
- `e20-k2-w512` reduces the expert FFN pool by one sixth while retaining top-2 and width 512. The plan has 8,719 steps, 2,285,633,536 input positions and 12.26% non-vocabulary activation.

Run sequentially on the same eight GPUs, with the same GPU 0 and evaluation settings.

```bash
for variant in e24-k2-w512 e48-k4-w256 e20-k2-w512; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EVAL_GPU=0 EVAL_BATCH_SIZE=64 \
    MOE_EXPERT_VARIANT="$variant" bash qwen3_moe/run_3e17.sh \
      /mnt/token_cache_24k_v1_full_20261101 \
      /mnt/tok_bpe24k_v1/tokenizer \
      /mnt/1103_data/tok_bpe24k_v1_dolma3_mix_100M_a_npy \
      "/mnt/experiments/moe-3e17-$variant" || break
done
```

The output includes the exact training YAML, checkpoint, HF model, evaluation JSON and completed FLOPs audit. The evaluator scores the existing NPY Official-A cache and records both raw forward seconds and normalized EvalTime.

Each training update logs `train/router_layer_max_to_mean` in decoder order. The ratio uses assignment counts pooled across all microbatches and training ranks for that update. Compare the mean and peak of each layer's ratio over its last 200 updates.

```bash
python qwen3_moe/expert_ablation.py compare \
  /mnt/experiments/moe-3e17-e24-k2-w512 \
  /mnt/experiments/moe-3e17-e48-k4-w256 \
  /mnt/experiments/moe-3e17-e20-k2-w512
```

Run the reference again when using the reported Official-A BPB of 1.144821 as context for a new comparison. The rerun supplies matched timing, training settings and the new layer metrics. Candidate BPB and speed are measured outcomes.
