# 3e17 Qwen3 MoE token sweep

The reference is `gengsen14/scicomp26` `qwen3-moe-paper-scaling` at `947cda1`.
It uses 24 width-512 experts with top-2 routing and budgets 2,284,847,104 token positions.
The eight recipes change only the expert FFN width, keeping all eight decoder layers,
24 experts per layer, top-2 routing, the 24K vocabulary, SOAP, and 262,144 token positions
per optimizer update. Each recipe recomputes the 2K/4K/8K curriculum under 3e17 FLOPs
and must pass the non-vocabulary activation check.

Print the planned counts and FLOPs for all eight widths.

```bash
python qwen3_moe/token_sweep.py
```

Run the following from the repo root in the existing paper-scaling environment.
Each width uses all eight training GPUs. The subsequent evaluation uses GPU 0, batch 64,
sequence length 8192, and the same pretokenized evaluation cache for every width.

```bash
for width in 448 384 320 256 192 128 96 64; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EVAL_GPU=0 EVAL_BATCH_SIZE=64 \
    bash qwen3_moe/run_token_3e17.sh "$width" \
      /mnt/token_cache_24k_v1_full_20261101 \
      /mnt/tok_bpe24k_v1/tokenizer \
      /mnt/1103_data/tok_bpe24k_v1_dolma3_mix_100M_a_npy \
      "/mnt/experiments/moe-token-3e17-w$width" || break
done
```

Each completed output directory contains training checkpoints, `hf-model/`,
`eval-result.json` with BPB and timing, `training-config.yaml`, `token-plan.json`,
and `flops-report.json`. The token plan counts input positions processed by the
training batches. The FLOPs audit checks the full completed training log.
