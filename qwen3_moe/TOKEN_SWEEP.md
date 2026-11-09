# 3e17 Qwen3 MoE token sweep

The reference is `gengsen14/scicomp26` `qwen3-moe-paper-scaling` at `947cda1`.
It uses 24 width-512 experts with top-2 routing and budgets 2,284,847,104 token positions.
The size ladder steps down by approximately 9.44–10.49M total parameters between
adjacent experiments. The first seven recipes keep hidden size 256 and reduce the
expert FFN width. The last recipe uses hidden size 144 and width-48 experts to
reach 11,975,056 total parameters, including both vocabulary matrices. All eight
keep eight decoder layers, four query heads and two KV heads of dimension 64,
24 experts per layer, top-2 routing, the 24K vocabulary, SOAP, and 262,144 token
positions per optimizer update. Each recipe recomputes the 2K/4K/8K curriculum
under 3e17 FLOPs and must pass the non-vocabulary activation check.

Print the planned counts and FLOPs for all eight widths.

```bash
python qwen3_moe/token_sweep.py
```

Run the following from the repo root in the existing paper-scaling environment.
Each width uses all eight training GPUs. The subsequent evaluation uses GPU 0, batch 64,
sequence length 8192, and the same pretokenized evaluation cache for every width.

```bash
for width in 440 376 312 248 184 120 56 48; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EVAL_GPU=0 EVAL_BATCH_SIZE=64 \
    bash qwen3_moe/run_token_3e17.sh "$width" \
      /mnt/token_cache_24k_v1_full_20261101 \
      /mnt/tok_bpe24k_v1/tokenizer \
      /mnt/1103_data/tok_bpe24k_v1_dolma3_mix_100M_a_npy \
      "/mnt/experiments/moe-token-3e17-ladder-w$width" || break
done
```

Each completed output directory contains training checkpoints, `hf-model/`,
`eval-result.json` with BPB and timing, `training-config.yaml`, `token-plan.json`,
and `flops-report.json`. The token plan counts input positions processed by the
training batches. The FLOPs audit checks the full completed training log.
