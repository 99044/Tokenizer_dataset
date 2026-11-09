# Qwen3 MoE 1e18

The 12-layer model uses 1024-token causal windows in nine layers, with full causal attention in layers 4, 8 and 12 (one-based). It routes to 12 of 128 experts per token per layer. There are 155,523,520 parameters, of which 27,236,800 are active per token including vocabulary matrices. The non-vocabulary active share is 14,653,888 / 142,940,608 (10.25%).

The eight-rank SOAP schedule uses BF16 weights, FP32 master weights and 393,216 input positions per optimizer update. The 2K, 4K and 8K stages run for 8,676, 3,391 and 4,888 updates. The total is 16,955 updates and 6,666,977,280 input positions, with 999,995,547,452,912,640 estimated logical FLOPs. The count includes the per-layer attention windows and retains the existing prediction-graph accounting conventions. The 24K Hybrid 1e18 reference has about 200M parameters, 15,024 updates and 1.48B planned input positions.

Run inside the Qwen3 runtime image with Transformers 5.18 and CUDA FlashAttention 2.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash qwen3_moe_1e18/run_all.sh \
  /path/to/token_cache_24k_v1_full_20261101 \
  /path/to/tok_bpe24k_v1/tokenizer \
  /path/to/tok_bpe24k_v1_dolma3_mix_100M_a_npy \
  /path/to/new-qwen3-moe-1e18-run
```

The pipeline saves native checkpoints every 800 steps and at the final step. It exports `train/step-16955.pt` to `hf-model/`, then evaluates the existing NPY cache at context 8192 and batch size 32. The exported HF directory includes the Python modeling file that preserves the mixed attention pattern. Load that local model with `trust_remote_code=True`.

Regenerate the configuration with the planner.

```bash
python qwen3_moe_a30m/plan.py --config qwen3_moe_1e18/config.json \
  --tier 1e18 --world-size 8 --grad-accumulation 6 \
  --target-tokens 6666666667 --write-training-config qwen3_moe_1e18/training.yaml
```
