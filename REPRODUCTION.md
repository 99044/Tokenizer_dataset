# Training, export, and evaluation

This branch contains only `FLOPs_3e17.yaml`, `FLOPs_3e18.yaml`, and `FLOPs_1e19.yaml`. The 1e18 configuration is pending.

| Tier | Source | Vocabulary | GPUs | Accumulation | Tokens/update | Steps | Optimizer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3e17 | `submission_20261107` | 24,576 | 8 | 4 | 262,144 | 5,502 | SOAP; compressed branch AdamW |
| 3e18 | `submission_20261103` | 24,576 | 8 | 4 | 262,144 | 16,902 | SOAP |
| 1e19 | completed wide 330M trunk + compressed branch run | 32,768 | 8 | 8 | 524,288 | 8,143 | Muon; scalar AdamW group includes large compression matrices |

The 3e17 YAML is unchanged from its source branch. Its 2048/4096/8192 context curriculum, FP32 master weights, residual extensions, and 5% warmup with a final linear cooldown remain in place. The 3e18 YAML keeps the 1103 model, data volume, and SOAP optimizer. Changing from four GPUs with eight accumulation steps to eight GPUs with four accumulation steps preserves 262,144 tokens/update; it uses 845 warmup steps (about 5%), a plateau, and 5,071 cooldown steps ending at 3e-5. The 1e19 YAML uses the completed 32K, 16-layer, hidden-1024, FFN-4352 model with a 388-token compressed branch. Its Muon grouping preserves the completed run's AdamW treatment of the two large compression matrices.

The 24K and 32K tiers require **different token caches, evaluation caches, and tokenizer directories**. Supply the matching files to each command. The launcher checks the vocabulary and tokenizer identity before training. Each run trains, exports its final checkpoint to Hugging Face format, and evaluates the HF model with one GPU, batch 32, and FFN + RMSNorm fusion. `TRITON_F32_DEFAULT=ieee` is set by the launcher. Logs and `summary.json` are written under each output directory.

```bash
bash run_submission_3e17.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir /path/to/24k-train-cache \
  --eval-cache-dir /path/to/24k-eval-cache \
  --tokenizer-dir /path/to/24k-tokenizer \
  --output-dir /path/to/runs/3e17

bash run_submission_3e18.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir /path/to/24k-train-cache \
  --eval-cache-dir /path/to/24k-eval-cache \
  --tokenizer-dir /path/to/24k-tokenizer \
  --output-dir /path/to/runs/3e18

bash run_submission_1e19.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir /path/to/32k-train-cache \
  --eval-cache-dir /path/to/32k-eval-cache \
  --tokenizer-dir /path/to/32k-tokenizer \
  --output-dir /path/to/runs/1e19
```

Append `--dry-run` to inspect the resolved command without starting training. After an interruption, rerun the same command with `--resume`; the launcher uses the latest complete checkpoint and stops after repeated attempts without progress. Append `--evaluate-only` to export and evaluate an existing final checkpoint. The YAMLs retain at most two rolling checkpoints.
