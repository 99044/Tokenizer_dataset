# Training, export, and evaluation

This branch contains one configuration for each of the four FLOPs tiers.

| Tier | Source | Vocabulary | GPUs | Accumulation | Tokens/update | Steps | Optimizer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3e17 | `submission_20261107` | 24,576 | 8 | 4 | 262,144 | 5,502 | SOAP; compressed branch AdamW |
| 1e18 | `MuonF_Muown` SOAP FP32-master/EMA recipe, scaled to 8 GPUs | 24,576 | 8 | 2 | 131,072 | 11,268 | SOAP with FP32 master weights and EMA |
| 3e18 | `submission_20261103` | 24,576 | 8 | 4 | 262,144 | 16,902 | SOAP |
| 1e19 | completed wide 330M trunk + compressed branch run | 32,768 | 8 | 8 | 524,288 | 8,143 | Muon; scalar AdamW group includes large compression matrices |

The 3e17 architecture and training settings come from its source branch; its run name and output directory use `submission_1108`. Its 2048/4096/8192 context curriculum, FP32 master weights, residual extensions, and 5% warmup with a final linear cooldown remain in place. The 1e18 recipe uses BF16 model weights with FP32 SOAP master weights and two FP32 EMA tracks, starting after 70% of updates with token half-lives of 20M and 80M. It uses eight GPUs with two accumulation steps, preserving the original 1,476,919,296 training tokens and estimated FLOPs while reducing optimizer updates to 11,268. Warmup is 30 updates and cooldown is 1,883 updates, preserving their approximate token durations. A four-GPU checkpoint cannot be resumed under this plan. The 3e18 YAML keeps the 1103 model, data volume, and SOAP optimizer. Changing from four GPUs with eight accumulation steps to eight GPUs with four accumulation steps preserves 262,144 tokens/update; it uses 845 warmup steps (about 5%), a plateau, and 5,071 cooldown steps ending at 3e-5. The 1e19 YAML uses the completed 32K, 16-layer, hidden-1024, FFN-4352 model with a 388-token compressed branch. Its Muon grouping preserves the completed run's AdamW treatment of the two large compression matrices.

The 24K and 32K tiers require **different token caches, evaluation caches, and tokenizer directories**. Supply the matching files to each command. The launcher checks the vocabulary and tokenizer identity before training. Each run trains, exports its final checkpoint to Hugging Face format, and evaluates the HF model with one GPU, batch 32, and FFN + RMSNorm fusion. The 1e18 launcher additionally exports and evaluates the raw weights and both EMA variants separately; it does not silently select a variant. `TRITON_F32_DEFAULT=ieee` is set by the launcher. Logs and `summary.json` are written under each output directory.

```bash
bash run_submission_3e17.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir /path/to/24k-train-cache \
  --eval-cache-dir /path/to/24k-eval-cache \
  --tokenizer-dir /path/to/24k-tokenizer \
  --output-dir /path/to/runs/3e17

bash run_submission_1e18.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir /path/to/24k-train-cache \
  --eval-cache-dir /path/to/24k-eval-cache \
  --tokenizer-dir /path/to/24k-tokenizer \
  --output-dir /path/to/runs/1e18

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
