# Four Qwen3 MoE recipes for 8 x H800

These are **unmeasured starting recipes** for optimizing the competition's
four-tier Performance score, combining fitted BPB and scaling consistency.
No full training run or BPB improvement is claimed. The hardware target
assumes 80 GiB per H800. All tiers use the same established 24,576-token
tokenizer and native NPY/text-mask caches.

The design follows [Slicing and Dicing](https://arxiv.org/html/2605.11689),
sections 3.1–3.3 and appendix C.3: increase inactive capacity within memory
limits, use coarser experts at smaller active scales, avoid top-1, and use
dropless token-choice routing without shared experts. The paper also finds
that small MoEs need sufficient training data. These trends motivate the
recipes; they do not establish the best BPB on our data.

| Tier | Layers / hidden | Heads / KV heads / head dim | Experts / top-k / expert width | Granularity | SOAP peak LR | Tokens/update |
| --- | --- | --- | --- | --- | --- | ---: |
| 3e17 | 8 / 256 | 4 / 2 / 64 | 24 / 2 / 512 | 1/2 | 0.003 | 262,144 |
| 1e18 | 12 / 384 | 6 / 2 / 64 | 24 / 2 / 768 | 1/2 | 0.002 | 524,288 |
| 3e18 | 16 / 512 | 8 / 2 / 64 | 47 / 4 / 512 | 1/4 | 0.0015 | 1,048,576 |
| 1e19 | 24 / 768 | 12 / 2 / 64 | 94 / 8 / 384 | 1/8 | 0.001 | 1,048,576 |

## Shared scaling rule and minimum activation fraction

All four sizes use the same code and one predeclared family. The rules permit
size parameters to scale; they do not require identical expert count/top-k:
`layers = hidden // 32`, `query_heads = hidden // 64`, `kv_heads = 2`,
`head_dim = 64`, `top_k = max(2, 2**(hidden // 256 + granularity_shift))`,
`expert_width = 4 * hidden / top_k`. The default shift is zero, giving
top-2/2/4/8 and expert granularities 1/2, 1/2, 1/4, 1/8. This transfers the
paper's coarse-to-fine trend into a single size rule; it is a candidate to
validate on our data, not a measured optimum. Layer types, connections,
SwiGLU, RoPE and the softmax/token-choice/top-k algorithm stay the same.
`architecture_family` records the formulas, model/generator source hashes,
Transformers version and a shared family ID. Launch and model construction
check the declaration and reject per-tier architecture edits.

The supplied rules require `N_active / N_total >= 10%` after excluding
**both input embedding and output vocabulary projection**. For fixed-width,
dropless top-k experts the active count is exact, independent of which experts
are selected. Routers, attention and norms remain counted in numerator and
denominator. To exploit the paper's capacity trend within the competition
constraint, expert count is the **largest legal integer at each scale**:

```text
A = always-active non-vocabulary parameters other than router weights
P = 3 * layers * hidden * expert_width       (one expert across all layers)
R = layers * hidden                        (one router row across all layers)
N_active = A + E*R + top_k*P
N_total  = A + E*(R+P)
E_max = floor((9*A + 10*top_k*P) / (P - 9*R))
```

Thus experts scale to 24/24/47/94. Each count passes the integer inequality
`10*N_active >= N_total`; adding one more expert fails it. The expert-only
activation rate may be below 10% because the rule covers all non-vocabulary
parameters. Generation checks the complete numerator/denominator.

| Tier | Active non-vocabulary parameters | Total non-vocabulary parameters | Active fraction |
| --- | ---: | ---: | ---: |
| 3e17 | 7,918,848 | 77,124,864 | 10.27% |
| 1e18 | 26,073,984 | 259,644,288 | 10.04% |
| 3e18 | 61,221,376 | 602,286,592 | 10.16% |
| 1e19 | 204,672,768 | 2,030,767,872 | 10.08% |

The earlier 256/128-expert drafts failed this rule and have been replaced.
The archived A20M/A30M presets also fail the supplied minimum and are no longer
accepted by the training model-config validator. Their files remain historical
experiment records.

Each active FFN has combined width `4 * hidden`. The 64-dimensional attention
heads replace the old 8/16-dimensional, 32-head presets. Larger tiers increase
trunk capacity as well as data volume. The fixed router auxiliary coefficient
remains 0.001 with the existing accumulated global-count approximation.
Router Z-loss also uses coefficient **0.001** in all four tiers:
`mean_layers(mean_tokens(logsumexp(router_logits.float())**2))`.
It is computed in FP32 and added only during training. Logs distinguish
`train/router_z_loss`, its weighted contribution and `train/token_nll`;
validation NLL and BPB exclude both router penalties.

The 3e17 preset weights text-token CE using an EMA of prior losses for the same target token ID.
All sampled tokens enter the forward pass and the existing text mask still applies to CE.
The first 290 updates use uniform weights, followed by a 290-update ramp to raw weights in [0.8, 1.2].
The trainer normalizes weighted CE within each microbatch before adding both router penalties.
The logs include the unweighted token NLL and the weighted CE separately. Start from scratch and
resume from this variant's own checkpoints to restore the EMA. The existing prediction-graph ledger
does not account for the extra loss-side weighting and EMA statistics.
Packed Qwen3 experts process every selected assignment without dropping.

SOAP with BF16 model weights, FP32 master weights, betas `(0.95, 0.95)` and
weight decay 0.1 retains the existing training path. The LR values are local
starting hypotheses, **not** values established by the paper, which uses Adam.
Warmup uses 2% and linear cooldown uses 20% of planned training FLOPs.
The 2K/4K/8K curriculum allocates 50%/20%/the remainder of compute, preserving
global tokens per update. Each tier's step count is recalculated independently.
Stage boundaries and ten milestones within the 8K stage receive validation
and rolling checkpoints; the pipeline exports and scores the final checkpoint.

Global batch is selected independently by tier. The two smaller recipes retain
more optimizer updates; the larger two try a doubled batch to reduce optimizer
and gradient-synchronization overhead. This is an unmeasured tradeoff, not a
BPB optimum. The paper uses `512 * 2048 = 1,048,576` tokens/update
([appendix A.3](https://arxiv.org/html/2605.11689#A3)), but does not establish
that batch as optimal for our data, SOAP optimizer or FLOPs budgets.

| Tier | Previous tokens/update | Selected tokens/update | Accumulation/rank | Previous updates | Selected updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3e17 | 262,144 | 262,144 | 4 | 8,716 | 8,716 |
| 1e18 | 524,288 | 524,288 | 8 | 6,006 | 6,006 |
| 3e18 | 524,288 | 1,048,576 | 16 | 9,069 | 4,534 |
| 1e19 | 524,288 | 1,048,576 | 16 | 10,750 | 5,375 |

Each rank still processes 8192 tokens per physical micro-batch: 4/2/1 sequences
at 2K/4K/8K. Larger accumulation does not itself increase the physical batch
or GPU utilization; activation memory is approximately unchanged. The same
total data still needs forward/backward passes. Fewer optimizer updates do
not imply proportional wall-time speedup. Router-count collectives still
occur for every micro-batch. Learning rate, betas, weight decay and SOAP
preconditioner frequency retain their current per-update values; the resulting
optimization trajectory changes and must be compared using validation BPB.

Warmup/cooldown, validation/checkpoint milestones, stage boundaries and the
complete-update FLOPs plan are regenerated for each chosen batch. The new
`3e18` plan has 4,754,243,584 training tokens after rounding whole updates;
`1e19` retains 5,636,096,000. Use a fresh output directory for a changed batch.

For a per-tier batch experiment, use `--tokens-per-update` (a multiple of
65536 for eight GPUs). This does not change the architecture family:

```bash
# Example: compare the previous 0.524M batch against the new 3e18 default.
python qwen3_moe/plan.py --tier 3e18 --tokens-per-update 524288 \
  --output-dir runs/configs-batch-comparison
TRAIN_CONFIG="$PWD/runs/configs-batch-comparison/3e18/training.yaml" \
  bash qwen3_moe/run_all.sh 3e18 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/3e18-batch524k"
```

## Generate and validate

Use the repository's pinned training environment (including Transformers
5.18.0 with packed 3D experts):

```bash
python qwen3_moe/plan.py --output-dir qwen3_moe/configs
python -m unittest discover -s qwen3_moe -p 'test_*.py'
python qwen3_moe/validate.py
```

Each `configs/TIER/` contains a stock HF `config.json`, the production
`training.yaml`, and `budget.json` with exact step/token counts and the
full-8K reference. Validation checks full model parameter counts on the meta
device and runs tiny CPU forward/backward/FP32 SOAP updates for each recipe.
This does not measure GPU throughput, peak memory or trained BPB.

## Competition-reference FLOPs accounting

The new protocol is `scicomp26-megatron-qwen3-moe-router-zloss-v2`. It invokes the
reviewed `num_floating_point_operations` snapshot directly, using MoE's
**activated** expert count and actual sequence length/batch size. The snapshot
and manifest are bundled under `training/src/flops_reference/`; its function
hash matches the repository's existing audit record. The supplied rules do
not pin a Git commit, so this is a reproducible reviewed snapshot, not a claim
that upstream `main` is frozen.

The base accounts for attention projections, causal QK/AV, active expert
SwiGLU projections and vocabulary logits. Each GEMM uses `2mnk`, with separate
input and weight gradients. Ordinary Transformer norm/activation/RoPE/
attention-softmax/LM-loss elementwise exclusions follow the reference.
No total-parameter `6N` approximation is used.

`training/src/qwen3_moe_flops.py` separately lists the routing graph's
projection, full expert softmax, top-k renormalization, weighted merge,
active-path gradient scatter, accumulated auxiliary loss and Router Z-loss
(stable logsumexp, square, reductions, coefficients and backward merges).
Its forward and backward are counted independently; the old
`32*(experts+top_k)` proxy is gone from this protocol. Integer counts, top-k
comparisons, indexing, copies and casts do not add floating-point arithmetic.
Optimizer updates, activation recomputation and communication are excluded;
precision does not change logical FLOPs.

Routing exp/log/divide are explicitly counted as one operation per call. **Their
weights are not specified by the supplied rules**; this convention is recorded
in every report and remains subject to organizer interpretation. Source-based
local accounting is distinct from organizer certification. Actual hardware
kernel padding/temporary zero buffers are not extra logical model operations.

Each `budget.json` contains the per-length operator tables for the real
micro-batch (4/2/1 sequences at 2K/4K/8K). Auxiliary scalar work is counted
once per micro-batch, then multiplied by accumulation and eight ranks. It is
not incorrectly multiplied once per sequence. Plans use complete updates
below the tier cap and within the supplied +/-1% interval.

After a complete pipeline run, `flops-report.json` is generated from the
continuous `train/metrics.jsonl` path. It verifies observed sequence lengths,
micro-batches, accumulation, tokens and cumulative FLOPs at every update.
Missing/duplicate steps, incomplete runs and unaccounted numerical retries
are rejected. Use its `actual_training_flops` for local curve fitting rather
than nominal tier labels. Failed attempts absent from logs require separate
review; this report does not certify unobserved execution.

The per-GPU persistent-state lower bound is recorded in `budget.json` and
includes replicated experts, BF16 gradients/DDP buckets, FP32 masters/gradients
and SOAP moments. Activations, preconditioners and optimizer temporaries are
additional. Eight-way DDP does not divide expert storage across the cards.
Measure a short run through all three lengths before a full training run;
`--stop-after 1` alone only measures the 2K stage.

## H800 kernels

Training and evaluation select local Hopper backends automatically:

* **CUTLASS grouped GEMM 0.2.0** replaces the Python loop over individual
  experts. Sorting, counts and dispatch stay on GPU; one grouped gate/up GEMM
  and one grouped down GEMM retain the existing packed parameters. No tokens
  are dropped. Optional `torch.compile` fuses only the SwiGLU pointwise work,
  including its backward; the dynamic router is not compiled as a whole.
* **FlashAttention-3** is selected when a compatible local `flash-attn-3`
  installation passes a BF16 causal GQA forward/backward check. Otherwise
  attention uses the existing PyTorch SDPA FlashAttention backend. There is no
  kernel-hub download during training. BF16 and FP32 master weights remain.
* Qwen DDP uses **gradient bucket views**, sharing BF16 gradient/bucket storage
  after bucket construction. The persistent-state estimate becomes 20 rather
  than 22 bytes/parameter; this is not a peak-memory estimate.
* Evaluation defaults to **Triton RMSNorm fusion**, including Q/K head norms.
  It preserves Qwen's cast-before-weight multiplication and falls back to
  stock autograd when gradients are enabled. Gate/up weights and HF state-dict
  names are unchanged by the Qwen kernel installers.

Auto modes run small numerical/gradient checks before installing native
kernels and report the selected backends and fallback reasons. These appear
in `train/run-plan.json` and `eval-result.json`; exact resume rejects a changed
backend. The updated source hashes/configs require a fresh run. Mathematical
FLOPs and parameter counts are independent of the selected kernel. BF16
rounding/order can differ; BPB must be measured with the chosen evaluator.

Install the expert extension **in the pinned training environment**, either
from a matching Python/PyTorch/CUDA/ABI wheel or by compiling for SM90:

```bash
bash qwen3_moe/install_h800.sh /path/to/grouped_gemm-0.2.0-cp311-cp311-linux_x86_64.whl
# Or build the pinned source with nvcc/CUDA 12.4; needs a devel environment,
# network and build tools. A CUDA runtime image alone does not include nvcc.
bash qwen3_moe/install_h800.sh

# Run on an H800; repeat for other tiers to exercise their expert dimensions.
python qwen3_moe/verify_h800.py --tier 3e17 --output /tmp/h800-3e17.json
```

The verifier compares stock/grouped outputs, input/weight/routing gradients,
unused-expert zeros and fused RMSNorm; it times isolated expert forward and
forward/backward. It does not measure full training or SOAP/DDP overhead.
FlashAttention-3 is optional and must be installed separately for the pinned
PyTorch/CUDA environment; see its [official Hopper instructions](https://github.com/Dao-AILab/flash-attention/tree/main/hopper).

`QWEN_MOE_EXPERTS=grouped_gemm` and
`QWEN_MOE_ATTENTION=flash_attention_3` require the requested kernels and fail
at startup if unavailable. To reproduce the eager-expert/SDPA baseline:

```bash
QWEN_MOE_EXPERTS=eager QWEN_MOE_ATTENTION=sdpa QWEN_MOE_COMPILE_SWIGLU=0 \
EVAL_FUSION_MODE=none bash qwen3_moe/run_all.sh 3e17 TRAIN_CACHE TOKENIZER EVAL_CACHE NEW_OUTPUT
```

The development environment has no usable GPU. CPU output/gradient and FLOPs
checks used Python 3.10, torch 2.8.0 and the pinned Transformers 5.18.0; the target training
environment remains torch 2.6.0+cu124. Native SM90 kernels, throughput and peak
memory still need verification on the H800. No speedup or BPB improvement is
claimed yet.

The required environment files (`pyproject.toml`, `requirements.txt`, `uv.lock`)
are unchanged from `qwen3-moe`: Python 3.11, torch 2.6.0+cu124, Transformers
5.18.0, tokenizers 0.23.1 and Triton 3.2.0. Optional native extensions must
match Python, torch, CUDA and C++ ABI. Missing/incompatible optional extensions
fall back in auto mode; they are not required to use the baseline kernels.
The acceleration report stores the torch version as a plain string so that
native checkpoints can be read by `torch.load(weights_only=True)` on resume.
This fix updates the recorded source/family hashes; a run already started
under an older revision retains that revision's declaration. Do not combine
its checkpoint with the new generated settings for exact resume.

## Train, export and evaluate

For **3e17 only**, the dedicated wrapper fixes the paper-scaling configuration
and uses eight GPUs. Its three cache/tokenizer arguments match the archived
A20M launcher; an optional fourth argument specifies a new output directory.
Without it, the wrapper prints a unique directory under `runs/`. It completes
all 8,716 optimizer updates, CPU HF export, packed BPB evaluation and the
successful-training-path FLOPs audit:

```bash
git fetch origin
git switch qwen3-moe-paper-scaling
git pull --ff-only origin qwen3-moe-paper-scaling

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EVAL_GPU=0 EVAL_BATCH_SIZE=64 \
bash qwen3_moe/run_3e17.sh \
  /mnt/token_cache_24k_v1_full_20261101 \
  /mnt/tok_bpe24k_v1/tokenizer \
  /mnt/1103_data/tok_bpe24k_v1_dolma3_mix_100M_a_npy
```

`EVAL_BATCH_SIZE=64` applies to evaluation only. The wrapper also accepts a
fourth path, for example `/mnt/runs/qwen3-moe-paper-3e17-001`; it must not
already exist. Results appear in `train/`, `hf-model/`, `eval-result.json`
and `flops-report.json` under the printed directory. The shared pipeline
checks the caches and tokenizer before starting training.
Invalid evaluation batch sizes and fusion modes are also rejected before
training, rather than after completing the training/export pipeline.

Run each tier with a new output directory. All four commands use **24K**
training/evaluation caches and the matching tokenizer, including `1e19`.

```bash
bash qwen3_moe/run_all.sh 3e17 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/3e17"
bash qwen3_moe/run_all.sh 1e18 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/1e18"
bash qwen3_moe/run_all.sh 3e18 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/3e18"
bash qwen3_moe/run_all.sh 1e19 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/1e19"
```

These commands use eight GPUs, FP32 SOAP training, CPU HF export and one-GPU
8K packed BPB evaluation with Hopper backend selection and RMSNorm fusion. Evaluation batches
default to 8/4/1/1; `EVAL_BATCH_SIZE` can override this after a memory check.
The result is `OUTPUT/eval-result.json`. Evaluation batch size changes memory
and throughput, not the intended scoring context or byte denominator.

For training alone, smoke runs or resumption from a checkpoint:

```bash
TRAIN_CONFIG="$PWD/qwen3_moe/configs/3e17/training.yaml" \
  bash qwen3_moe_a30m/train.sh "$TRAIN_CACHE" "$RUNS/3e17/train" --stop-after 2

TRAIN_CONFIG="$PWD/qwen3_moe/configs/3e17/training.yaml" \
  bash qwen3_moe_a30m/train.sh "$TRAIN_CACHE" "$RUNS/3e17/train" \
  --resume "$RUNS/3e17/train/step-2.pt"
```

## Select recipes by the four-tier score

The supplied competition rules fit the four observations **separately for
tracks A and B**. Use reviewed actual training FLOPs `C_i` and measured BPB
`L_i`, rather than nominal tier caps or token cross-entropy:

```text
L_hat(C) = E + k / C**p
fit objective = sum_i [ln L_hat(C_i) - ln L_i]**2   (equal weights)
R_squared = 1 - sum_i [ln L_i - ln L_hat(C_i)]**2 / sum_i [ln L_i - mean(ln L)]**2
A_front = mean of L_hat(C) under d ln C, from C_min=3e17 to C_max=1e19
Performance = A_front / R_squared                  (lower is better)
```

The integration averages **BPB itself**, not log BPB. A lower BPB at a single
tier does not by itself establish a better overall Performance score. Compare
whole four-tier families before choosing a final configuration set.

The provided reference evaluator confirms packed documents, unscored inserted
EOS, accumulated text-token NLL and normalization by original UTF-8 bytes:
`BPB = total_nll_nats / (ln(2) * original_utf8_bytes)`. Its EvalTime normalizes
forward seconds by non-vocabulary active parameters and scored tokens; EvalTime
is not actual training FLOPs. It contains no scaling-fit implementation or
parameter bounds. The existing evaluator uses this packing and BPB convention.

The local scoring utility is separate from training:

```bash
python -m pip install -r evaluation/requirements-scaling.txt
cp evaluation/scaling_points.example.json /path/to/four_points.json
# Fill in measured bpb, reviewed actual_training_flops and each run's architecture_family_id.
python evaluation/score_scaling.py --points /path/to/four_points.json --output /path/to/scaling-score.json
python -m unittest discover -s evaluation -p test_score_scaling.py
```

Set `track` to A or B and run separately for each track. The template contains
nulls to prevent planned or invented observations from becoming a score.
Copy each family ID from its run configuration or budget record; inconsistent
or partially supplied IDs are rejected. If all IDs are omitted, the numerical
fit is available but family consistency is explicitly marked as unchecked.
The fitter uses a stable equivalent parameterization `E + a*(C/C_ref)**(-p)`
and multiple initializations. Until official bounds or fitting code are
provided, **E >= 0, k > 0, p > 0 are local assumptions**; outputs are marked as
local reproductions rather than official scores. Undefined/nonpositive
log-space R-squared does not produce a numeric Performance score.

## Compare architecture and optimizer candidates

Architecture alternatives are generated as **whole four-tier families**.
`--granularity-shift` moves all scales toward coarser or finer experts, while
keeping the activated FFN width at `4*hidden`. Each family retains the same
generation rule and ID across its four models. Legal expert counts and router
costs are recalculated, so total capacity may vary slightly at the integer
activation boundary. For example:

```bash
python qwen3_moe/plan.py --granularity-shift -1 --output-dir /path/to/candidates/coarser
TRAIN_CONFIG=/path/to/candidates/coarser/3e17/training.yaml \
  bash qwen3_moe/run_all.sh 3e17 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/3e17-coarser"
# Repeat for the other three tiers from this same generated family.
```

The default already maximizes legal inactive capacity. Compare a smaller
capacity fraction or aligned expert counts if they improve optimization or
H800 throughput. These choices also apply as one declared family rule.
Learning rate may vary independently per tier:

```bash
python qwen3_moe/plan.py --granularity-shift 1 --output-dir /path/to/candidates/finer
python qwen3_moe/plan.py --expert-capacity-fraction 0.85 --output-dir /path/to/candidates/capacity
python qwen3_moe/plan.py --expert-alignment 8 --output-dir /path/to/candidates/aligned
python qwen3_moe/plan.py --tier 1e18 --learning-rate 0.003 --output-dir /path/to/candidates/lr
```

Each candidate has its own complete compute cap. Sweep compute is additional
and is not included in the cost of any single candidate. Hold data order,
seed and tokenizer fixed; use held-out development BPB at the same 8K context
and the four-tier Performance score to select architecture/LR, and reserve
the final scoring set for the selected recipe. CE during 2K/4K stages is a
stability diagnostic, not an 8K BPB result.
Do not combine points from different architecture families into a submitted
four-tier scaling curve.
