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
| 3e18 | 16 / 512 | 8 / 2 / 64 | 47 / 4 / 512 | 1/4 | 0.0015 | 524,288 |
| 1e19 | 24 / 768 | 12 / 2 / 64 | 94 / 8 / 384 | 1/8 | 0.001 | 524,288 |

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
Stock packed Qwen3 experts process every selected assignment without dropping.

SOAP with BF16 model weights, FP32 master weights, betas `(0.95, 0.95)` and
weight decay 0.1 retains the existing training path. The LR values are local
starting hypotheses, **not** values established by the paper, which uses Adam.
Warmup uses 2% and linear cooldown uses 20% of planned training FLOPs.
The 2K/4K/8K curriculum allocates 50%/20%/the remainder of compute, preserving
global tokens per update. Each tier's step count is recalculated independently.
Stage boundaries and ten milestones within the 8K stage receive validation
and rolling checkpoints; the pipeline exports and scores the final checkpoint.

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

The new protocol is `scicomp26-megatron-qwen3-moe-router-v1`. It invokes the
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
active-path gradient scatter and the runner's accumulated auxiliary loss.
Its forward and backward are counted independently; the old
`32*(experts+top_k)` proxy is gone from this protocol. Integer counts, top-k
comparisons, indexing, copies and casts do not add floating-point arithmetic.
Optimizer updates, activation recomputation and communication are excluded;
precision does not change logical FLOPs.

Routing exp/divide are explicitly counted as one operation per call. **Their
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

## Train, export and evaluate

Run each tier with a new output directory. All four commands use **24K**
training/evaluation caches and the matching tokenizer, including `1e19`.

```bash
bash qwen3_moe/run_all.sh 3e17 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/3e17"
bash qwen3_moe/run_all.sh 1e18 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/1e18"
bash qwen3_moe/run_all.sh 3e18 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/3e18"
bash qwen3_moe/run_all.sh 1e19 "$TRAIN_CACHE" "$TOKENIZER_DIR" "$EVAL_CACHE" "$RUNS/1e19"
```

These commands use eight GPUs, FP32 SOAP training, CPU HF export and one-GPU
8K packed BPB evaluation with fusion disabled for MoE. Evaluation batches
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
