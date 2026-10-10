# FLOPs accounting references

These source snapshots are read as evidence by the FLOPs calculator. They are
never imported by training and do not replace the installed model or kernels.
`manifest.json` records their paths, origins and SHA-256 hashes.

- OLMo-core 2.6.0: sources copied from `scicomp26:submission-env-v1`, under the
  Apache 2.0 license in `OLMO_LICENSE`.
- Flash Linear Attention: commit
  `3cf180339b8a1cbad823f553541cd531d18670ea`, referenced by OLMo-core's GDN
  implementation, under the MIT license in `FLA_LICENSE`.

`wy_fast.py` includes the reviewed Triton 3.2 compatibility patch in
`training/src/patch_fla_torch26.py`: multiply before transposing in the backward
kernel. The manifest records both upstream and patched hashes. This preserves
the mathematical graph and operation count; the runtime must match the patched
reference rather than silently accepting an arbitrary kernel change.

The automatic Hybrid calculation uses the chunk-64 logical tensor graph. It
counts projections, depthwise convolution, WY representation, triangular
inverse, recurrent state updates/readout, their separate derivatives, and the
additional detached signal scan. Standard attention uses Megatron's causal
`S²/2` convention. GDN's masked-after-matmul chunk operations stay dense.

Saved forward intermediates may be reused when deriving backward FLOPs.
FLA's memory-saving reconstruction of WY/state/attention intermediates and the
convolution preactivation is excluded, as required by the supplied rules.
Warp tiling, tensor-core padding, repeated hardware-local scalar calculations,
tensor copies, casts, comparisons and indexing do not change this logical
graph. Loss computation and diagnostic statistics are outside prediction.

The rule supplied by the user fixes addition/multiplication and GEMM costs,
but does not specify exp/sqrt/division/log costs. `--scalar-op-cost` must be
explicit, and is recorded in the result. Sigmoid is decomposed into negation,
exp, addition and reciprocal; softplus uses its unbranched exp/add/log reference
definition. This convention, RMSNorm's saved-activation logical derivative and
Megatron's causal approximation are disclosed counting conventions, not a claim
of a hardware instruction count or an official organizer certification.

`--plan-only` produces a complete reference calculation without importing any
model or launching a kernel. Actual-training automatic accounting additionally
requires the trainer's `runtime_graph-start-<n>.json` for the initial run and every resume
segment (repeat `--runtime-metadata`): model/kernel source hashes must
match these references, all parameters must be trainable, and all 36 short
convolutions must use the reviewed bias-free Triton/SiLU graph. A different FLA
version, backend, frozen projection, recurrence or routing graph requires a
separately reviewed `--operator-ledger`; it cannot silently reuse this formula.
