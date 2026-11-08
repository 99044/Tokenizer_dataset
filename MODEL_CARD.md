# Model Card

`submission_1108` defines three OLMo hybrid configurations. Each uses the repeating `[gdn, gdn, gdn, attn]` block pattern, a Dense SwiGLU FFN in each block, and a causal compressed K/V branch beside the final Attention layer. The branch output is added to the original Attention output at the configured scale.

| FLOPs tier | Total parameters | Layers | Hidden / FFN | Tokenizer vocabulary | Optimizer |
| --- | ---: | ---: | ---: | ---: | --- |
| 3e17 | 47,756,790 | 12 | 256 / 1984 | 24,576 | SOAP plus AdamW for the compressed branch |
| 3e18 | 200,197,018 | 12 | 640 / 2048 | 24,576 | SOAP |
| 1e19 | 1,171,647,936 | 16 | 1024 / 4352 | 32,768 | Muon plus AdamW for embeddings, vectors, and the two large K/V compression matrices |

The 3e17 configuration is carried from `submission_20261107`. The 3e18 architecture and optimizer are from `submission_20261103`, adapted to eight data-parallel GPUs at the same global tokens per update. The 1e19 architecture and optimizer grouping reproduce the completed `wide_330m_block388_plateau_1e19_8gpu_muon` run. Configuration and launch details are in [REPRODUCTION.md](REPRODUCTION.md).
