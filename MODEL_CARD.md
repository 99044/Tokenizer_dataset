# Model Card

本分支 `submission_20261103` 定义四个 OLMo Hybrid 模型，使用 24576 词表、8192 上下文和 BF16。各模型均为 12 层，按三层 GDN、一层全局 Attention 排列；每层保留 Dense SwiGLU FFN，最后一层 Attention 旁接 128-token 因果 K/V 压缩分支，输出按 0.001 加权。

| 档位 | 总参数 | 非词表参数 | hidden | FFN | attention heads | 优化器 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3e17 | 47,756,772 | 41,465,316 | 256 | 1984 | 2 | SOAP |
| 1e18 | 200,197,018 | 184,468,378 | 640 | 2048 | 5 | SOAP |
| 3e18 | 200,197,018 | 184,468,378 | 640 | 2048 | 5 | SOAP |
| 1e19 | 1,175,476,632 | 1,137,727,896 | 1536 | 6144 | 12 | SOAP |

GDN 使用每头 K96/V192、4 宽卷积；Attention 每头 128 维。模型采用 tied embeddings、RMSNorm、RoPE、QK Norm 和零 dropout。所有压缩分支参数参与训练。主干指定投影及 embedding 使用 0.8 初始化缩放。四档由 `training/configs/FLOPs_*.yaml` 定义；训练、HF 导出及评测入口见 `REPRODUCTION.md`。
