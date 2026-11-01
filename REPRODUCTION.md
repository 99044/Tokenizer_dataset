# 训练与导出

本分支 `submission_20261103` 提供四档 24K 词表、8192 上下文的宽浅 Hybrid 配置。四档都采用 12 层 `[gdn, gdn, gdn, attn]`、Dense SwiGLU，并在第 12 层全局 Attention 旁接 128-token 因果 K/V 压缩分支，输出系数固定为 `0.001`。原 Attention 输出保留。

| 配置 | 总参数 | hidden | FFN | heads | 优化器 | 步数 | tokens/update | 计划训练 tokens |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `FLOPs_3e17.yaml` | 47,756,772 | 256 | 1984 | 2 | SOAP | 10,090 | 131,072 | 1,322,516,480 |
| `FLOPs_1e18.yaml` | 200,197,018 | 640 | 2048 | 5 | SOAP | 7,512 | 196,608 | 1,476,919,296 |
| `FLOPs_3e18.yaml` | 200,197,018 | 640 | 2048 | 5 | SOAP | 16,902 | 262,144 | 4,430,757,888 |
| `FLOPs_1e19.yaml` | 1,175,476,632 | 1536 | 6144 | 12 | SOAP | 5,346 | 524,288 | 2,802,843,648 |

实际步数和 token 以 YAML 的 `training` 与 `budget` 字段为准。四档按 4 卡 DDP、每卡 micro batch 1 配置，梯度累积分别为 4、6、8、16。模型使用 BF16，四档均采用 SOAP；学习率统一为 0.003、warmup 为 20 步、末端学习率为 0，沿用首次提交配方。初始化时主干指定投影和 embedding 权重乘以 0.8；压缩分支的输出另乘 0.001。每档训练进度约每 10% 保存一次，自动只保留最近两个完整 checkpoint。中断后用相同 YAML、数据和 GPU 数量传 `--resume` 继续。

`training/generate_submission_20261103_configs.py` 可重新生成四个 YAML；`python training/validate_submission_20261103_configs.py` 检查配置与评测默认值。预算由原 Hybrid 逐算子 ledger 与新增压缩分支的保守三倍前向估计相加，计划 FLOPs 留约 1% 余量；该估计不是压缩分支完整逐算子审计。`training/count_flops.py --auto-hybrid` 对此分支会拒绝把缺少压缩分支的 ledger 当成完整核算。

训练数据目录需包含 `metadata.json`、`train.npy`、`validation.npy`，其 tokenizer 词表大小必须为 24576。若使用现有 24K 缓存，先核对目录结构与 `metadata.json`；训练器会检查词表及上下文长度。启动示例：

```bash
bash training/train.sh --image scicomp26:submission-env-hybrid-v3 \
  --config training/configs/FLOPs_3e17.yaml --stage stage1 --gpus 0,1,2,3 \
  --data-dir /path/to/24k-tokenized-data --data-format packed \
  --run-name submission_20261103_3e17
```

训练其他档位时只需替换 `--config` 与 `--run-name`。

将原生 checkpoint 导出为离线 HF 模型，导出时在 CUDA 上进行原生与 HF 的权重及 logits 核对：

```bash
python training/export_hf.py --checkpoint /path/to/step-N.pt \
  --tokenizer-dir /path/to/24k-tokenizer --output-dir /path/to/hf-model \
  --validation-device cuda:0
```

评测从 HF 模型加载。`FFN＋RMSNorm` 融合和每卡 batch 32 是默认值，可通过环境变量调整：

```bash
NPROC_PER_NODE=1 BATCH_SIZE=32 FUSION_MODE=ffn_rmsnorm \
  bash evaluation/eval.sh /path/to/dev.jsonl /path/to/hf-model /path/to/result.json
```

评测包含完整 dev 集 BPB 和按非词表参数量归一化的 EvalTime。融合需要 BF16 CUDA 与 Triton；`FUSION_MODE=none` 可用于同一 HF 权重的标准路径对照。batch 32、8192 上下文在较大模型上需要较多显存，若设备无法容纳，应显式覆盖 `BATCH_SIZE` 并在结果中记录该值。当前分支只完成代码与静态配置检查，GPU 训练、HF 导出和实际评测仍需在运行环境执行。
