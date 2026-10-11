# OLMo Hybrid 多规模短程预训练

这组配置用于阶段 A 的 scaling 探索，不替代正式四档 FLOPs 配置。六个模型沿用 `olmo_hybrid_1b` 的结构规则：每四层为 `GDN/GDN/GDN/Attention`，所有块保留 Dense SwiGLU FFN，Attention 为全局注意力，context length 为 8192，词表固定为 32,768。

## 固定条件

- 训练硬件基线：2 张 GPU。
- `micro_batch_size=1`，`gradient_accumulation_steps=32`。
- Global batch：64 条序列，即每次参数更新 524,288 tokens。
- AdamW：`betas=(0.9, 0.95)`、`eps=1e-8`、`weight_decay=0.1`。
- 主训练使用 constant LR，前 20 step（10,485,760 tokens）warmup。
- 每 24 step 在固定随机种子下计算 8 个 batch 的 token NLL；每 48 step（25,165,824 tokens）保存 checkpoint。
- 六个 run 必须使用相同 tokenizer、相同训练数据顺序及同一份文档级隔离验证集。不同规模不能各自 shuffle。

配置名按近似规模命名，拟合时必须使用实际参数量，不可把文件名中的规模代入：

| 配置 | 层数 | hidden | heads | 实际参数量 | 计划 tokens | steps | 逐算子计划 FLOPs | peak LR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_80m.yaml` | 8 | 640 | 5 | 78,414,652 | 400,031,744 | 763 | 221,314,213,567,689,984 | 3.35e-4 |
| `hybrid_110m.yaml` | 12 | 640 | 5 | 107,135,898 | 300,417,024 | 573 | 230,402,469,130,783,872 | 3.35e-4 |
| `hybrid_190m.yaml` | 16 | 768 | 6 | 190,571,664 | 220,200,960 | 420 | 295,516,819,999,887,360 | 3.06e-4 |
| `hybrid_250m.yaml` | 16 | 896 | 7 | 254,467,368 | 170,393,600 | 325 | 299,638,724,787,417,600 | 2.83e-4 |
| `hybrid_330m.yaml` | 16 | 1024 | 8 | 327,544,256 | 135,266,304 | 258 | 301,653,766,530,785,280 | 2.65e-4 |
| `hybrid_500m.yaml` | 20 | 1152 | 9 | 502,815,438 | 100,139,008 | 191 | 339,398,850,334,084,224 | 2.49e-4 |

阶段 A 合计 `1,687,924,844,350,648,320` FLOPs。该值来自 Hybrid 完整逐算子 ledger，`scalar-op-cost=1`；它比 `6ND` 近似值高，因此实验记录和后续拟合应使用表内值。

peak LR 使用 `2e-4 * sqrt(1792 / hidden_size)` 后取整。这只是首轮规模缩放规则。正式投入六个 run 前，先用 110M、250M、500M 各跑 20–40 step，确认无 loss spike 或梯度异常；若修改 LR，六份配置的规则和修改原因必须一起记录。

## 启动与检查

先验证所有 YAML 的架构关系、参数量、batch、步数和逐算子 FLOPs：

```bash
python training/validate_scaling_configs.py
```

单个实验的启动方式如下；配置已经给出 `max_steps`、batch、LR、warmup、验证和保存频率：

```bash
torchrun --standalone --nproc-per-node=2 training/src/train_ddp.py \
  --config training/configs/scaling/hybrid_250m.yaml \
  --data-dir /path/to/token-cache \
  --output-dir runs/scaling/olmo_hybrid_scaling_250m \
  --wandb
```

`validation/token_nll` 是固定 token 样本上的平均自然对数损失，只用于监控训练。它不是 BPB，不能作为 scaling law 的目标值。

## BPB

BPB 沿用 `evaluation/eval.py` 的定义与打包逻辑：每篇文档前插入一个不计分 EOS，所有原始文本 token 计分，分母为原始文本的 UTF-8 字节数：

\[
\mathrm{BPB}=\frac{\mathrm{NLL}_{\mathrm{nats}}}{\ln(2)\,\mathrm{UTF8Bytes}}.
\]

Hybrid 中间 checkpoint 尚未导出为 Hugging Face 目录时，直接运行原生 checkpoint 入口。它复用 `evaluation/eval.py` 的 `pack_jsonl` 和 `packed_batch`，因此与最终评测使用相同的字节和 mask 口径：

```bash
torchrun --standalone --nproc-per-node=2 evaluation/eval_native_checkpoint.py \
  --dataset-path /path/to/heldout.jsonl \
  --checkpoint runs/scaling/olmo_hybrid_scaling_250m/step-48.pt \
  --tokenizer-dir /path/to/tokenizer \
  --output runs/scaling/olmo_hybrid_scaling_250m/bpb-step-48.json
```

每个输出 JSON 已包含拟合需要的 `run_id`、`N_total`、`N_nonembed`、`D`、`FLOPs`、`bpb`、学习率、schedule 和 seed。同一个 checkpoint 输出文件不可覆盖。

阶段 A 完成后再根据 BPB 曲线决定阶段 B：延长哪个中等规模模型，以及从哪些 checkpoint 分支 cooldown。阶段 B 不应在看到阶段 A 结果前固定成六个模型都继续训练。
