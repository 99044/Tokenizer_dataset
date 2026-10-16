# SCICOMP26 Model Card

## 1. 队伍与架构

- 队伍名称及编号：<待填写>
- 提交版本：`submission_20261017_soap`
- 架构名称：OLMo Hybrid scaling family
- 代码入口：模型构建位于 `training/src/model_config.py`，四卡训练位于
  `training/src/train_ddp.py`，四档配置位于 `training/configs/FLOPs_*.yaml`。
- 每四层采用 `[gdn, gdn, gdn, attn]`；所有块保留 Dense SwiGLU FFN，
  attention 层使用全局因果注意力。
- 公共设置：32768 词表、8192 上下文、BF16、RMSNorm、RoPE、QK Norm、
  tied embeddings、无 dropout。

四个 FLOPs 档使用同一家族的不同模型规模：

| model_id | 标称规模 | 实际参数量 | 层数 | hidden | FFN | attention/KV heads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOPs_3e17 | 110M | 107,135,898 | 12 | 640 | 2560 | 5 |
| FLOPs_1e18 | 190M | 190,571,664 | 16 | 768 | 3072 | 6 |
| FLOPs_3e18 | 250M | 254,467,368 | 16 | 896 | 3584 | 7 |
| FLOPs_1e19 | 330M | 327,544,256 | 16 | 1024 | 4096 | 8 |

所有 attention head dimension 为 128。GDN head dimension 为 96，value expand 为 2.0，
conv size 为 4；GDN heads/value heads 与对应档位的 attention heads 相同。

## 2. 优化器与训练计划

四档统一使用 SOAP：peak LR `0.003`、betas `(0.95, 0.95)`、eps `1e-8`、
weight decay `0.1`、precondition frequency `10`、max precondition dimension `1024`。
该配置来自 250M 优化器第一阶段中 BPB 最低的 O12。各档 warmup 20 步，最后约
16.8% 的更新执行 linear cooldown 到 0。

训练使用 4 张 GPU，每卡 `micro_batch_size=1`、梯度累积 16 次：

```text
4 × 1 × 16 × 8192 = 524,288 tokens/update
```

| model_id | 目标预算 | 计划训练 FLOPs | 更新步数 | 累计训练 token | cooldown |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLOPs_3e17 | 300,000,000,000,000,000 | 299,965,518,274,982,144 | 746 | 391,118,848 | 125 |
| FLOPs_1e18 | 1,000,000,000,000,000,000 | 999,831,907,666,285,568 | 1,421 | 745,013,248 | 239 |
| FLOPs_3e18 | 3,000,000,000,000,000,000 | 2,999,153,143,795,290,624 | 3,253 | 1,705,508,864 | 547 |
| FLOPs_1e19 | 10,000,000,000,000,000,000 | 9,999,003,920,043,704,320 | 8,552 | 4,483,710,976 | 1,438 |

计划使用完整 Hybrid 逐算子 ledger、`scalar-op-cost=1`。每档选择预算内可完成的最大
整数 update，因此计划 FLOPs 不超过目标，且再增加一步就会超过预算。

每档在训练进度 10%、20%……90% 保存中间 checkpoint，并在 100% 保存最终 checkpoint，
共 10 个。相同节点执行固定验证，最终节点使用 64 batches。空间不足时，最终 checkpoint
及对应评测确认完成后，可以删除该 run 的 10%–90% 中间 checkpoint。

## 3. 参数与数据状态

| 标称规模 | 非词表参数量 |
| --- | ---: |
| 110M | 86,164,378 |
| 190M | 165,405,840 |
| 250M | 225,107,240 |
| 330M | 293,989,824 |

当前没有 MoE 或 token 级稀疏路由，所有非词表参数均参与每个 token 的计算。四档复用
`tok_bpe32k_v1` tokenizer 和同一训练数据接口，均从随机初始化开始训练。

四档静态一致性使用以下命令检查：

```bash
python training/validate_submission_20261017_configs.py
```

最终提交时以实际完成的 checkpoint、训练日志、BPB 结果和导出的离线模型目录为准。
