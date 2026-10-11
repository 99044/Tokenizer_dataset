# OLMo Hybrid scaling 阶段 B

阶段 B 只运行 B1–B3：延长 330M 的 constant-LR 曲线，并用 190M、250M 的中间 checkpoint 检查 cooldown 是否能改善末期 BPB。三组实验继续使用阶段 A 的同一份 token cache、tokenizer、验证集和随机种子。

## 计划

| 实验 | 配置 | Stage A checkpoint | 新增 steps | 结束 step | 新增 tokens | 新增训练 FLOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B1 | `b1_hybrid_330m_constant.yaml` | 330M step 258 | 153 | 411 | 80,216,064 | 178,887,698,756,628,480 |
| B2 | `b2_hybrid_190m_cooldown.yaml` | 190M step 382 | 77 | 459 | 40,370,176 | 54,178,083,666,646,016 |
| B3 | `b3_hybrid_250m_cooldown.yaml` | 250M step 287 | 58 | 345 | 30,408,704 | 53,473,987,808,216,064 |

阶段 B 合计 `286,539,770,231,490,560` 训练 FLOPs。加上阶段 A 后为 `1,974,464,614,582,138,880`，距离 `2e18` 还剩 `25,535,385,417,861,120` FLOPs。

三个实验都恢复模型、AdamW optimizer 和各 rank 的 RNG 状态，不重新 warmup。B1 保留原 constant scheduler 状态，学习率继续保持 `2.65e-4`。B2 和 B3 仅重建 scheduler，分别在 77 和 58 个新增 step 内将学习率线性降至 0；optimizer 的动量和方差状态不重置。

## 启动

阶段 A 实际使用 4 张 GPU、`micro_batch_size=1`、`gradient_accumulation_steps=16`，每次更新 524,288 tokens。Stage B 必须保持这三个运行参数，才能恢复四个 rank 的 RNG 状态并保持训练 token 口径。YAML 中保留两卡基线的 `1 × 32`；在四卡服务器上通过命令行覆盖为 `1 × 16`。

先检查三个配置的参数量、token 和 FLOPs：

```bash
python training/validate_scaling_stage_b_configs.py
```

使用阶段 A 相同的 `DATA_DIR`，分别启动三个独立 run：

```bash
bash training/train.sh \
  --stage stage-b \
  --gpus 0,1,2,3 \
  --grad-accum-steps 16 \
  --config training/configs/scaling_stage_b/b1_hybrid_330m_constant.yaml \
  --data-dir "$DATA_DIR" \
  --branch-from /path/to/330m/step-258.pt \
  --run-name scaling-b1-330m-constant \
  --wandb

bash training/train.sh \
  --stage stage-b \
  --gpus 0,1,2,3 \
  --grad-accum-steps 16 \
  --config training/configs/scaling_stage_b/b2_hybrid_190m_cooldown.yaml \
  --data-dir "$DATA_DIR" \
  --branch-from /path/to/190m/step-382.pt \
  --run-name scaling-b2-190m-cooldown \
  --wandb

bash training/train.sh \
  --stage stage-b \
  --gpus 0,1,2,3 \
  --grad-accum-steps 16 \
  --config training/configs/scaling_stage_b/b3_hybrid_250m_cooldown.yaml \
  --data-dir "$DATA_DIR" \
  --branch-from /path/to/250m/step-287.pt \
  --run-name scaling-b3-250m-cooldown \
  --wandb
```

配置会额外保存需要计算 BPB 的 checkpoint：B1 保存 step 306、354、382、411；B2 保存 step 420、459；B3 保存 step 325、345。中断后的 Stage B run 使用同一配置和 `--resume` 恢复自身 checkpoint，不再使用 `--branch-from`。

## 结果使用

B1 的 constant-LR 点用于补齐规模曲线，并参与 scaling-law 重拟合。B2、B3 用于判断 cooldown 对末期 loss 的影响，不与 constant-LR 点混在同一个拟合中。若重拟合仍无法得到稳定的正向规模收益，则按相同 FLOPs 下的 BPB 在 190M、250M、330M 中做实验范围内的选择，并在记录中说明没有得到可信的解析 scaling-law 最优点。
