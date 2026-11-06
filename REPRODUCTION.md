# 训练与导出

本分支 `submission_20261103` 提供四档 24K 词表、8192 上下文的宽浅 Hybrid 配置。四档都采用 12 层 `[gdn, gdn, gdn, attn]`、Dense SwiGLU，并在第 12 层全局 Attention 旁接 128-token 因果 K/V 压缩分支。3e17 档输出系数为 `0.0001`，其余三档为 `0.001`；原 Attention 输出保留。

| 配置 | 总参数 | hidden | FFN | heads | 优化器 | 步数 | tokens/update | 计划训练 tokens |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `FLOPs_3e17.yaml` | 47,756,790 | 256 | 1984 | 2 | SOAP | 5,502 | 262,144 | 1,442,316,288 |
| `FLOPs_1e18.yaml` | 200,197,018 | 640 | 2048 | 5 | SOAP | 8,072 | 196,608 | 1,587,019,776 |
| `FLOPs_3e18.yaml` | 200,197,018 | 640 | 2048 | 5 | SOAP | 18,162 | 262,144 | 4,761,059,328 |
| `FLOPs_1e19.yaml` | 1,175,476,632 | 1536 | 6144 | 12 | Muon | 5,346 | 524,288 | 2,802,843,648 |

实际步数和 token 以 YAML 的 `training` 与 `budget` 字段为准。四档按 8 卡 DDP 配置，梯度累积分别为 4、3、4、8。8192 上下文时每卡 micro batch 为 1。模型使用 BF16 计算，前三档主干采用 SOAP，1e19 档主干采用 Muon，四档压缩分支均采用 AdamW，峰值学习率为 0.003。3e17 档采用约 5% warmup、保持峰值学习率到 70% 进度、最后 30% 线性降至 3e-5。1e19 档采用 20 步 warmup、最后约 16.7% 线性降至 0。初始化时主干指定投影和 embedding 权重乘以 0.8，压缩分支的输出另乘对应 YAML 中的系数。每档训练进度约每 10% 保存一次，自动只保留最近两个完整 checkpoint。中断后用相同 YAML、数据和 GPU 数量传 `--resume` 继续。1e19 档更换优化器后必须从头训练。

3e17 采用 `scicomp26-small_model_opt_E16_dual_length_fp32_4gpu_20261107` 的策略。第 9 至 12 层使用持续双流残差，初始交叉混合比例为 0.25，每行总增益为 2，末层初始合并权重各为 0.5。第 8、12 层 Attention 将当前 V 与第 4 层 V 按各 0.5 混合，梯度回传至第 4 层。新增 18 个 FP32 混合参数，与全部可训练参数一起纳入 FP32 主权重和优化器。训练、原生评测及 HF 导出均加载这两项结构。8 卡配置每次更新 262144 tokens，梯度累积为 4，学习率按累计 FLOPs 推进，采用现有生成器取整得到的 `253 / 5045` warmup 比例和 `1514 / 5045` cooldown 比例。新结构从头训练，后续使用同一计划的 checkpoint 续训。

1e18 和 3e18 采用 `scicomp26-small_model_opt_E15_fp32_4gpu_20261106` 的短到长上下文课程与 FP32 主权重策略，并按各自模型、预算和 8 卡配置重新计算计划。前三档课程依次把约 50%、20%、30% 的 FLOPs 预算用于 2048、4096、8192 上下文，每卡 micro batch 依次为 4、2、1，每次更新的总 token 数固定。全部可训练参数保留 FP32 主权重，SOAP 和 AdamW 在主权重上更新，随后回写计算权重。

| 档位 | 2048 阶段步数 | 4096 阶段步数 | 8192 阶段步数 | 梯度累积 | 计划 FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3e17 | 2,887 | 1,101 | 1,514 | 4 | 296,981,168,907,986,688 |
| 1e18 | 4,202 | 1,616 | 2,254 | 3 | 989,896,105,748,962,176 |
| 3e18 | 9,454 | 3,637 | 5,071 | 4 | 2,969,685,208,764,306,048 |

这两档学习率按累计 FLOPs 推进，warmup 和 cooldown 比例沿用各自原计划的 `20 / 原总步数` 与 `原 cooldown 步数 / 原总步数`，峰值为 0.003，最终降至 0。验证始终使用 8192 上下文。阶段结束会保存完整 checkpoint，恢复时核对课程计划、累计 FLOPs、FP32 主权重及其对应的 BF16 权重。新课程从头启动，后续通过 `--resume` 恢复同一课程的 checkpoint。短程检查可使用 `--stop-after`。

课程预算沿用 E15 的解析估计，以原 8192 上下文 ledger 为基准扣除随长度变化的 Attention GEMM 成本。3e17 另按 E16 为双流残差及 Value Residual 加入每 token 75264 FLOPs 的前向与反向保守额度。softmax 和 GDN 边界项沿用基准估计，优化器更新、FP32 权重复制、通信及重计算开销需单独评估。原计划约 1% 的预算余量已保留。3e17 的双流和 Value Residual 使用完整序列前向，训练时采用 eager 执行，推理时设置 `use_cache=False`。

需要在进程意外退出后自动续训时，用 `training/src/auto_resume.sh` 包住原来的启动命令，并保持固定的 `--run-name` 和 `--runs-dir`。例如 1e19 档：

```bash
RUNS_DIR=/path/to/runs
RUN_NAME=wide_compressed_1e19
mkdir -p "$RUNS_DIR"
nohup bash training/src/auto_resume.sh \
  --run-dir "$RUNS_DIR/$RUN_NAME" --max-steps 5346 -- \
  bash training/src/start_train.sh \
    --config training/configs/FLOPs_1e19.yaml --gpus 0,1,2,3,4,5,6,7 \
    --data-dir /path/to/token-cache --data-format packed \
    --run-name "$RUN_NAME" --runs-dir "$RUNS_DIR" \
  > "$RUNS_DIR/$RUN_NAME.watchdog.log" 2>&1 < /dev/null &
```

脚本仅从同一输出目录最近的完整 `step-*.pt` 恢复。最终 checkpoint 存在时停止；没有可恢复 checkpoint、达到重试上限，或连续重试仍停在同一 checkpoint 时停止并写入 `*.auto-resume.tsv`。恢复要求模型配置、数据、GPU 数和训练计划完全一致。持续复现的 NaN 需要先修复原因，自动续训不能消除数值错误。

`training/generate_submission_20261103_configs.py` 可重新生成四个 YAML；`python training/validate_submission_20261103_configs.py` 检查配置与评测默认值。预算由原 Hybrid 逐算子 ledger 与新增压缩分支的保守三倍前向估计相加，计划 FLOPs 留约 1% 余量；该估计不是压缩分支完整逐算子审计。`training/count_flops.py --auto-hybrid` 对此分支会拒绝把缺少压缩分支的 ledger 当成完整核算。

训练数据目录需包含 `metadata.json`、`train.npy`、`validation.npy`，其 tokenizer 词表大小必须为 24576。若使用现有 24K 缓存，先核对目录结构与 `metadata.json`；训练器会检查词表及上下文长度。启动示例：

```bash
bash training/train.sh --image scicomp26:submission-env-hybrid-v3 \
  --config training/configs/FLOPs_3e17.yaml --stage stage1 --gpus 0,1,2,3,4,5,6,7 \
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
