# 训练与导出

四档均使用 24K 词表、8192 最大上下文及 `[gdn, gdn, gdn, attn]` 层模式。前三档为 12 层，最后一档为 16 层；压缩 K/V 分支旁接在各模型最后一层 Attention 上。前三档压缩块为 128 token，最后一档为 388 token。3e17 档分支输出系数为 `0.0001`，其余三档为 `0.001`；原 Attention 输出保留。

| 配置 | 总参数 | hidden | FFN | heads | 优化器 | 步数 | tokens/update | 计划训练 tokens |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `FLOPs_3e17.yaml` | 47,756,790 | 256 | 1984 | 2 | SOAP | 5,502 | 262,144 | 1,442,316,288 |
| `FLOPs_1e18.yaml` | 200,197,036 | 640 | 2048 | 5 | SOAP | 6,052 | 262,144 | 1,586,495,488 |
| `FLOPs_3e18.yaml` | 200,197,036 | 640 | 2048 | 5 | SOAP | 9,078 | 524,288 | 4,759,486,464 |
| `FLOPs_1e19.yaml` | 1,163,259,328 | 1024 | 4352 | 8 | Muon | 8,324 | 524,288 | 4,364,173,312 |

另有 `FLOPs_1e19_1.yaml` 对照配置：保留 1e19 档的主干、压缩分支和 Muon/AdamW 参数组，增加 BF16 前向配 FP32 主权重、最后四层双流残差和 Attention 值残差。它按累计 FLOPs 依次使用 2048/4096/8192 上下文，分别训练 4,459/1,742/2,498 步，共 8,699 步、4,560,781,312 tokens；每次更新仍为 524,288 tokens，最终学习率为 3e-5。训练和续训使用独立的输出目录 `runs/submission_20261103/1e19_1`。

在已安装训练依赖的 GPU 环境中，可用 `run_1e19_1.sh` 直接启动这份配置，不经过 Docker。它接受 `--gpus`、`--data-dir`、`--eval-cache-dir`、`--tokenizer-dir`、`--output-dir`，先核对数据与分词器，再训练、导出 HF，并用预分词评测缓存执行单卡 batch 32、FFN＋RMSNorm 融合评测。输出包含 `train.log`、`export.log`、`eval.log`、`summary.json` 和最多两个滚动训练 checkpoint。训练意外退出后，若已有完整 checkpoint，入口会从最近一步续训；重复无进展会停止并记录到 `auto-resume.tsv`。重新运行已建立的输出目录须加 `--resume`，只做导出与评测可加 `--evaluate-only`，只检查环境可加 `--preflight-only`。

一次运行四档时，使用 `run_submission_all.sh`。它依次运行 `3e17 → 1e18 → 3e18 → 1e19`，每档复用同一组 8 张 GPU，完成训练、HF 导出和评测后进入下一档：

```bash
bash run_submission_all.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir /path/to/24k-tokenized-data \
  --eval-cache-dir /path/to/packed-eval-cache \
  --tokenizer-dir /path/to/24k-tokenizer \
  --output-dir /mnt/experiments/submission_20261107 \
  --resume
```

每档输出保存在 `--output-dir/<档位>/`，其中 `hf-model/` 是导出的 HF 模型，`result.json` 是评测结果。`--resume` 对已有档位目录启用续跑，对未建立的目录正常开始；已完成档位会核对并复用结果。任一档位失败时停止队列，修复后重新执行同一命令即可。追加 `--dry-run` 可预览四档配置及命令。只运行 3e17 档时可用 `run_submission_20261107.sh`，其 `--output-dir` 直接指定该档目录。

实际步数和 token 以 YAML 的 `training` 与 `budget` 字段为准。四档按 8 卡 DDP 配置，梯度累积分别为 4、4、8、8。8192 上下文时每卡 micro batch 为 1。模型使用 BF16 计算，前三档使用 FP32 主权重、主干 SOAP、压缩分支 AdamW；1e19 档主干 Muon、压缩分支 AdamW。峰值学习率均为 0.003。前三档按累计 FLOPs 约 5% warmup、约 70% 后线性 cooldown 至 3e-5；1e19 档按步数 4% warmup、最后 30% 线性 cooldown 至 3e-5。初始化时主干指定投影和 embedding 权重乘以 0.8，压缩分支的输出另乘对应 YAML 中的系数。每档训练进度约每 10% 保存一次，自动只保留最近两个完整 checkpoint。中断后用相同 YAML、数据和 GPU 数量传 `--resume` 继续。

前三档的第 9 至 12 层使用持续双流残差，初始交叉混合比例为 0.25，每行总增益为 2，末层初始合并权重各为 0.5。第 8、12 层 Attention 将当前 V 与第 4 层 V 按各 0.5 混合，梯度回传至第 4 层。每档新增 18 个 FP32 混合参数，纳入 FP32 主权重和优化器。训练、原生评测及 HF 导出均加载这两项结构。更改模型结构或更新计划后需从头训练，后续仅使用同一配置的 checkpoint 续训。

前三档使用短到长上下文课程，依次把约 50%、20%、30% 的 FLOPs 预算用于 2048、4096、8192 上下文，每卡 micro batch 依次为 4、2、1，每次更新的总 token 数在各自档位内固定。全部可训练参数保留 FP32 主权重，SOAP 和 AdamW 在主权重上更新，随后回写计算权重。

| 档位 | 2048 阶段步数 | 4096 阶段步数 | 8192 阶段步数 | 梯度累积 | 计划 FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3e17 | 2,887 | 1,101 | 1,514 | 4 | 296,981,168,907,986,688 |
| 1e18 | 3,150 | 1,212 | 1,690 | 4 | 989,871,178,409,137,408 |
| 3e18 | 4,725 | 1,818 | 2,535 | 8 | 2,969,613,535,227,412,224 |

前三档学习率按累计 FLOPs 推进，峰值为 0.003，最终降至 3e-5。验证始终使用 8192 上下文。恢复时核对课程计划、累计 FLOPs、FP32 主权重及其对应的 BF16 权重。新配置从头启动，后续通过 `--resume` 恢复同一课程的 checkpoint。短程检查可使用 `--stop-after`。

课程预算使用解析估计，以原 8192 上下文 ledger 为基准扣除随长度变化的 Attention GEMM 成本。前三档均为双流残差及 Value Residual 加入每 token 的前向与反向保守额度：3e17 为 75,264 FLOPs，1e18 和 3e18 为 188,160 FLOPs。softmax 和 GDN 边界项沿用基准估计，优化器更新、FP32 权重复制、通信及重计算开销需单独评估。原计划约 1% 的预算余量已保留。双流和 Value Residual 使用完整序列前向，训练时采用 eager 执行，推理时设置 `use_cache=False`。

需要在进程意外退出后自动续训时，用 `training/src/auto_resume.sh` 包住原来的启动命令，并保持固定的 `--run-name` 和 `--runs-dir`。例如 1e19 档：

```bash
RUNS_DIR=/path/to/runs
RUN_NAME=wide_compressed_1e19
mkdir -p "$RUNS_DIR"
nohup bash training/src/auto_resume.sh \
  --run-dir "$RUNS_DIR/$RUN_NAME" --max-steps 8324 -- \
  bash training/src/start_train.sh \
    --config training/configs/FLOPs_1e19.yaml --gpus 0,1,2,3,4,5,6,7 \
    --data-dir /path/to/token-cache --data-format packed \
    --run-name "$RUN_NAME" --runs-dir "$RUNS_DIR" \
  > "$RUNS_DIR/$RUN_NAME.watchdog.log" 2>&1 < /dev/null &
```

脚本仅从同一输出目录最近的完整 `step-*.pt` 恢复。最终 checkpoint 存在时停止；没有可恢复 checkpoint、达到重试上限，或连续重试仍停在同一 checkpoint 时停止并写入 `*.auto-resume.tsv`。恢复要求模型配置、数据、GPU 数和训练计划完全一致。持续复现的 NaN 需要先修复原因，自动续训不能消除数值错误。

`training/generate_submission_20261103_configs.py` 可重新生成四档 YAML 和 1e19 对照配置；`python training/validate_submission_20261103_configs.py` 检查配置与评测默认值。预算由原 Hybrid 逐算子 ledger 与新增压缩分支的保守三倍前向估计相加，计划 FLOPs 留约 1% 余量；该估计不是压缩分支完整逐算子审计。`training/count_flops.py --auto-hybrid` 对此分支会拒绝把缺少压缩分支的 ledger 当成完整核算。

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
