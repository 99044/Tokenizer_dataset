# 环境与运行说明

本文对应 `olmo_hybrid_250m`，工作目录为仓库根目录。目录与 `infra` 对齐，
模型实现、训练循环和四档模型配置保留本分支版本；不复用 OLMo3 的历史训练记录。

## 1. 环境与镜像

固定依赖为根目录 `requirements.txt`，以及配套 `pyproject.toml` / `uv.lock`。
运行环境使用 Docker，uv 文件用于依赖锁定；镜像构建文件为 `training/src/Dockerfile`。

| 项目 | 对齐 infra 的基础环境 |
| --- | --- |
| 系统 | Ubuntu 22.04，Linux x86_64 |
| Python | 3.11 |
| 框架 | PyTorch 2.6.0+cu124，ai2-olmo-core 2.6.0 |
| GPU 运行库 | CUDA 12.4，cuDNN 9.1，NCCL 2.21.5 |
| 分词与模型工具 | Transformers 5.18.0，tokenizers 0.23.1，safetensors 0.8.0 |
| 其他 | NumPy 1.26.4，PyYAML 6.0.3，W&B 0.30.0 |

若归档文件为空占位，先用真实环境镜像执行
`docker save -o docker_image.tar scicomp26:submission-env-v1`；已有真实归档直接加载：

```bash
docker load -i docker_image.tar
docker image inspect scicomp26:submission-env-v1
```

需要重建基础镜像时，在仓库根目录执行：

```bash
docker build -f training/src/Dockerfile -t scicomp26:submission-env-v1 .
```

镜像只包含运行环境及 OLMo-core 兼容补丁；源码、权重、数据和日志通过挂载提供。
现有基础镜像没有安装 `flash_attn` / `fla`；原模型仍指定 `flash_2` 并使用 GDN。
当前训练使用已安装 FlashAttention/FLA 和编译工具的 `scicomp26:submission-env-hybrid-v3`。
从现有 Hybrid v2 环境生成 v3（保留依赖版本，修复 Triton 3.2 首次反向编译错误）：

```bash
docker build --target hybrid_kernel_compat -f training/src/Dockerfile \
  -t scicomp26:submission-env-hybrid-v3 .
```

该构建目标需要已有 `scicomp26:submission-env-hybrid-v2`；可用
`--build-arg HYBRID_KERNEL_BASE_IMAGE=<兼容的 Hybrid 镜像>` 指定基础环境。
补丁只调整乘法与转置顺序，保持模型和算术结果。
已有 `docker_image.tar` 保留，不因目录整理自动重建或纳入 Git。

## 2. 数据与 tokenizer

原始官方 JSONL 为 `/home/pretrain10/dolma3_official/extracted/*.jsonl`，
现有 tokenizer 为 `/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k/`。
复用 `tok_bpe32k_v1` 的 32768 词表，不重新训练 tokenizer。
四档数据路径可用命令行覆盖；数据处理来源和文件属性保存在缓存 `metadata.json`；tokenizer 小文件保留 SHA-256。

`training/src/prepare_token_cache.py` 按分片名字典序、文件内原行序读取，不清洗或截断正文；
按 `SHA-256(str(id))` 前 64 位大端整数 `%10000<100` 划分 1% 内部验证集，缺少 id 时使用 text。
每篇文档前置 EOS，正文完整分词后拼接为连续 uint32 数组，无 padding。
插入 EOS 的 `is_text` 为 0，正文 token 为 1；同时记录原 text 的 UTF-8 字节数。
缓存指纹包含来源文件属性、tokenizer 哈希、固定的分词配方标识、序列长度和划分比例，
首次生成时完成 token 范围、数组类型/长度和掩码值/计数的全量校验，写入 `.validation/`。
后续启动核对元数据、数组头以及 size/mtime/ctime；匹配则复用，变化才重新扫描。
数据只读时，启动器在 GPU 启动前完成 CPU 校验，将记录持久化到
`../data/token-cache-validation/`；不对大型数据文件计算 SHA-256。
默认缓存根目录 `../data/extracted-token-cache/`。

```bash
mkdir -p /home/zhangrui/comp/data/extracted-token-cache
docker run --rm --pull=never --network none \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /workspace/comp/scicomp26 \
  --mount "type=bind,src=$PWD,dst=/workspace/comp/scicomp26,readonly" \
  --mount type=bind,src=/home/pretrain10/dolma3_official/extracted,dst=/raw,readonly \
  --mount type=bind,src=/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k,dst=/tokenizer,readonly \
  --mount type=bind,src=/home/zhangrui/comp/data/extracted-token-cache,dst=/cache \
  scicomp26:submission-env-v1 python training/src/prepare_token_cache.py \
  --source-root /raw --tokenizer /tokenizer --output /cache \
  --model-config training/configs/FLOPs_1e19.yaml
```

单独预处理时先创建宿主机缓存根目录；`training/train.sh` 会自动创建。
已有 packed 数据可用 `--data-format packed --data-dir <目录>`。
旧训练器需要顶层 `vocab_size`、`context_length`、`train_tokens`、`validation_tokens`；
新缓存同时提供这些兼容字段。两份 `.npy` 必须为一维整数数组，token ID 小于词表大小，
每个 split 至少包含 `context_length+1` 个 token。
采样保持原有随机位置、有放回方式，不保证一个 FLOPs 预算覆盖整份数据一遍。

原训练器未读取 `is_text` 掩码，也未传递 `cu_doc_lens` 文档边界；本次保留原采样和 loss 行为。
GDN 文档边界重置仍需后续数据接口适配。
本分支提供代码和训练计划；GPU 运行验证及完整预算训练由 infra 组执行。

## 3. 四档配置与训练

四档配置文件名保持提交规范不变：

- `training/configs/FLOPs_3e17.yaml`
- `training/configs/FLOPs_1e18.yaml`
- `training/configs/FLOPs_3e18.yaml`
- `training/configs/FLOPs_1e19.yaml`

四档使用相同的 254,467,368 参数 OLMo Hybrid 模型：16 层、hidden size 896、
FFN intermediate size 3584、7 个 attention/KV heads，block pattern 为
`[gdn, gdn, gdn, attn]`。词表为 32768，上下文长度为 8192。

两卡基准配置为每卡 `micro_batch_size=1`、`gradient_accumulation_steps=32`，
即每次 optimizer update 处理 524,288 tokens。优化器为 AdamW，peak LR 为 2.83e-4，
warmup 20 步；随后保持 constant LR，在最后约 16.8% 的更新中线性降至 0。
该比例来自 250M 的 B3 cooldown 实验，四档按总步数等比例取整。

| 配置 | 目标 FLOPs | 计划 FLOPs | max_steps | 计划 tokens | cooldown_steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLOPs_3e17 | 300,000,000,000,000,000 | 299,638,724,787,417,600 | 325 | 170,393,600 | 55 |
| FLOPs_1e18 | 1,000,000,000,000,000,000 | 999,410,392,829,417,472 | 1,084 | 568,328,192 | 182 |
| FLOPs_3e18 | 3,000,000,000,000,000,000 | 2,999,153,143,795,290,624 | 3,253 | 1,705,508,864 | 547 |
| FLOPs_1e19 | 10,000,000,000,000,000,000 | 9,999,635,720,136,403,968 | 10,846 | 5,686,427,648 | 1,823 |

以上计划使用 Hybrid 逐算子口径
`1,800,713,490,309 / 1,024 FLOPs/token`。计划 FLOPs 均不超过对应预算，
且与目标差异小于 1%。可在仓库根目录静态检查四档配置：

```bash
python training/validate_250m_configs.py
```

训练入口会直接读取 YAML 内的模型、步数、batch、学习率、warmup、验证和保存间隔。
使用已生成的 packed-token 数据时，只需替换档位、GPU 和数据目录：

```bash
BUDGET=1e19
bash training/train.sh \
  --image scicomp26:submission-env-hybrid-v3 \
  --config "training/configs/FLOPs_${BUDGET}.yaml" \
  --stage stage1 --gpus 0,1 \
  --data-dir /path/to/packed-token-data \
  --data-format packed \
  --run-name "olmo_hybrid_250m_${BUDGET}" \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

不要在上述命令中覆盖 `--steps`、`--micro-batch-size`、`--grad-accum-steps`、
`--learning-rate` 或 `--warmup-steps`，除非同时重新核算该档预算。四张 GPU 训练若要保持
相同的 524,288 tokens/update，可将梯度累积改为 16；此时 checkpoint 不能与两卡计划直接
按完整 `run-plan.json` 恢复。

从原始 JSONL 启动时增加 tokenizer 路径，启动器会先生成或复用缓存：

```bash
bash training/train.sh \
  --image scicomp26:submission-env-hybrid-v3 \
  --config training/configs/FLOPs_1e19.yaml \
  --stage stage1 --gpus 0,1 \
  --data-dir /home/pretrain10/dolma3_official/extracted \
  --data-format jsonl \
  --tokenizer-dir /home/pretrain10/dolma3_official/tokenizer_v1_bpe32k \
  --run-name olmo_hybrid_250m_1e19 \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

未传 `--resume` 或 `--branch-from` 时从随机初始化开始。中断后使用相同配置、GPU 数、
数据 metadata 和训练参数，并传 `--resume <checkpoint>` 恢复。`--dry-run` 只打印预处理与
训练命令。GPU 运行验证与完整训练由 infra 组执行。

## 4. 训练记录与 FLOPs 核算

运行目录为 `../runs/<run-name>/`，保存 `console.log`、`run-plan.json`、`metrics.jsonl`、
`step-<n>.pt`、各次启动的 `runtime_graph-start-<n>.json`；启用 W&B 时增加 `wandb-run.json`。
原 DDP 训练器每个 optimizer update 记录 loss、lr、耗时及吞吐，满足不超过 10 步的记录间隔。
最终分别归档到 `training/logs/FLOPs_3e17/`、`FLOPs_1e18/`、`FLOPs_3e18/`、`FLOPs_1e19/`。
断点恢复须按累计 step 衔接日志，去除重复步；以选定 checkpoint 的实际 step 统计 tokens。

`training/count_flops.py` 的默认入口保留 infra 的标准稠密公式及适用范围检查。
Hybrid 使用 `--auto-hybrid --scalar-op-cost 1` 或经过审计的 `--operator-ledger`；
核算已完成训练时须提供完整 metrics、run-plan、checkpoint step，以及初始运行和
每次恢复的 `--runtime-metadata runtime_graph-start-<n>.json`，核对实际模块和内核源码。
审计 GDN 的 forward/backward FLOPs/token 后，也可作如下累计核算：

```bash
python training/src/token_flops.py \
  --flops-per-token "$AUDITED_HYBRID_FLOPS_PER_TOKEN" \
  --tokens "$ACTUAL_TRAINING_TOKENS" --budget 1e19 > training/logs/FLOPs_1e19/flops_record.json
```

其他三档用对应预算和日志目录。`ACTUAL_TRAINING_TOKENS` 为实际更新步数乘输入 tokens/update，
包括 EOS 的前向计算；核算值、所选 checkpoint 与完整日志须对应，不把计划数当实际数。

## 5. HF 导出与评测

四档最终目录为 `models/FLOPs_3e17/`、`FLOPs_1e18/`、`FLOPs_3e18/`、`FLOPs_1e19/`。
最终需保存模型配置、safetensors 分片及索引、生成配置、完整 tokenizer；
自定义架构还需 HF configuration/modeling 类及注册信息，且与原 checkpoint logits 一致。
当前 `training/export_hf.py` 的原生 checkpoint 转换仅支持 OLMo3，明确拒绝 Hybrid。
Hybrid/GDN的 HF 适配尚待单独实现验证，空占位文件不能离线加载。
最终每档须记录源运行、源 checkpoint step、导出版本及离线加载检查；此处没有宣称完成导出。

`evaluation/eval.py` 和 `evaluation/eval.sh` 与 infra 原文件一致，序列长度 8192，
每 GPU batch 1，BF16，`attn_implementation=sdpa`，`use_cache=False`，不继承 KV cache。
它们通过参数接收原始 JSONL 和 HF 模型目录；完整分词、按顺序拼接分块，尾块 padding。
仅额外插入 EOS 和补齐 PAD 不作为 loss 目标；原 text 的 UTF-8 字节用于 bpb。
需要能够被该固定评测入口离线加载的 HF 模型；当前尚未支持自定义 Hybrid 的加载注册。

```bash
mkdir -p evaluation_output
docker run --rm --pull=never --network none --runtime=nvidia --gpus all --shm-size 64g \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --workdir /output \
  --mount "type=bind,src=$PWD/evaluation,dst=/evaluation,readonly" \
  --mount "type=bind,src=$PWD/models/FLOPs_1e19,dst=/model,readonly" \
  --mount "type=bind,src=$EVAL_JSONL,dst=/dataset.jsonl,readonly" \
  --mount "type=bind,src=$PWD/evaluation_output,dst=/output" \
  scicomp26:submission-env-v1 bash /evaluation/eval.sh /dataset.jsonl /model
```

评测入口固定启动 8 个 GPU 进程，输出为宿主机 `evaluation_output/result.json`，
包含 `model_path`、`bpb`、`evaltime`。本机四卡无法直接完成该固定八卡入口的评测。
