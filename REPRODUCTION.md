# 环境与运行说明

本文对应 `olmo_hybrid_1b`，工作目录为仓库根目录。目录与 `infra` 对齐，
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
Hybrid 所需算子及其固定版本尚须另行配置验证，本次不替换 attention 或 GDN 实现。
已有 `docker_image.tar` 保留，不因目录整理自动重建或纳入 Git。

## 2. 数据与 tokenizer

原始官方 JSONL 为 `/home/pretrain10/dolma3_official/extracted/*.jsonl`，
现有 tokenizer 为 `/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k/`。
复用 `tok_bpe32k_v1` 的 32768 词表，不重新训练 tokenizer。
四档数据路径可用命令行覆盖；完整数据处理来源及文件 SHA-256 保存在缓存 `metadata.json`。

`training/src/prepare_token_cache.py` 按分片名字典序、文件内原行序读取，不清洗或截断正文；
按 `SHA-256(str(id))` 前 64 位大端整数 `%10000<100` 划分 1% 内部验证集，缺少 id 时使用 text。
每篇文档前置 EOS，正文完整分词后拼接为连续 uint32 数组，无 padding。
插入 EOS 的 `is_text` 为 0，正文 token 为 1；同时记录原 text 的 UTF-8 字节数。
缓存指纹包含来源文件属性、tokenizer 哈希、脚本哈希、序列长度和划分比例，
首次完整生成并验证后发布，后续启动复用；默认根目录 `../data/extracted-token-cache/`。

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
因此目录与数据入口对齐不代表 Hybrid 全流程训练已经验证。

## 3. 四档配置与训练

四档配置分别为 `training/configs/FLOPs_3e17.yaml`、`FLOPs_1e18.yaml`、
`FLOPs_3e18.yaml`、`FLOPs_1e19.yaml`。模型结构、信号配置及预算值保留原文件。

| 设置 | 原训练器及下列命令 |
| --- | --- |
| 随机种子 | 42 |
| 优化器 | AdamW，betas=(0.9, 0.95)，eps=1e-8，weight decay=0.1，gradient clip=1.0 |
| 学习率与调度 | CLI 默认 2e-4；warmup 默认 0，后续 cosine；用 --learning-rate / --warmup-steps 覆盖 |
| 序列长度 / 精度 | 8192 / bfloat16 |
| 硬件与并行 | 两张 A100，GPU 1、2，torchrun DDP，一卡一个进程 |
| batch | 每卡 1 条序列，累积 128 次；全局 256 条，即 2,097,152 输入 tokens/update |
| 初始化与恢复 | 新运行从随机初始化；--resume 恢复 model/optimizer/scheduler/每 rank RNG，计划必须一致 |

先设置四档 `STEPS_3E17`、`STEPS_1E18`、`STEPS_3E18`、`STEPS_1E19` 为各自已核定的正整数步数。
GDN FLOPs 公式尚未接入，不能将稠密 OLMo3 的 563/676 步作为 Hybrid 预算依据。
当前启动器只传递手动 `--steps`，不接受 `--target-flops` 自动预算。
W&B 登录入口为 `bash training/src/login_wandb.sh scicomp26:submission-env-v1`。

```bash
bash training/train.sh \
  --image scicomp26:submission-env-v1 \
  --config training/configs/FLOPs_3e17.yaml \
  --stage stage1 --gpus 1,2 \
  --data-dir /home/pretrain10/dolma3_official/extracted \
  --data-format jsonl \
  --tokenizer-dir /home/pretrain10/dolma3_official/tokenizer_v1_bpe32k \
  --micro-batch-size 1 --grad-accum-steps 128 \
  --steps "$STEPS_3E17" --save-every 10 --eval-every 10 \
  --run-name olmo_hybrid_1b-3e17 \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

```bash
bash training/train.sh \
  --image scicomp26:submission-env-v1 \
  --config training/configs/FLOPs_1e18.yaml \
  --stage stage1 --gpus 1,2 \
  --data-dir /home/pretrain10/dolma3_official/extracted \
  --data-format jsonl \
  --tokenizer-dir /home/pretrain10/dolma3_official/tokenizer_v1_bpe32k \
  --micro-batch-size 1 --grad-accum-steps 128 \
  --steps "$STEPS_1E18" --save-every 10 --eval-every 10 \
  --run-name olmo_hybrid_1b-1e18 \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

```bash
bash training/train.sh \
  --image scicomp26:submission-env-v1 \
  --config training/configs/FLOPs_3e18.yaml \
  --stage stage1 --gpus 1,2 \
  --data-dir /home/pretrain10/dolma3_official/extracted \
  --data-format jsonl \
  --tokenizer-dir /home/pretrain10/dolma3_official/tokenizer_v1_bpe32k \
  --micro-batch-size 1 --grad-accum-steps 128 \
  --steps "$STEPS_3E18" --save-every 10 --eval-every 10 \
  --run-name olmo_hybrid_1b-3e18 \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

```bash
bash training/train.sh \
  --image scicomp26:submission-env-v1 \
  --config training/configs/FLOPs_1e19.yaml \
  --stage stage1 --gpus 1,2 \
  --data-dir /home/pretrain10/dolma3_official/extracted \
  --data-format jsonl \
  --tokenizer-dir /home/pretrain10/dolma3_official/tokenizer_v1_bpe32k \
  --micro-batch-size 1 --grad-accum-steps 128 \
  --steps "$STEPS_1E19" --save-every 10 --eval-every 10 \
  --run-name olmo_hybrid_1b-1e19 \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

启动器实际执行以下 Docker 挂载方式；其中 `CACHE_PATH` 为已发布的指纹缓存目录，
`RUN_DIR` 为新建输出目录，`CONFIG_PATH` 为四档配置之一，`STEPS` 为人工核定步数：

```bash
docker run --rm --pull=never --runtime=nvidia --gpus '"device=1,2"' --shm-size 64g \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=1 --workdir /workspace/comp/scicomp26 \
  --mount "type=bind,src=$PWD,dst=/workspace/comp/scicomp26,readonly" \
  --mount "type=bind,src=$CACHE_PATH,dst=/data,readonly" \
  --mount "type=bind,src=$RUN_DIR,dst=/runs" \
  scicomp26:submission-env-v1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  training/src/train_ddp.py --config "$CONFIG_PATH" --data-dir /data --output-dir /runs \
  --max-steps "$STEPS" --micro-batch-size 1 --grad-accum-steps 128 \
  --learning-rate 2e-4 --warmup-steps 0 --eval-every 10 --save-every 10
```

启用 W&B 时，启动器增加 host 网络、代理变量、凭据只读挂载及 `/runs` 日志环境。
恢复使用新的 `--run-name`，并追加 `--resume <原运行目录>/step-<n>.pt`，其余训练计划保持一致。
`--dry-run` 只打印预处理和训练命令，不生成缓存或启动训练。

## 4. 训练记录与 FLOPs 核算

运行目录为 `../runs/<run-name>/`，保存 `console.log`、`run-plan.json`、`metrics.jsonl`、
`step-<n>.pt`；启用 W&B 时增加 `wandb-run.json`。
原 DDP 训练器每个 optimizer update 记录 loss、lr、耗时及吞吐，满足不超过 10 步的记录间隔。
最终分别归档到 `training/logs/FLOPs_3e17/`、`FLOPs_1e18/`、`FLOPs_3e18/`、`FLOPs_1e19/`。
断点恢复须按累计 step 衔接日志，去除重复步；以选定 checkpoint 的实际 step 统计 tokens。

`training/count_flops.py` 保留 infra 的标准稠密公式及适用范围检查，明确拒绝 Hybrid 类型。
审计 GDN的 forward/backward FLOPs/token 后，可作如下累计核算：

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
