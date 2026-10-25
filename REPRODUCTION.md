# 环境与运行说明

本文对应 2026-10-25 提交分支 `submission_20261025`，工作目录为仓库根目录。
四档采用同一 330M OLMo Hybrid 架构。

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

- `training/configs/FLOPs_3e17.yaml`：330M
- `training/configs/FLOPs_1e18.yaml`：330M
- `training/configs/FLOPs_3e18.yaml`：330M
- `training/configs/FLOPs_1e19.yaml`：330M

四档均使用 `[gdn, gdn, gdn, attn]` OLMo Hybrid、32768 词表和 8192 上下文。
训练固定使用四张 GPU，每卡 `micro_batch_size=1`、梯度累积 16 次：

```text
4 × 1 × 16 × 8192 = 524,288 tokens/update
```

前三档使用 SOAP：peak LR `0.003`、betas `(0.95, 0.95)`、eps `1e-8`、
weight decay `0.1`、precondition frequency `10`、max precondition dimension `1024`。
1e19 档使用 AdamW：peak LR `0.0003`、betas `(0.9, 0.95)`、eps `1e-8`、weight decay `0.1`；
模型前向/反向计算保持 BF16，更新 FP32 主权重，再复制回 BF16 模型参数。AdamW moments 保持 FP32。
训练 warmup 20 步，最后约 16.8% 的更新线性 cooldown 到 0。

GDN 初始化增加数值保护：OLMo-core 的 BF16 `uniform(0, 16)` 可能采到精确的 0，
随后 `log(0)` 使 `A_log=-inf`，SOAP 的 weight decay 会将其变为 NaN。
`training/src/model_initialization.py` 仅将这些零衰减率替换为参数 dtype 的最小正规正数
所对应的有限 log 值；不重新采样其他权重，不改变随机数状态、架构或训练超参数。
DDP 在初始化后、创建优化器前执行保护，并记录 `initialization-repairs.json`。
2026-10-25 的旧 500M 运行在第 3 步因此失败；修复后的运行单独保存，原失败日志保留。
本版 1e19 档改用与前三档相同的 330M 架构，须从随机初始化重新训练，不能恢复旧 500M checkpoint。
1e19 档 checkpoint 额外保存 `fp32_master`，恢复时必须同时加载模型、AdamW 状态与主权重；
仅从 BF16 模型参数重建主权重会丢失更新中的低位信息。

GDN 保留原始 BF16 计算路径，未增加 FP32 归一化或递推；衰减项及原内核自身的内部精度保持原实现。
SOAP 恢复 checkpoint 时，`training/src/third_party/soap.py` 会直接恢复原文件中的 FP32 moments、
预条件器和特征基，避免 PyTorch 先转换到 BF16 后造成精度丢失及混合 dtype 错误。
该优化器状态修复不改变模型计算精度、学习率或 scheduler。
`training/src/diagnostic_hooks.py` 保留此前 BF16 运行的只读前向/梯度检查，不修改张量或参数更新。
从 checkpoint 恢复时使用其累计 step、optimizer、scheduler 及各 rank RNG，后续数据按 seed/step/rank 重建。

| 配置 | 实际参数量 | 目标 FLOPs | 计划 FLOPs | max_steps | 计划 tokens | cooldown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOPs_3e17 | 327,544,256 | 300,000,000,000,000,000 | 299,315,365,239,848,960 | 256 | 134,217,728 | 43 |
| FLOPs_1e18 | 327,544,256 | 1,000,000,000,000,000,000 | 999,666,551,875,276,800 | 855 | 448,266,240 | 144 |
| FLOPs_3e18 | 327,544,256 | 3,000,000,000,000,000,000 | 2,998,999,655,625,830,400 | 2,565 | 1,344,798,720 | 431 |
| FLOPs_1e19 | 327,544,256 | 10,000,000,000,000,000,000 | 9,999,003,920,043,704,320 | 8,552 | 4,483,710,976 | 1,438 |

每档均在 10%–90% 的九个进度节点和 100% 最终节点保存 checkpoint，共 10 个；
相同节点执行验证，最终验证使用 64 batches。恢复运行可以额外传 `--save-every 100`，
增加每 100 步的保存，不改变训练计划、学习率和原有验证节点。最终 checkpoint 与 BPB 结果确认可用后，
可删除该 run 的 10%–90% checkpoint 释放空间。

使用完整 Hybrid 逐算子 FLOPs 口径检查架构、参数量、四卡 batch、对应优化器、预算和
checkpoint 节点：

```bash
python training/validate_submission_20261025_configs.py
```

使用已有 packed-token 数据运行单档：

```bash
BUDGET=1e19
bash training/train.sh \
  --image scicomp26:submission-env-hybrid-v3 \
  --config "training/configs/FLOPs_${BUDGET}.yaml" \
  --stage stage1 --gpus 0,1,2,3 \
  --data-dir /path/to/packed-token-data --data-format packed \
  --run-name "submission_20261025_${BUDGET}" \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

不要覆盖 YAML 中的 GPU batch、步数、学习率、warmup 或保存间隔，否则预算与 10% 节点
将不再匹配。未传 `--resume` 时从随机初始化开始；中断恢复必须使用同一配置、四张 GPU、
相同数据 metadata，并传 `--resume <checkpoint>`。

从原始 JSONL 启动时增加 tokenizer 路径，启动器会生成或复用 token cache：

```bash
bash training/train.sh \
  --image scicomp26:submission-env-hybrid-v3 \
  --config training/configs/FLOPs_1e19.yaml \
  --stage stage1 --gpus 0,1,2,3 \
  --data-dir /home/pretrain10/dolma3_official/extracted \
  --data-format jsonl \
  --tokenizer-dir /home/pretrain10/dolma3_official/tokenizer_v1_bpe32k \
  --run-name submission_20261025_1e19 \
  --wandb --wandb-entity echo-lab --wandb-project scicomp26
```

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
