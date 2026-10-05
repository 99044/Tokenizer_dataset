# Tokenizer 候选版本：训练方法、服务器位置与 24K 对照

当前维护两个 tokenizer 候选版本：

| 版本 | revision | 词表大小 |
|---|---|---:|
| 32K 基线 | `tok_bpe32k_v1` | 32,768 |
| 24K 对照 | `tok_bpe24k_v1` | 24,576 |

两个版本使用完全相同的训练数据、抽样规则、特殊 Token 和 Byte-level BPE 配置，唯一的主要变量是词表大小。它们都是候选版本；相同 FLOPs 下的模型 dev bpb 对照尚未完成。

## 共同训练方法

- 训练来源：主办方训练子集经过团队处理后的 `splits_v2/train`，没有使用 dev 数据、外部语料或生成数据。
- 类型：Byte-level BPE，完整 256-byte alphabet。
- 抽样：按记录做确定性 SHA-256 5% 抽样，不取文件前 5%。有 `id` 时使用 `id:` + `str(id)`，否则使用 `text:` + `str(text)`；取 digest 前 8 字节按大端整数解释，保留 `% 100 < 5` 的记录。
- `min_frequency=2`。
- 输入已经按 `normalized_v1` 处理；tokenizer 不再执行 NFC/NFKC、大小写转换、strip 或换行修改。
- ByteLevel 参数：`add_prefix_space=False`、`trim_offsets=False`、`use_regex=True`；normalizer 为 `None`。
- 后处理器不自动注入 BOS/EOS/PAD。

特殊 Token 在两个版本中保持一致：

| 名称 | 字面量 | ID |
|---|---|---:|
| UNK | `<|scicomp26_unk_v1|>` | 0 |
| BOS | `<|scicomp26_bos_v1|>` | 1 |
| EOS | `<|scicomp26_eos_v1|>` | 2 |
| PAD | `<|scicomp26_pad_v1|>` | 3 |

Train 和 Infer 编码原文时显式使用 `add_special_tokens=False`。Tokenizer 本身不自动加入 BOS/EOS；packing 契约规定在每篇非空源文档前放置恰好一个 EOS，切块边界不重复插入 EOS。PAD 只用于批处理补齐并屏蔽 loss；24K 训练元数据也已按该前置 EOS 契约更新。

## 服务器位置

训练输入：

```text
/home/pretrain10/dolma3_official/splits_v2/train/
```

32K 产物：

```text
/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k/
```

24K 产物：

```text
/home/pretrain10/dolma3_official/tokenizer_v1_bpe24k/
```

24K 目录包含：

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
training_metadata.json
SHA256SUMS
README.md
```

训练脚本：

```text
/home/pretrain10/tokenizer_workflow/train_byte_bpe.py
```

24K 训练命令：

```bash
/home/pretrain10/cleaning_shard_20261004/venv/bin/python \
  /home/pretrain10/tokenizer_workflow/train_byte_bpe.py \
  --input /home/pretrain10/dolma3_official/splits_v2/train \
  --output /home/pretrain10/dolma3_official/tokenizer_v1_bpe24k \
  --vocab-size 24576 \
  --sample-percent 5 \
  --revision tok_bpe24k_v1
```

## 验证结果

24K 实际词表大小为 24,576，特殊 Token ID 为 `0/1/2/3`。服务器环境的离线加载、示例文本 byte-level round-trip 和 `<unk>` 检查通过；`tokenizer.json` SHA-256 为：

```text
6f1812b3fbf8758220be646974072ef08c66df2d2d5ef9d398f0e158b9ee947c
```

完整 train+dev 特殊 Token 字面量扫描结果：

```text
/home/pretrain10/dolma3_official/logs/tokenizer_v1_bpe24k/collision_scan_train_dev.json
```

扫描覆盖 16 个文件、11,337,467 条记录和 89,013,225,290 个 UTF-8 字节；四个项目特殊 Token 的字面量碰撞均为 0。

## 24K/32K smoke 对照

报告位置：

```text
/home/pretrain10/dolma3_official/logs/tokenizer_v1_bpe24k/tokenizer_compare_24k_vs_32k_smoke20k.json
```

该报告处理的是 dev 文件顺序中的前 20,000 条非空记录，`is_prefix_smoke_test=true`；它不是随机抽样，也不是完整 dev 结论。

| 指标 | 32K | 24K |
|---|---:|---:|
| UTF-8 字节 | 119,985,925 | 119,985,925 |
| Token 数 | 27,936,558 | 28,644,451 |
| Token/byte | 0.232832 | 0.238732 |
| byte/token | 4.294943 | 4.188802 |
| Token 长度 p50 | 786 | 806 |
| Token 长度 p90 | 2,801 | 2,868 |
| Token 长度 p99 | 10,628 | 10,908 |
| round-trip 失败 | 0 | 0 |

24K 在该 smoke 样本上比 32K 多 2.53393% Token。这个结果只说明 tokenizer 压缩率差异，不能替代相同模型、相同 FLOPs 下的 dev bpb 对照。

## 使用限制

- 24K 与 32K 的普通 Token ID 映射不能混用；切换版本必须重新 tokenize/pack，并将模型配置中的 `vocab_size` 改为对应值。
- `model_max_length=8192` 只是 tokenizer 配置提示，不代表模型一定支持 8192 上下文；训练 context length、packing 长度和官方评测长度必须分别核对。
- 当前 metadata 记录了规则和路径，但没有记录完整输入 shard 哈希、实际抽样记录/字节数及运行包版本，因此只能保证规则级复现，不能宣称逐比特重训复现。
- 24K/32K 模型的正式选择仍需按赛事 FLOPs 公式训练独立模型，并比较官方 bpb。

离线加载示例：

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "/home/pretrain10/dolma3_official/tokenizer_v1_bpe24k",
    local_files_only=True,
)
ids = tokenizer(
    "Hello, 世界",
    add_special_tokens=False,
    truncation=False,
    padding=False,
)["input_ids"]
```

