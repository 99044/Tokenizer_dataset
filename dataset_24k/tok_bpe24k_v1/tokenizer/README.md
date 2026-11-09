# tok_bpe24k_v1 对照版本

## 训练配置

- 训练输入：`/home/pretrain10/dolma3_official/splits_v2/train`
- 输出目录：`/home/pretrain10/dolma3_official/tokenizer_v1_bpe24k`
- 版本：`tok_bpe24k_v1`
- 词表：24,576（包含 256-byte alphabet 和 4 个特殊 Token）
- 抽样：确定性 SHA-256 5%，规则与 `tok_bpe32k_v1` 完全相同
- 算法：Byte-level BPE，`min_frequency=2`
- 特殊 Token ID：UNK=0、BOS=1、EOS=2、PAD=3
- 未使用 dev 或外部语料

训练命令：

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

- 实际词表大小：24,576
- 离线加载：通过
- 示例文本 round-trip：通过
- `<unk>`：0
- Train+Dev 特殊 Token 正文碰撞：0
- 扫描范围：16 个文件、11,337,467 条记录、89,013,225,290 个 UTF-8 字节

## 20,000 条 dev smoke 对照

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

24K 在该 smoke 样本上比 32K 多约 2.53% Token。该结果仅用于 tokenizer 压缩率和接口诊断；正式选择仍需在相同模型和相同 FLOPs 下比较 dev bpb。

详细报告：`logs/tokenizer_compare_24k_vs_32k_smoke20k.json`
碰撞报告：`logs/collision_scan_train_dev.json`
