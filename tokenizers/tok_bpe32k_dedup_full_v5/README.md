# tok_bpe32k_dedup_full_v5

这是一个 32,768 词表项（包含 4 个特殊 token）的 byte-level BPE tokenizer 候选版本。目录内包含离线加载所需的 tokenizer 文件、训练元数据、验证摘要和 SHA-256 校验清单；完整训练语料不随 GitHub 仓库发布。验证环境使用 `transformers==5.18.0` 和 `tokenizers==0.23.1`；其他版本请先确认兼容性。

## 使用

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "tokenizers/tok_bpe32k_dedup_full_v5",
    local_files_only=True,
    use_fast=True,
)
encoded = tokenizer(
    "Hello, 世界",
    add_special_tokens=False,
    truncation=False,
    padding=False,
)["input_ids"]
```

Tokenizer 不自动添加 BOS/EOS。特殊 token ID 为 UNK/BOS/EOS/PAD = `0/1/2/3`。下游 packing 按训练元数据中的契约在每篇文档前插入一个 EOS，并屏蔽插入的 EOS 预测目标。不同版本间普通 token ID 不兼容；切换 tokenizer 时，必须重新 tokenize/pack，并让模型词表与所用版本一致。

## 训练与验证范围

训练使用冻结的内部去重训练集中的全部非空唯一文档（`sample_percent=100`）；按原始文本 SHA-256 全局去重，保留首次出现，不使用 ID 作为去重键，也不执行文本归一化。训练元数据记录了来源分片哈希、特殊 token 碰撞扫描和运行库版本。

已记录的验证覆盖 1,031 条内部验证文档与合成测试向量：2,043,444 个原文 token，ID 不匹配、往返失败和未知/保留 ID 均为 0。该范围不包含官方评测集或完整 dev 集。此版本尚无同算力模型 BPB 对照，因此不能据此断定它是最终最优 tokenizer。

校验文件完整性：

```bash
sha256sum -c tokenizers/tok_bpe32k_dedup_full_v5/SHA256SUMS
```

服务器上的共享副本位于：

```text
/home/pretrain10/dolma3_official/tokenizer_retrain_20261010_dedup_full32k_v5/tokenizer/
```

`training_metadata.json` 中的绝对路径保留了原训练环境中的记录；该共享副本路径是当前服务器位置。
