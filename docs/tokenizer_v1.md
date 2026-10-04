# Tokenizer v1：训练方法与服务器位置

当前 tokenizer 候选版本为 `tok_bpe32k_v1`，使用主办方最终训练集的 `train` 子集训练，没有使用 dev 数据或外部语料。

## 训练方法

- 类型：Byte-level BPE。
- 词表大小：32,768，包含完整 256-byte alphabet 和 4 个特殊 token。
- 抽样：从 `splits_v2/train` 按记录做确定性 5% 抽样，不取文件前 5%。
- 抽样规则：有 `id` 时使用 `id:` + `str(id)`，否则使用 `text:` + `str(text)`；计算 SHA-256，取前 8 字节按大端整数解释，保留 `% 100 < 5` 的记录。
- tokenizer 不做二次 NFC/NFKC、大小写转换、strip 或换行修改；训练输入已经按 `normalized_v1` 处理。
- ByteLevel 参数为 `add_prefix_space=False`、`trim_offsets=False`。

特殊 token 固定为：

| 名称 | 字面量 | ID |
|---|---|---:|
| UNK | `<|scicomp26_unk_v1|>` | 0 |
| BOS | `<|scicomp26_bos_v1|>` | 1 |
| EOS | `<|scicomp26_eos_v1|>` | 2 |
| PAD | `<|scicomp26_pad_v1|>` | 3 |

Train 和 Infer 调用 tokenizer 时显式使用 `add_special_tokens=False`。tokenizer 本身不自动加入 BOS/EOS；packing 代码在每篇非空文档前放置一个 EOS，PAD 只用于批处理补齐并屏蔽 loss。

## 服务器位置

训练输入：

```text
/home/pretrain10/dolma3_official/splits_v2/train/
```

已训练产物：

```text
/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k/
```

该目录包含：

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
training_metadata.json
```

训练脚本：

```text
/home/pretrain10/tokenizer_workflow/train_byte_bpe.py
```

Infra/Infer 契约：

```text
/home/pretrain10/dolma3_official/logs/tokenizer_v1_bpe32k/tokenizer_infer_contract_v2.json
```

特殊 token 字面量碰撞扫描结果：

```text
/home/pretrain10/dolma3_official/logs/tokenizer_v1_bpe32k/collision_scan_train_dev.json
```

扫描覆盖 train+dev 共 11,337,467 条记录、16 个文件和 89,013,225,290 个 UTF-8 字节；四个项目特殊 token 的正文碰撞均为 0。

## 离线加载

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k",
    local_files_only=True,
)
ids = tokenizer(
    "Hello, 世界",
    add_special_tokens=False,
    truncation=False,
    padding=False,
)["input_ids"]
```

当前版本是候选 tokenizer。16K/32K 词表在相同 FLOPs 下的模型对照和官方 bpb 结果尚未完成，不能把本版本描述为最终最优方案。
