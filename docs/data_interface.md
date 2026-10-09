# Training Data Interface

数据接口：

```text
data/<data_version>/
├── tokenizer/
├── train.npy
├── validation.npy
└── metadata.json
```

## 文件规定

- `train.npy` 和 `validation.npy` 是一维、连续的 token ID 数组；
- dtype 使用 `uint32` 或训练框架要求的整数类型；
- token ID 必须小于 tokenizer 的 `vocab_size`；
- 文件中不存 padding；
- 每个文档之间是否插入 EOS 由 `metadata.json` 记录，训练版本内保持一致；
- DataLoader 读取 `context_length + 1` 个 token，前 `context_length` 个作为输入，后移一位作为标签；
- tokenizer 目录包含训练和评测所需的完整文件。

## metadata.json

每个数据版本提供一个简单的元数据文件，例如：

```json
{
  "data_version": "dolma3-mix-150b-1025-v1",
  "source": "AllenAI Dolma3 mix-150B-1025",
  "tokenizer": "<tokenizer-name-or-revision>",
  "vocab_size": 50304,
  "eos_token_id": 0,
  "train_tokens": 0,
  "validation_tokens": 0,
  "context_length": 8192,
  "created_at": "2026-10-03",
  "notes": "<preprocessing notes>"
}
```

数据预处理规则或数据来源发生变化时，递增 `data_version`，并在 `notes` 中记录变更说明。
训练配置至少记录 `data_version`、tokenizer 路径和两个 `.npy` 文件路径。
