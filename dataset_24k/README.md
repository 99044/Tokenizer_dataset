# 24K tokenizer、训练和评测缓存

`tok_bpe24k_v1` 的词表有 24,576 个 token。训练和评测缓存的上下文长度为 8,192，元数据中的 tokenizer SHA256 一致。

- Tokenizer 在 `tok_bpe24k_v1/tokenizer/`。
- 测试集为官方 Dolma3 Mix 100M A 的预分词 NumPy 缓存，压缩包分成 5 个 Git 文件。解压后包含 `validation.npy`、`validation_is_text.npy` 和 `metadata.json`。
- 完整训练压缩包为 31,657,767,241 字节，使用项目的现有私有 OSS 对象。包内有训练、验证、官方 A 数据及对应的 `is_text` 掩码和元数据。

## 下载和解包

在已安装 zstd 和 ossutil 且拥有对象读取权限的环境中运行。

```bash
git clone --single-branch --branch Tokenizer_dataset https://github.com/99044/Tokenizer_dataset.git
cd Tokenizer_dataset
cd dataset_24k

sha256sum -c tok_bpe24k_v1_dolma3_mix_100M_a_npy.tar.zst.parts.sha256
cat tok_bpe24k_v1_dolma3_mix_100M_a_npy.tar.zst.part-* > tok_bpe24k_v1_dolma3_mix_100M_a_npy.tar.zst
sha256sum -c tok_bpe24k_v1_dolma3_mix_100M_a_npy.tar.zst.sha256
tar --zstd -xf tok_bpe24k_v1_dolma3_mix_100M_a_npy.tar.zst

ossutil cp \
  oss://scicomp26-handoff/scicomp26/data/token_cache_24k_v1_full_20261101.tar.zst \
  ./token_cache_24k_v1_full_20261101.tar.zst \
  --region cn-beijing
sha256sum -c SHA256SUMS
mkdir -p token_cache_24k_v1_full_20261101
tar --zstd -xf token_cache_24k_v1_full_20261101.tar.zst --strip-components=1 -C token_cache_24k_v1_full_20261101
```

训练数据目录是 `token_cache_24k_v1_full_20261101/`，评测数据目录是 `tok_bpe24k_v1_dolma3_mix_100M_a_npy/`。训练压缩包解开约需 257 GiB，下载并保留两个压缩包时建议预留约 300 GiB。
