# Full data location

The `scicomp26_data_v0.1` release contains manifests, validation records, reproducibility contracts, and small smoke-test samples. The complete cleaned corpus is intentionally kept outside normal Git because it is approximately 228 GB uncompressed.

## Canonical server location

The current canonical copy is on the controlled Linux training server:

```text
/home/pretrain10/dolma3_official/splits_v2/
├── train/
└── dev/
```

This is release version `splits_v2_cleaned`. It contains 11,224,115 train records and 113,352 dev records (11,337,467 total) across the validated `.jsonl.zst` shards. Use `train/` for model training and `dev/` for evaluation only.

## Access and transfer

This path is server-local and is not created by cloning the repository. Access requires an account and filesystem permission on the controlled server. Do not place credentials or private host details in Git. If the corpus is later moved to an approved object store, DVC, or Git LFS, preserve the `train/` and `dev/` layout and publish an immutable location together with release-level checksums.

## Reproduction and verification

Use `data/splits_v2/manifest.json` as the authoritative split and record-count description and `data/splits_v2/verification.json` for shard validation. The tracked `sample_train_10000.jsonl.zst` and `sample_dev_1000.jsonl.zst` are smoke-test samples only; they are not substitutes for the full corpus. After mounting or copying the server tree, verify the expected shard names, record counts, decompression and JSON parsing, and `bad_json=0` before training.
