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

## Content overview

This is a mixed-domain text pretraining corpus assembled from organizer-provided Dolma3 training data. The collection includes long-form web prose, educational and academic text, technical documentation, source code, question-and-answer material, and structured web-page fragments. A record can contain URLs, HTML-like markup, code, navigation text, or template remnants because these are properties of the source mixture rather than automatic deletion labels.

Representative content patterns include:

- Explanatory prose describing a scientific, educational, or technical concept.
- A programming or configuration fragment, such as a short C/C++/C# or scripting example.
- A question followed by a multi-sentence answer or worked explanation.
- A web-page fragment containing a title, body text, links, or navigation text.

These examples are paraphrased format examples, not verbatim excerpts from the corpus. The source mixture may contain mature, offensive, or otherwise sensitive language; the repository does not reproduce such text in this description.

## Record format

Each validated shard is a Zstandard-compressed JSONL file. Each line is a JSON object with the following fields:

```json
{
  "id": "stable record identifier",
  "text": "normalized document text"
}
```

The `id` is used for deterministic train/dev assignment. The `text` field is normalized and participates in global exact deduplication. The current release contains 16 validated shards: 8 in `train/` and 8 in `dev/`.

## Access and transfer

This path is server-local and is not created by cloning the repository. Access requires an account and filesystem permission on the controlled server. Do not place credentials or private host details in Git. If the corpus is later moved to an approved object store, DVC, or Git LFS, preserve the `train/` and `dev/` layout and publish an immutable location together with release-level checksums.

## Reproduction and verification

Use `data/splits_v2/manifest.json` as the authoritative split and record-count description and `data/splits_v2/verification.json` for shard validation. The tracked `sample_train_10000.jsonl.zst` and `sample_dev_1000.jsonl.zst` are smoke-test samples only; they are not substitutes for the full corpus. After mounting or copying the server tree, verify the expected shard names, record counts, decompression and JSON parsing, and `bad_json=0` before training.
