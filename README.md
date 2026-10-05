# SCICOMP26

当前基线是约 200M 参数的 dense decoder-only Transformer，配置见
[`configs/200m_dense.yaml`](configs/200m_dense.yaml)。

## Pipeline validation

先生成固定的合成 packed-token 数据：

```bash
python scripts/make_toy_data.py --output-dir data/synthetic-v1
```

运行 10 步训练并保存 checkpoint：

```bash
python scripts/train.py \
  --data-dir data/synthetic-v1 \
  --output-dir runs/synthetic-v1 \
  --max-steps 10
```

需要记录到 Weights & Biases 时，在训练命令末尾加上 `--wandb`；不加该参数则只保存本地日志和 checkpoint。

从 checkpoint 继续训练：

```bash
python scripts/train.py \
  --data-dir data/synthetic-v1 \
  --output-dir runs/synthetic-v1 \
  --resume runs/synthetic-v1/step-10.pt \
  --max-steps 20
```

合成数据只用于验证训练、评估和恢复流程。真实数据目录应遵循
[`docs/data_interface.md`](docs/data_interface.md) 的定义。

## Data release

真实数据的服务器位置、内容范围、记录格式和示例见 [`docs/DATA_LOCATION.md`](docs/DATA_LOCATION.md)。
