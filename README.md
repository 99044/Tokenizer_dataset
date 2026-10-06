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

## Docker 多卡训练与 W&B

在有 Docker 权限的宿主机终端运行；镜像需提前构建：

```bash
docker build -t scicomp26:cu121-train-v1 .
```

需要上传到 W&B 时，先在宿主机执行 `wandb login`，或在同一终端设置
`WANDB_API_KEY`。启动脚本会将环境变量或只读的 `~/.netrc` 传入容器，
不会把 API key 写入训练配置或打印到命令中。

也可以直接使用训练镜像中的 W&B CLI 登录，无需在宿主机安装 W&B：

```bash
bash scripts/login_wandb.sh
```

在命令提示中输入 API key。脚本会验证登录，并保存到宿主机权限为 600 的
`~/.netrc`，后续训练可以复用。

登录和启用 W&B 的训练容器使用宿主机网络，并继承终端中的
`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY` 及其小写版本。
这样可以访问指向 `127.0.0.1` 的宿主机代理；代理值不会打印到启动命令中。

如果宿主机没有安装 W&B CLI，可在终端隐藏输入 API key：

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
```

以下是四卡、1,000 步的样本训练示例：

```bash
bash scripts/start_train.sh \
  --wandb --gpus 0,1,2,3 --steps 1000 \
  --eval-every 50 --save-every 200 \
  --data-dir /home/zhangrui/comp/data/splits_v2-tok32k-smoke-ctx1024-v1 \
  --runs-dir "$PWD/runs" \
  --run-name wb-smoke-4gpu-1000
```

默认使用配置中的 `echo-lab/scicomp26`，可用 `--wandb-entity` 和
`--wandb-project` 覆盖。每次使用新的 `--run-name`；添加 `--dry-run`
可查看启动命令而不启动训练。

这里只训练发布的 10,000 条 train / 1,000 条 dev 样本；增加步数不会切换到全量语料。
DDP 只在 rank 0 创建 W&B run，上传训练与验证 loss、学习率、step 耗时和
token 吞吐，同时保留本地日志与 checkpoint。W&B run 链接保存在输出目录的
`wandb-run.json` 中。显式启用 W&B 后，初始化失败会终止训练。
