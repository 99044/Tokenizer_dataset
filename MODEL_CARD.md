# SCICOMP26 Model Card

## 1. 队伍与架构

- 队伍名称及编号：<待填写>
- 架构名称：`olmo_hybrid_250m`
- 代码入口：模型构建位于 `training/src/model_config.py`，多卡训练位于
  `training/src/train_ddp.py`，四档配置位于 `training/configs/FLOPs_*.yaml`。
- 架构：16 层 OLMo Hybrid，每四层采用 `[gdn, gdn, gdn, attn]`；每层保留
  Dense SwiGLU FFN，attention 层使用全局因果注意力。
- 公共模型参数：hidden size 896，FFN intermediate size 3584，attention/KV heads 7，
  attention head dimension 128；GDN heads/value heads 7，GDN head dimension 96，
  value expand 2.0，conv size 4。
- 公共训练参数：32768 词表，8192 上下文，BF16，AdamW，peak LR 2.83e-4，
  warmup 20 步，主训练段保持 constant LR，最后约 16.8% 的更新做 linear cooldown 到 0。
- 四个预算档使用同一个 254,467,368 参数模型，只改变训练步数和与总步数成比例的
  cooldown 步数。

## 2. 四档训练计划

按两张 GPU、每卡 micro batch 1、梯度累积 32 次计算，每次 optimizer update 处理
524,288 个输入 token。当前 Hybrid 逐算子核算口径为
`1,800,713,490,309 / 1,024 = 1,758,509,267.8798828 FLOPs/token`。

| model_id | 目标预算 | 计划训练 FLOPs | 更新步数 | 累计训练 token | cooldown 步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLOPs_3e17 | 300,000,000,000,000,000 | 299,638,724,787,417,600 | 325 | 170,393,600 | 55 |
| FLOPs_1e18 | 1,000,000,000,000,000,000 | 999,410,392,829,417,472 | 1,084 | 568,328,192 | 182 |
| FLOPs_3e18 | 3,000,000,000,000,000,000 | 2,999,153,143,795,290,624 | 3,253 | 1,705,508,864 | 547 |
| FLOPs_1e19 | 10,000,000,000,000,000,000 | 9,999,635,720,136,403,968 | 10,846 | 5,686,427,648 | 1,823 |

总参数量为 254,467,368。排除共享的输入 embedding/输出词表投影后，非词表参数量为
225,107,240；当前没有 MoE 或 token 级稀疏路由，因此所有非词表参数均参与每个 token 的
计算。最终提交时以实际完成的 checkpoint step 和完整训练日志替换计划值。

## 3. 数据与导出状态

- 四档复用 `tok_bpe32k_v1` 的 32768 词表，训练数据和 tokenizer 路径由 infra 在启动时传入。
- 未指定 `--resume` 或 `--branch-from` 时，四档均从随机初始化开始训练。
- `models/` 当前只有目录占位；最终权重、完整 tokenizer、HF 自定义模型代码与离线加载验证
  由训练和提交阶段补齐。
- 训练 FLOPs 可由 `training/count_flops.py --auto-hybrid --scalar-op-cost 1` 复算；
  四档 YAML 的静态一致性由 `training/validate_250m_configs.py` 检查。
