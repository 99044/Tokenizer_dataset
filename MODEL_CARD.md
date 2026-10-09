# SCICOMP26 Model Card

## 1. 队伍与架构

- 队伍名称及编号：<待填写>
- 架构名称及代码版本：`olmo_hybrid_1b`；迁移前算法版本 `c3caeee2a5f83493c0d7b697f7bc3f274ca03380`。
  模型构建为 `training/src/model_config.py:build_model_config()`，
  多卡训练为 `training/src/train_ddp.py`；本次只整理目录及外围入口。
- 相对于主流架构的实质改动与设计依据：使用三层 Gated Delta Net（GDN）
  加一层全局因果 attention 的混合结构，保留各层 SwiGLU FFN。
  GDN 与全局 attention 按固定周期交替，不增加信号门控。
- 四个模型共用的架构规则：16 层，hidden size 1792，FFN intermediate size 7168；
  reordered RMSNorm（epsilon 1e-6）、QK norm、RoPE（theta 500000）、残差连接；
  block pattern 为 `[gdn, gdn, gdn, attn]`，attention heads / KV heads 均为 14，
  attention head dimension 为 128。GDN heads / value heads 均为 14，head dimension 为 96，
  value expand 2.0，conv size 4，norm epsilon 1e-5，允许负特征值。
  共享输入词表与输出投影权重，dropout 为 0，dtype 为 bfloat16。
- 可缩放超参数及缩放规则：四档现有 YAML 使用相同模型结构，预算、运行名称和输出目录不同；
  本次保留原配置，不改变层数、宽度、attention/GDN 配置或信号规则。
  训练与声明评测序列长度均为 8192。

## 2. 四档模型与评分相关参数

下表需在完成本分支训练、核算及导出后填写，不引用 `infra` 分支的 OLMo3 权重或训练统计。

| model_id | 模型目录 | 模型 FLOPs/token | 实际训练 FLOPs C_i | 优化器更新步数 | 累计训练 token 数 |
| --- | --- | --- | --- | --- | --- |
| FLOPs_3e17 | models/FLOPs_3e17/ | <待核算> | <待记录> | <待记录> | <待记录> |
| FLOPs_1e18 | models/FLOPs_1e18/ | <待核算> | <待记录> | <待记录> | <待记录> |
| FLOPs_3e18 | models/FLOPs_3e18/ | <待核算> | <待记录> | <待记录> | <待记录> |
| FLOPs_1e19 | models/FLOPs_1e19/ | <待核算> | <待记录> | <待记录> | <待记录> |

| model_id | 非词表总参数 N | 激活参数量 N_a | 序列长度 |
| --- | --- | --- | --- |
| FLOPs_3e17 | <待统计> | <待统计> | 8192 |
| FLOPs_1e18 | <待统计> | <待统计> | 8192 |
| FLOPs_3e18 | <待统计> | <待统计> | 8192 |
| FLOPs_1e19 | <待统计> | <待统计> | 8192 |

- 参数统计依据：导出后由 `evaluation/eval.py:active_parameter_count()`
  排除输入 embedding 与输出词表投影；按参数对象身份去重，共享权重只排除一次。
  `training/src/model_config.py:count_parameters()` 仅统计模型总参数，不能直接替代非词表 N。
- 条件计算架构的激活统计：当前配置没有 MoE 或跳过计算的稀疏路由，`N_a=N`。
  固定 GDN/attention 周期不属于 token 级稀疏路由。
- FLOPs 核算依据：`training/count_flops.py` 从 `infra` 保留标准稠密 Transformer 核算代码，
  会拒绝本分支的 Hybrid 类型。GDN的 FLOPs/token
  须另行核算后填写；`training/src/token_flops.py` 只对显式提供的已审计值做累计乘法。
  记录位置与运行步骤见 `REPRODUCTION.md`；预算名称不代表已完成预算验证。
- 分词器：复用官方训练子集生成的 `tok_bpe32k_v1`，Byte-level BPE，词表 32768；
  来源为 `/home/pretrain10/dolma3_official/tokenizer_v1_bpe32k/`，
  UNK/BOS/EOS/PAD ID 分别为 0/1/2/3，四档共用，方法见 `training/src/docs/tokenizer_v1.md`。
  最终须将完整 tokenizer 文件分别放入四个模型目录。
- 预训练声明：训练入口未指定 `--resume` 时从随机初始化开始。
  计划仅使用官方训练集，不加载已有模型权重或外部语言模型蒸馏；
  最终提交前须根据本分支的实际权重、数据来源与全程日志核验该声明。

`models/` 中空文件仅用于目录占位，尚不是可离线加载的 Hybrid HF 模型。
