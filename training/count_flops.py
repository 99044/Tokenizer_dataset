#!/usr/bin/env python3
"""按 SCICOMP26 PDF 指定的 Megatron-LM 参考口径核算训练 FLOPs。

默认入口仅支持固定 8192 长度的标准稠密 Transformer。
Hybrid/GDN/信号使用 --auto-hybrid 自动生成完整的版本化逐算子参考计算图，
或使用 --operator-ledger 提供自定义清单。未核定算子不会退化为 6P 或统一乘 3。

必须注意：比赛指定的 ``num_floating_point_operations`` 不接收滑动窗口参数；
其标准 Transformer 分支对所有注意力层统一使用 ``batch_size * seq_length²``。
因此本脚本也严格按该函数核算，不自行按 4096 窗口折减注意力 FLOPs。
使用 --plan-only 可在正式训练前规划预算，无需伪造训练日志。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# 赛事 PDF 的四档 IsoFLOPs 预算及完整文件哈希。
PDF_SHA256 = "ca89a5506ecf7cdfc1d97716fe9b0455aad02f969476996565057aed964be935"
PDF_BUDGETS = (
    300_000_000_000_000_000,
    1_000_000_000_000_000_000,
    3_000_000_000_000_000_000,
    10_000_000_000_000_000_000,
)
PDF_TOLERANCE_NUMERATOR = 1
PDF_TOLERANCE_DENOMINATOR = 100

# 项目负责人确认的当前训练/核算序列长度。不再兼容旧的短序列运行。
REQUIRED_SEQUENCE_LENGTH = 8192

# PDF 指向 Megatron-LM training.py 的 num_floating_point_operations。
# PDF 没有给出 Git commit；下面的提交只用于记录本实现核对时所见的参考源码版本，
# 不能被误写成赛事官方锁定的版本。
MEGATRON_REFERENCE_FUNCTION_SHA256 = (
    "4d6e6daf89926f83f4e5009f337540cf1f16850e1b18d9b1c6cb6fb74b93fc97"
)


@dataclass(frozen=True)
class DenseTransformerSpec:
    """Megatron 参考公式计算标准稠密 Transformer 所需的最小结构参数。"""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    intermediate_size: int
    sequence_length: int
    padded_vocab_size: int
    activation: str

    def validate(self) -> None:
        """拒绝无法严格映射到 Megatron 标准 Transformer 分支的配置。"""

        for field_name, value in asdict(self).items():
            if field_name == "activation":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_query_groups > self.num_attention_heads:
            raise ValueError("num_query_groups cannot exceed num_attention_heads")
        if self.num_attention_heads % self.num_query_groups != 0:
            raise ValueError("num_attention_heads must be divisible by num_query_groups")
        if self.activation not in {"swiglu", "gelu"}:
            raise ValueError(
                "only explicit 'swiglu' or 'gelu' is supported; other activations require "
                "project-specific FLOPs accounting"
            )


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，供训练材料审计与结果复算使用。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8_388_608), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    """读取严格正整数；布尔值不能冒充整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    """读取顶层必须为对象的 JSON 文件。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def audit_megatron_reference(path: Path) -> dict[str, Any]:
    """定位用户提供的参考函数，并锁定其规范化源码哈希。"""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "num_floating_point_operations"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Megatron reference must contain exactly one top-level "
            "num_floating_point_operations function"
        )
    function = matches[0]
    function_source = "\n".join(
        source.splitlines()[function.lineno - 1 : function.end_lineno]
    ) + "\n"
    function_sha256 = hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    if function_sha256 != MEGATRON_REFERENCE_FUNCTION_SHA256:
        raise ValueError(
            "provided Megatron num_floating_point_operations differs from the "
            "reviewed reference; review and version the FLOPs implementation before use"
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "function": "num_floating_point_operations",
        "function_start_line": function.lineno,
        "function_end_line": function.end_lineno,
        "normalized_function_sha256": function_sha256,
        "matches_reviewed_reference": True,
        "official_git_commit_pinned_by_pdf": False,
    }


def model_spec_from_config(
    model: dict[str, Any], num_query_groups: int, padded_vocab_size: int
) -> DenseTransformerSpec:
    """Map supported model configurations without changing their architecture type.

    OLMo3 normalization/activation elementwise operations follow the supplied
    reference's exclusions. The supplied reference has no sliding-window input,
    so its standard Transformer branch uses the declared sequence length squared
    for every attention layer.
    """
    if model.get("type") not in {"dense_decoder_only", "olmo3_1b"}:
        raise ValueError("only dense_decoder_only and olmo3_1b are supported")
    sequence_length = _positive_int(model.get("context_length"), "context_length")
    if sequence_length != REQUIRED_SEQUENCE_LENGTH:
        raise ValueError(
            f"context_length must be {REQUIRED_SEQUENCE_LENGTH}; "
            "the old short-sequence FLOPs path is not supported"
        )
    if model.get("type") == "olmo3_1b":
        expected = {
            "num_layers": 16, "hidden_size": 2048, "num_attention_heads": 16,
            "num_key_value_heads": 16, "head_dim": 128, "intermediate_size": 8192,
            "activation": "swiglu", "normalization_style": "reordered_norm",
            "rms_norm_eps": 1e-6, "qk_norm": True, "rope_theta": 500000,
            "sliding_window_pattern": [4096, 4096, 4096, -1],
            "force_full_attention_on_first_layer": False,
            "force_full_attention_on_last_layer": True,
            "tie_word_embeddings": True,
        }
        for name, value in expected.items():
            if model.get(name) != value:
                raise ValueError(f"olmo3_1b preset field {name} does not match the training builder")
    windows = model.get("sliding_window_pattern")
    if windows is not None:
        if not isinstance(windows, list) or not windows:
            raise ValueError("sliding_window_pattern must be a nonempty list")
        for window in windows:
            if isinstance(window, bool) or not isinstance(window, int):
                raise ValueError("attention windows must be integers")
            if window != -1 and window <= 0:
                raise ValueError("attention windows must be positive or -1 for full attention")
    num_query_groups = _positive_int(num_query_groups, "num_query_groups")
    declared = model.get("num_key_value_heads")
    if declared is not None and declared != num_query_groups:
        raise ValueError("--num-query-groups conflicts with model.num_key_value_heads")
    padded_vocab_size = _positive_int(padded_vocab_size, "padded_vocab_size")
    if model.get("vocab_size") is not None:
        if padded_vocab_size < _positive_int(model["vocab_size"], "model.vocab_size"):
            raise ValueError("padded_vocab_size cannot be smaller than model.vocab_size")
    spec = DenseTransformerSpec(
        num_layers=_positive_int(model.get("num_layers"), "num_layers"),
        hidden_size=_positive_int(model.get("hidden_size"), "hidden_size"),
        num_attention_heads=_positive_int(model.get("num_attention_heads"), "num_attention_heads"),
        num_query_groups=num_query_groups,
        head_dim=_positive_int(model.get("head_dim"), "head_dim"),
        intermediate_size=_positive_int(model.get("intermediate_size"), "intermediate_size"),
        sequence_length=sequence_length,
        padded_vocab_size=padded_vocab_size,
        activation=str(model.get("activation", "")).lower(),
    )
    spec.validate()
    return spec


def load_and_validate_inputs(
    model_config_path: Path,
    run_plan_path: Path,
    num_query_groups: int,
    padded_vocab_size: int | None,
    data_parallel_size: int,
) -> tuple[DenseTransformerSpec, dict[str, Any], dict[str, Any]]:
    """读取模型与训练计划，并核对会影响 FLOPs 的重复字段。"""

    import yaml

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict):
        raise ValueError("model YAML must contain an object")
    run_plan = _load_json(run_plan_path)

    yaml_model = model_config.get("model")
    plan_model = run_plan.get("settings", {}).get("model")
    if not isinstance(yaml_model, dict) or not isinstance(plan_model, dict):
        raise ValueError("both model YAML and run-plan must contain a model object")

    # 这些字段会直接改变计算图；YAML 与实际运行计划不一致时不得继续计算。
    compared_fields = (
        "type",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "intermediate_size",
        "activation",
        "context_length",
        "num_key_value_heads",
        "sliding_window_pattern",
        "force_full_attention_on_first_layer",
        "force_full_attention_on_last_layer",
        "normalization_style",
        "qk_norm",
        "tie_word_embeddings",
        "vocab_size",
    )
    for field_name in compared_fields:
        if yaml_model.get(field_name) != plan_model.get(field_name):
            raise ValueError(
                f"model field {field_name!r} differs between YAML "
                f"({yaml_model.get(field_name)!r}) and run-plan ({plan_model.get(field_name)!r})"
            )

    plan_data = run_plan.get("data")
    if not isinstance(plan_data, dict):
        raise ValueError("run-plan.data must be an object")
    sequence_length = _positive_int(yaml_model.get("context_length"), "model.context_length")
    if _positive_int(plan_data.get("context_length"), "run-plan.data.context_length") != sequence_length:
        raise ValueError("model and data context lengths differ")

    plan_vocab_size = _positive_int(plan_data.get("vocab_size"), "run-plan.data.vocab_size")
    if padded_vocab_size is None:
        padded_vocab_size = plan_vocab_size
    else:
        padded_vocab_size = _positive_int(padded_vocab_size, "padded_vocab_size")
        if padded_vocab_size < plan_vocab_size:
            raise ValueError("padded_vocab_size cannot be smaller than the tokenizer vocabulary")

    # KV/query-group 数决定 GQA 投影计算量。原配置没有该字段时必须由调用者显式给出；
    # 本脚本绝不依据模型名称猜测 MHA/GQA。
    num_query_groups = _positive_int(num_query_groups, "num_query_groups")
    declared_query_groups = yaml_model.get("num_key_value_heads")
    if declared_query_groups is not None and declared_query_groups != num_query_groups:
        raise ValueError("--num-query-groups conflicts with model.num_key_value_heads")

    world_size = _positive_int(run_plan.get("world_size"), "run-plan.world_size")
    data_parallel_size = _positive_int(data_parallel_size, "data_parallel_size")
    if data_parallel_size > world_size or world_size % data_parallel_size != 0:
        raise ValueError("data_parallel_size must divide world_size")

    # 根级字段是 CLI/自动预算展开后的运行值；settings.training 保留 YAML 默认值，
    # 两者可以不同。实际 batch 由根级字段核算。
    plan_training = run_plan.get("settings", {}).get("training")
    if not isinstance(plan_training, dict):
        raise ValueError("run-plan.settings.training must be an object")
    micro_batch_size = _positive_int(run_plan.get("micro_batch_size"), "run-plan.micro_batch_size")
    grad_accum_steps = _positive_int(run_plan.get("grad_accum_steps"), "run-plan.grad_accum_steps")
    spec = model_spec_from_config(yaml_model, num_query_groups, padded_vocab_size)

    runtime = {
        "world_size": world_size,
        "data_parallel_size": data_parallel_size,
        "model_parallel_size": world_size // data_parallel_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "global_batch_size": data_parallel_size * micro_batch_size * grad_accum_steps,
        "batch_basis": "resolved run-plan root fields; YAML training fields are defaults",
    }
    return spec, runtime, run_plan


def megatron_dense_flops_per_step(
    spec: DenseTransformerSpec, global_batch_size: int
) -> dict[str, int]:
    """逐项复现 Megatron 标准 Transformer 分支的一次训练步 FLOPs。

    计数常量直接对应 PDF：一次乘或加各 1 FLOP，所以 GEMM 使用 2mnk；标准
    矩阵乘的前向、权重梯度和输入梯度合计为前向的 3 倍。注意力因果掩码的
    1/2 与 QK^T、AV 两次矩阵乘恰好抵消。
    """

    spec.validate()
    global_batch_size = _positive_int(global_batch_size, "global_batch_size")

    training_factor = 3
    gemm_fma_factor = 2
    total_tokens = global_batch_size * spec.sequence_length
    # 官方指定函数的 BSHD 默认值：batch_size * llm_seq_length²。
    # 该函数没有滑动窗口参数，因此这里不能自行用窗口 4096 折减。
    sequence_squared_sum = global_batch_size * spec.sequence_length**2
    query_projection_size = spec.head_dim * spec.num_attention_heads
    key_value_projection_size = spec.head_dim * spec.num_query_groups
    ffn_projection_count = 3 if spec.activation == "swiglu" else 2

    # Q、K、V 和输出投影；GQA 通过 num_query_groups 缩小 K/V 投影。
    attention_projection = (
        total_tokens
        * spec.num_layers
        * training_factor
        * gemm_fma_factor
        * (
            spec.hidden_size
            * (query_projection_size + 2 * key_value_projection_size)
            + query_projection_size * spec.hidden_size
        )
    )
    # 因果注意力只计算半个矩阵，但包含 QK^T 和 AV 两次 GEMM。
    attention_core = (
        sequence_squared_sum
        * spec.num_layers
        * training_factor
        * gemm_fma_factor
        * query_projection_size
    )
    # SwiGLU 有 gate/up/down 三个投影，普通 GELU MLP 有 up/down 两个投影。
    mlp = (
        total_tokens
        * spec.num_layers
        * training_factor
        * gemm_fma_factor
        * spec.hidden_size
        * spec.intermediate_size
        * ffn_projection_count
    )
    # 即使输入嵌入与输出权重绑定，生成完整词表 logits 的矩阵乘仍然存在。
    logits = (
        total_tokens
        * training_factor
        * gemm_fma_factor
        * spec.hidden_size
        * spec.padded_vocab_size
    )
    total = attention_projection + attention_core + mlp + logits
    return {
        "attention_projection": attention_projection,
        "attention_core": attention_core,
        "mlp": mlp,
        "logits": logits,
        "total": total,
    }


def audit_metrics(
    metrics_path: Path, checkpoint_step: int, expected_tokens_per_step: int
) -> dict[str, Any]:
    """确认日志从第 1 步连续覆盖目标 checkpoint，并核对每步处理 token 数。"""

    checkpoint_step = _positive_int(checkpoint_step, "checkpoint_step")
    expected_tokens_per_step = _positive_int(expected_tokens_per_step, "expected_tokens_per_step")
    previous_step = 0
    max_absolute_token_error = 0.0
    rows = 0

    with metrics_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"metrics line {line_number}: JSON object required")
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int):
                raise ValueError(f"metrics line {line_number}: integer step required")
            if step != previous_step + 1:
                raise ValueError(
                    f"metrics line {line_number}: expected step {previous_step + 1}, got {step}; "
                    "gaps/restarts require a separate audited training-path reconstruction"
                )
            if step > checkpoint_step:
                raise ValueError("metrics contains steps after the requested checkpoint")

            seconds = row.get("train/step_seconds")
            tokens_per_second = row.get("train/tokens_per_second")
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"metrics line {line_number}: positive train/step_seconds required")
            if (
                not isinstance(tokens_per_second, (int, float))
                or not math.isfinite(tokens_per_second)
                or tokens_per_second <= 0
            ):
                raise ValueError(
                    f"metrics line {line_number}: positive train/tokens_per_second required"
                )
            logged_tokens = float(seconds) * float(tokens_per_second)
            token_error = abs(logged_tokens - expected_tokens_per_step)
            max_absolute_token_error = max(max_absolute_token_error, token_error)
            if not math.isclose(
                logged_tokens,
                expected_tokens_per_step,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"metrics line {line_number}: timing fields imply {logged_tokens:.9f} tokens, "
                    f"expected {expected_tokens_per_step}"
                )
            previous_step = step
            rows += 1

    if previous_step != checkpoint_step or rows != checkpoint_step:
        raise ValueError(
            f"metrics ends at step {previous_step} with {rows} rows; "
            f"checkpoint step {checkpoint_step} requires a complete 1..{checkpoint_step} trace"
        )
    return {
        "complete_contiguous_trace": True,
        "first_step": 1,
        "last_step": previous_step,
        "rows": rows,
        "expected_tokens_per_step": expected_tokens_per_step,
        "max_absolute_tokens_implied_by_timing_error": max_absolute_token_error,
    }


def budget_report(actual_flops: int, per_step_flops: int) -> list[dict[str, Any]]:
    """按 PDF 的 ±1% 闭区间检查四档预算，并给出对应整数训练步范围。"""

    actual_flops = _positive_int(actual_flops, "actual_flops")
    per_step_flops = _positive_int(per_step_flops, "per_step_flops")
    reports = []
    for target in PDF_BUDGETS:
        lower = target * (PDF_TOLERANCE_DENOMINATOR - PDF_TOLERANCE_NUMERATOR) // PDF_TOLERANCE_DENOMINATOR
        upper = target * (PDF_TOLERANCE_DENOMINATOR + PDF_TOLERANCE_NUMERATOR) // PDF_TOLERANCE_DENOMINATOR
        first_valid_step = (lower + per_step_flops - 1) // per_step_flops
        last_valid_step = upper // per_step_flops
        nearest_step = (target + per_step_flops // 2) // per_step_flops
        reports.append(
            {
                "target_flops": target,
                "lower_inclusive": lower,
                "upper_inclusive": upper,
                "actual_flops": actual_flops,
                "relative_error": actual_flops / target - 1.0,
                "within_pdf_tolerance": lower <= actual_flops <= upper,
                "first_valid_integer_step": first_valid_step,
                "last_valid_integer_step": last_valid_step,
                "nearest_integer_step": nearest_step,
                "nearest_integer_step_flops": nearest_step * per_step_flops,
            }
        )
    return reports


def positive_flops(value: str) -> int:
    """Accept exact integer scientific notation such as 1e19, without floats."""
    try:
        number = Decimal(value)
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            raise ValueError("target FLOPs must be a positive integer")
        return int(number)
    except (InvalidOperation, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def plan_training_budget(args: argparse.Namespace) -> dict[str, Any]:
    """Produce a pretraining plan; do not fabricate completed steps or logs."""
    import yaml

    config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("model YAML must contain a model object")
    spec = model_spec_from_config(config["model"], args.num_query_groups, args.padded_vocab_size)
    dp = _positive_int(args.data_parallel_size, "data_parallel_size")
    micro = _positive_int(args.micro_batch_size, "micro_batch_size")
    accumulation = _positive_int(args.grad_accum_steps, "grad_accum_steps")
    target = _positive_int(args.target_flops, "target_flops")
    batch = dp * micro * accumulation
    tokens_per_step = batch * spec.sequence_length
    per_step = megatron_dense_flops_per_step(spec, batch)
    cost = per_step["total"]
    floor_steps = target // cost
    nearest_steps = (target + cost // 2) // cost
    steps = floor_steps if args.rounding == "floor" else nearest_steps
    if steps < 1:
        raise ValueError("budget is smaller than one optimizer step; reduce the global batch")
    planned = cost * steps
    reference = audit_megatron_reference(args.megatron_reference) if args.megatron_reference else None
    lower = target * 99 // 100
    upper = target * 101 // 100
    if not lower <= planned <= upper:
        raise ValueError(
            "planned FLOPs fall outside the PDF ±1% interval; reduce the per-step "
            "global batch or choose a valid optimizer-step count"
        )
    return {
        "record_kind": "planned_training_budget_not_completed_training",
        "protocol": "scicomp26-pdf-megatron-reference-dense-8192-v2",
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "official_result": False,
        "megatron_reference": reference,
        "reference_source_audited": reference is not None,
        "pdf_sha256_declared_by_supplied_script": PDF_SHA256,
        "pdf_independently_audited": False,
        "model_config": {"path": str(args.model_config.resolve()), "sha256": sha256_file(args.model_config)},
        "model": asdict(spec),
        "runtime": {"data_parallel_size": dp, "micro_batch_size": micro,
                    "gradient_accumulation_steps": accumulation, "global_batch_size": batch},
        "flops_per_token": cost // tokens_per_step,
        "per_step_flops": per_step,
        "tokens_per_step": tokens_per_step,
        "target_training_flops": target,
        "rounding": args.rounding,
        "optimizer_steps": steps,
        "planned_training_tokens": steps * tokens_per_step,
        "planned_training_flops": planned,
        "relative_error_percent": str(Decimal(planned - target) * 100 / target),
        "within_target": planned <= target,
        "within_supplied_one_percent_tolerance": lower <= planned <= upper,
        "exact_target_at_fixed_step_possible": target % cost == 0,
        "exact_target_at_whole_token_possible": target % (cost // tokens_per_step) == 0,
        "floor_optimizer_steps": floor_steps,
        "floor_training_flops": floor_steps * cost,
        "nearest_optimizer_steps": nearest_steps,
        "nearest_training_flops": nearest_steps * cost,
        "allowed_optimizer_steps_under_supplied_tolerance": {
            "first": (lower + cost - 1) // cost, "last": upper // cost},
        "launcher_arguments": ["--stage", "stage1", "--config", str(args.model_config),
                               "--training-tokens", str(steps * tokens_per_step),
                               "--steps", str(steps), "--micro-batch-size", str(micro),
                               "--grad-accum-steps", str(accumulation)],
        "notes": [
            "Planning only: no completed training or checkpoint is claimed.",
            "Norm, RoPE, softmax and activation elementwise FLOPs follow the supplied reference exclusions.",
            "Sequence length is fixed to 8192; the old short-sequence path is rejected.",
            "The supplied Megatron function has no sliding-window argument and applies S^2 to every standard attention layer; this script does the same without a project-specific correction.",
            "The token budget covers all formal stages combined; reserve Stage 2 from this total if used.",
            "GPU count, batch, vocabulary width or sequence length changes require recalculation.",
        ],
    }


def calculate(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整的配置、日志、公式和预算核算。"""

    for path in (args.model_config, args.run_plan, args.metrics, args.megatron_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_step = _positive_int(args.checkpoint_step, "checkpoint_step")
    megatron_reference = audit_megatron_reference(args.megatron_reference)

    spec, runtime, run_plan = load_and_validate_inputs(
        args.model_config,
        args.run_plan,
        args.num_query_groups,
        args.padded_vocab_size,
        args.data_parallel_size,
    )
    max_steps = _positive_int(run_plan.get("max_steps"), "run-plan.max_steps")
    if checkpoint_step > max_steps:
        raise ValueError("checkpoint_step exceeds run-plan.max_steps")

    tokens_per_step = runtime["global_batch_size"] * spec.sequence_length
    metrics_audit = audit_metrics(args.metrics, checkpoint_step, tokens_per_step)
    per_step = megatron_dense_flops_per_step(spec, runtime["global_batch_size"])
    cumulative = {name: value * checkpoint_step for name, value in per_step.items()}
    budgets = budget_report(cumulative["total"], per_step["total"])

    return {
        "protocol": "scicomp26-pdf-megatron-reference-dense-8192-v2",
        "official_result": False,
        "official_result_reason": (
            "This is a reproducible local calculation; the organiser performs the official "
            "architecture, step, data-volume, and run-record audit."
        ),
        "pdf_sha256": PDF_SHA256,
        "pdf_rules_applied": {
            "addition_or_multiplication": "1 FLOP",
            "fused_multiply_add": "2 FLOPs",
            "matrix_multiply_mn_nk": "2mnk FLOPs",
            "standard_gemm_forward_and_backward": "3 times forward GEMM FLOPs",
            "precision": "BF16/FP16/FP8 do not change logical FLOPs",
            "standard_attention_sequence_term": (
                "batch_size * sequence_length^2 for every standard attention layer, "
                "exactly as the supplied Megatron function; sequence_length=8192"
            ),
            "cumulative_scope": "sum the model computation graph over every completed training step",
            "excluded": [
                "optimizer update",
                "activation recomputation used to save memory",
                "communication overhead",
            ],
        },
        "megatron_reference": megatron_reference,
        "supported_scope": "standard dense decoder-only Transformer at sequence length 8192",
        "model_config": {
            "path": str(args.model_config.resolve()),
            "sha256": sha256_file(args.model_config),
        },
        "run_plan": {
            "path": str(args.run_plan.resolve()),
            "sha256": sha256_file(args.run_plan),
        },
        "metrics": {
            "path": str(args.metrics.resolve()),
            "sha256": sha256_file(args.metrics),
            "audit": metrics_audit,
        },
        "model": asdict(spec),
        "runtime": runtime,
        "checkpoint_step": checkpoint_step,
        "processed_training_tokens": checkpoint_step * tokens_per_step,
        "per_step_flops": per_step,
        "cumulative_flops": cumulative,
        "actual_training_flops": cumulative["total"],
        "actual_training_flops_scientific": f"{cumulative['total']:.12e}",
        "budget_checks": budgets,
        "eligible_pdf_budget": next(
            (report["target_flops"] for report in budgets if report["within_pdf_tolerance"]),
            None,
        ),
        "audit_notes": [
            "num_query_groups is explicit and is checked against any declared KV heads.",
            "padded vocabulary defaults to run-plan.data.vocab_size unless explicitly overridden.",
            "The metrics trace must contain exactly one consecutive record for every step from 1.",
            "Training validation passes are not added; Megatron's cumulative training counter advances on training batches.",
            "Embedding lookup, normalization, activation, RoPE, softmax, and loss elementwise FLOPs are not extra terms in the referenced standard Transformer formula.",
            "The supplied Megatron function does not inspect sliding-window settings; no project-specific window correction is applied.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """声明严格核算所需的全部输入；不提供会静默猜测架构的默认值。"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Hybrid graph modes: --auto-hybrid --scalar-op-cost COST, --write-ledger-template PATH or --operator-ledger PATH; "
               "add --help with either mode to see its arguments.",
    )
    parser.add_argument("--plan-only", action="store_true", help="plan a budget before training; no run-plan or metrics required")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--megatron-reference",
        type=Path,
        help="provided training.py/text containing the reviewed num_floating_point_operations",
    )
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        required=True,
        help="actual DP degree; world_size alone is insufficient when TP/PP/CP may be present",
    )
    parser.add_argument(
        "--num-query-groups",
        type=int,
        required=True,
        help="number of KV/query groups; use attention-head count for ordinary MHA",
    )
    parser.add_argument(
        "--padded-vocab-size",
        type=int,
        help="actual output projection width; defaults to run-plan tokenizer vocabulary size",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-flops", type=positive_flops, help="planning target, e.g. 1e19")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, help="resolved accumulation for budget planning")
    parser.add_argument("--rounding", choices=["nearest", "floor"], default="nearest",
                        help="nearest uses the supplied tolerance; floor stays within the target")
    return parser


def main() -> None:
    """运行核算、原子化保存 JSON，并打印便于人工复核的摘要。"""

    if any(argument.split("=", 1)[0] in {"--auto-hybrid", "--operator-ledger", "--write-ledger-template"}
           for argument in sys.argv[1:]):
        from src.graph_accounting import main as graph_accounting_main

        graph_accounting_main()
        return
    if any(argument.split("=", 1)[0] == "--flops-per-token" for argument in sys.argv[1:]):
        from src.token_flops import main as token_flops_main

        token_flops_main()
        return
    parser = build_parser()
    args = parser.parse_args()
    required = (["padded_vocab_size", "target_flops", "grad_accum_steps"] if args.plan_only
                else ["run_plan", "metrics", "megatron_reference", "checkpoint_step"])
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for this mode")
    result = plan_training_budget(args) if args.plan_only else calculate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(args.output)
    if args.plan_only:
        print(json.dumps({name: result[name] for name in [
            "record_kind", "flops_per_token", "tokens_per_step", "optimizer_steps",
            "planned_training_tokens", "planned_training_flops", "relative_error_percent",
            "within_target", "within_supplied_one_percent_tolerance", "launcher_arguments",
        ]}, ensure_ascii=False, indent=2))
        return
    print(
        json.dumps(
            {
                "checkpoint_step": result["checkpoint_step"],
                "processed_training_tokens": result["processed_training_tokens"],
                "flops_per_step": result["per_step_flops"]["total"],
                "actual_training_flops": result["actual_training_flops"],
                "actual_training_flops_scientific": result["actual_training_flops_scientific"],
                "eligible_pdf_budget": result["eligible_pdf_budget"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
