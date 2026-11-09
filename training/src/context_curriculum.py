"""E16 length schedules derived from each tier's full-length FLOPs plan.

Attention GEMM length deltas follow E15, with E16 residual allowances.
The baseline's compressed-branch 3x-forward estimate and other analytical
accounting limitations are inherited.
"""

from __future__ import annotations

from fractions import Fraction
import math


STAGES = (
    ("short", 2048, Fraction(1, 2)),
    ("medium", 4096, Fraction(1, 5)),
    ("full", 8192, Fraction(3, 10)),
)
ACCOUNTING = {
    "kind": "baseline_matched_attention_gemm_length_proxy",
    "official_ledger": False,
    "length_formula": "c(L)=c(8192)-attention_layers*6*hidden*(8192-L)"
    "-3*4*hidden*(8192//compression_block-L//compression_block)",
    "fraction_basis": "FLOPs budget; final stage receives complete updates from the remainder",
    "limitations": "Analytical estimate inherited from the baseline. Length-dependent softmax "
    "and GDN boundary terms are held fixed. The compressed branch retains its 3x-forward "
    "upper estimate. Optimizer updates, FP32 master copies, communication, activation "
    "recomputation and training-loss softmax/CE are outside this prediction-graph ledger. "
    "Runtime numerical fallback costs require separate reporting.",
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _build_qwen3_moe_curriculum(settings, reference):
    from qwen3_flops import qwen3_moe_cost

    hf = settings["model"]["hf_config"]
    world = _integer(settings["submission"]["data_parallel_size"], "data_parallel_size")
    tokens = _integer(settings["scaling"]["tokens_per_update"], "tokens_per_update")
    budget = _integer(settings["budget"]["total_training_flops"], "total_training_flops")
    context = reference["context_length"]
    if context != 8192 or hf["max_position_embeddings"] != context or tokens % (world * context):
        raise ValueError("Qwen3 MoE curriculum requires an 8192 context and whole batches")
    accumulation = tokens // (world * context)
    full_cost, _ = qwen3_moe_cost(hf, context)
    full_update = full_cost * world * accumulation
    full_steps = budget // full_update
    if (
        Fraction(reference["flops_per_token_numerator"], reference["flops_per_token_denominator"])
        != Fraction(full_cost, context)
        or reference["max_steps"] != full_steps
        or reference["planned_training_flops"] != full_steps * full_update
        or reference["planned_training_tokens"] != full_steps * tokens
    ):
        raise ValueError("Qwen3 MoE full-length FLOPs reference disagrees with its ledger")
    if not 0 <= reference["warmup_steps"] < full_steps or not 0 < reference["cooldown_steps"] < full_steps:
        raise ValueError("Invalid Qwen3 MoE reference scheduler")
    if reference["warmup_steps"] + reference["cooldown_steps"] > full_steps:
        raise ValueError("Qwen3 MoE warmup and cooldown overlap")
    if not 0 <= reference["final_lr_ratio"] <= 1:
        raise ValueError("Invalid Qwen3 MoE final LR ratio")

    update_costs = [
        qwen3_moe_cost(hf, length)[0] * (context // length) * world * accumulation
        for _, length, _ in STAGES
    ]
    target_tokens = settings["budget"].get("target_training_tokens")
    stage_counts = None
    if target_tokens is not None:
        target_tokens = _integer(target_tokens, "target_training_tokens")
        total = (target_tokens + tokens - 1) // tokens
        medium = round(total * STAGES[1][2])
        short_update, medium_update, long_update = update_costs
        # Reserve roughly one fifth of the updates for 4K. Start with the
        # remaining updates at 2K, then move as many as possible to 8K.
        base = (total - medium) * short_update + medium * medium_update
        full = min(total - medium - 1, (budget - base) // (long_update - short_update))
        stage_counts = [total - medium - full, medium, full]
        if min(stage_counts) < 1:
            raise ValueError("The target tokens and all three lengths do not fit the FLOPs budget")

    stages = []
    steps = completed = 0
    for index, (name, length, fraction) in enumerate(STAGES):
        micro_batch_size = context // length
        update_cost = update_costs[index]
        if stage_counts is None:
            allocation = budget * fraction if index < len(STAGES) - 1 else budget - completed
            count = int(allocation // update_cost)
        else:
            count = stage_counts[index]
        if count < 1:
            raise ValueError("Every Qwen3 MoE curriculum stage needs a complete update")
        flops = count * update_cost
        stages.append(dict(
            stage=name, stage_index=index, length=length, micro_batch_size=micro_batch_size,
            grad_accum_steps=accumulation, tokens_per_update=tokens,
            **({"requested_flops_fraction": float(fraction)} if stage_counts is None
               else {"planned_flops_fraction": flops / budget}),
            first_step=steps + 1,
            last_step=steps + count, steps=count, flops_per_update=update_cost,
            flops=flops, cumulative_flops=completed + flops,
        ))
        steps += count
        completed += flops
    return dict(
        kind="qwen3_moe_length_fp32", version=1, world_size=world, tokens_per_update=tokens,
        reference=dict(reference), stages=stages, max_steps=steps,
        planned_training_flops=completed, planned_training_tokens=steps * tokens,
        unused_reference_flops=budget - completed,
        accounting=dict(
            kind="qwen3_moe_prediction_graph",
            official_ledger=False,
            **({} if target_tokens is None else {
                "target_training_tokens": target_tokens,
                "allocation": "whole-update token target; reserve 20% of updates for 4K and maximize 8K within the FLOPs cap",
            }),
            limitations="Excludes expert dispatch, top-k selection, language-model and router auxiliary losses, optimizer, communication and activation recomputation.",
        ),
    )


def _build_curriculum(settings, reference):
    model = settings["model"]
    if model["type"] == "qwen3_moe":
        return _build_qwen3_moe_curriculum(settings, reference)
    tier = settings["submission"]["budget_tier"]
    if tier not in {"3e17", "1e18", "3e18", "1e19"}:
        raise ValueError("Unsupported length + FP32 curriculum tier")
    if model["context_length"] != 8192 or reference["context_length"] != 8192:
        raise ValueError("The curriculum requires an 8192-token baseline")
    world = _integer(settings["submission"]["data_parallel_size"], "data_parallel_size")
    if world != 8:
        raise ValueError("The curriculum requires eight GPUs")
    tokens = _integer(settings["scaling"]["tokens_per_update"], "tokens_per_update")
    if tokens % (world * 8192):
        raise ValueError("tokens_per_update must be divisible by world_size * 8192")
    accumulation = tokens // (world * 8192)
    hidden = _integer(model["hidden_size"], "hidden_size")
    layers = _integer(model["num_layers"], "num_layers")
    pattern = model["block_pattern"]
    if not isinstance(pattern, list) or not pattern or any(item not in {"attn", "gdn"} for item in pattern):
        raise ValueError("Expected an attention/GDN block pattern")
    attention_indices = [index for index in range(layers) if pattern[index % len(pattern)] == "attn"]
    windows = model["attention_window_pattern"]
    if not isinstance(windows, list) or not windows or any(
        isinstance(window, bool) or not isinstance(window, int) or (window <= 0 and window != -1)
        for window in windows
    ):
        raise ValueError("Attention window pattern must contain positive sizes or -1")
    force_first = model["force_full_attention_on_first_layer"]
    force_last = model["force_full_attention_on_last_layer"]
    if not isinstance(force_first, bool) or not isinstance(force_last, bool):
        raise ValueError("Full-attention override flags must be booleans")
    for index in attention_indices:
        if (force_first and index == 0) or (force_last and index == layers - 1):
            continue
        # OLMo-core starts the pattern at layer 1 when the first layer is forced full.
        effective_index = index - int(force_first)
        if windows[effective_index % len(windows)] != -1:
            raise ValueError(
                f"The curriculum FLOPs proxy requires global attention at layer {index}"
            )
    attention_layers = len(attention_indices)
    branch = model.get("compressed_attention")
    block_size = _integer(branch["block_size"], "compression block_size") if branch else None
    if block_size and any(length < block_size for _, length, _ in STAGES):
        raise ValueError("Every curriculum length must contain a compression block")
    cost = Fraction(
        _integer(reference["flops_per_token_numerator"], "baseline FLOPs numerator"),
        _integer(reference["flops_per_token_denominator"], "baseline FLOPs denominator"),
    )
    baseline_steps = _integer(reference["max_steps"], "baseline max_steps")
    budget = _integer(reference["planned_training_flops"], "baseline planned_training_flops")
    if baseline_steps * tokens * cost != budget:
        raise ValueError("Baseline FLOPs, tokens/update and max_steps disagree")
    if reference["planned_training_tokens"] != baseline_steps * tokens:
        raise ValueError("Baseline planned token count disagrees")
    if budget > _integer(settings["budget"]["total_training_flops"], "total_training_flops"):
        raise ValueError("Baseline planned FLOPs exceed the nominal budget")
    accounting = dict(ACCOUNTING)
    if block_size and any(length % block_size for _, length, _ in STAGES):
        accounting["length_formula"] = (
            "c(L)=c(8192)-attention_layers*6*hidden*(8192-L)"
            "-branch_cost(8192)+branch_cost(L); branch_cost uses floor(L/compression_block)"
        )
    has_dual = model.get("dual_residual") is not None
    has_value = model.get("value_residual") is not None
    kind = "e16_dual_length_fp32" if has_dual else "e16_length_fp32"
    if has_dual:
        start_layer = 12 if tier == "1e19" else 8
        expected_layers = 16 if tier == "1e19" else 12
        expected_attention = list(range(3, expected_layers, 4))
        if model.get("dual_residual") != dict(enabled=True, start_layer=start_layer, cross_mix_init=0.25):
            raise ValueError("E16 requires the four-tail-block dual-residual configuration")
        if has_value and model["value_residual"] != dict(enabled=True, first_layer_weight=0.5):
            raise ValueError("Invalid value residual configuration")
        if (
            layers != expected_layers or attention_indices != expected_attention
            or model["num_key_value_heads"] * model["attention_head_dim"] != hidden
            or settings["scaling"].get("additional_residual_parameters") != 18
        ):
            raise ValueError("E16 residual accounting requires its declared model dimensions")
        dual_flops = (64 * (layers - start_layer) + 20) * hidden
        value_flops = (9 * (attention_layers - 1) * model["num_key_value_heads"]
                       * model["attention_head_dim"]) if has_value else 0
        cost += dual_flops + value_flops
        accounting.update(
            kind="e16_baseline_matched_with_residual_upper_bounds",
            length_formula=accounting["length_formula"] + "+residual_extra_flops_per_token",
            additional_residual_parameters=18,
            dual_residual_extra_training_flops_per_token_upper_bound=dual_flops,
            **({"value_residual_extra_training_flops_per_token_upper_bound": value_flops} if has_value else {}),
            residual_extra_flops_per_token=dual_flops + value_flops,
            dual_formula=f"(64 * {layers - start_layer} + 20) * {hidden} = {dual_flops}",
            **({"value_formula": f"9 * {attention_layers - 1} * {hidden} = {value_flops}"} if has_value else {}),
            full_length_flops_per_token_numerator=cost.numerator,
            full_length_flops_per_token_denominator=cost.denominator,
            residual_scope=f"Four dual-stream tail blocks{(' and ' + ('two' if attention_layers == 3 else 'three') + ' value residual connections') if has_value else ''}, "
            "8192 positions per rank microbatch at lengths 2048/4096/8192, no activation "
            "recomputation. Conservative forward/backward arithmetic allowances inherit "
            "the E16 reference and can double-count shared baseline additions.",
        )
    elif has_value or "additional_residual_parameters" in settings["scaling"]:
        raise ValueError("Residual accounting requires the dual-residual branch")
    warmup = Fraction(_integer(reference["warmup_steps"], "baseline warmup_steps", 0), baseline_steps)
    cooldown = Fraction(_integer(reference["cooldown_steps"], "baseline cooldown_steps", 0), baseline_steps)
    if not 0 <= warmup < 1 or not 0 < cooldown < 1 or warmup + cooldown > 1:
        raise ValueError("Invalid baseline warmup/cooldown fractions")
    final_ratio = reference["final_lr_ratio"]
    if isinstance(final_ratio, bool) or not isinstance(final_ratio, (int, float)) or not math.isfinite(final_ratio) or not 0 <= final_ratio <= 1:
        raise ValueError("final_lr_ratio must be finite and between zero and one")
    stages = []
    completed_steps = 0
    completed_flops = 0
    for index, (name, length, fraction) in enumerate(STAGES):
        stage_cost = cost - attention_layers * 6 * hidden * (8192 - length)
        if block_size:
            branch_width = _integer(branch["intermediate_size"], "compression intermediate_size")
            def branch_cost(sequence_length):
                blocks = sequence_length // block_size
                forward = Fraction(
                    4 * blocks * block_size * hidden * hidden + 8 * blocks * hidden * branch_width,
                    sequence_length,
                ) + 4 * hidden * blocks
                return math.ceil(3 * forward)
            stage_cost -= branch_cost(8192) - branch_cost(length)
        per_update = stage_cost * tokens
        if per_update <= 0 or per_update.denominator != 1:
            raise ValueError("Stage update FLOPs must be positive whole numbers")
        per_update = int(per_update)
        allocation = budget - completed_flops if index == len(STAGES) - 1 else budget * fraction
        count = int(allocation // per_update)
        if count <= 0:
            raise ValueError("Budget is too small to provide every curriculum stage")
        stages.append({
            "stage": name,
            "stage_index": index,
            "length": length,
            "micro_batch_size": 8192 // length,
            "grad_accum_steps": accumulation,
            "tokens_per_update": tokens,
            "requested_flops_fraction": float(fraction),
            "first_step": completed_steps + 1,
            "last_step": completed_steps + count,
            "steps": count,
            "flops_per_update": per_update,
            "flops": count * per_update,
            "cumulative_flops": completed_flops + count * per_update,
        })
        completed_steps += count
        completed_flops += count * per_update
    return {
        "kind": kind,
        "version": 1,
        "world_size": world,
        "tokens_per_update": tokens,
        "reference": dict(reference),
        "stages": stages,
        "max_steps": completed_steps,
        "planned_training_flops": completed_flops,
        "planned_training_tokens": completed_steps * tokens,
        "unused_reference_flops": budget - completed_flops,
        "accounting": accounting,
    }


def _steps_at_flops(curriculum, target, *, round_up):
    previous_flops = 0
    for stage in curriculum["stages"]:
        if target <= stage["cumulative_flops"]:
            updates = Fraction(target - previous_flops, stage["flops_per_update"])
            count = math.ceil(updates) if round_up else math.floor(updates)
            return stage["first_step"] - 1 + count
        previous_flops = stage["cumulative_flops"]
    return curriculum["max_steps"]


def _scheduler_fields(curriculum):
    reference = curriculum["reference"]
    warmup = Fraction(reference["warmup_steps"], reference["max_steps"])
    cooldown = Fraction(reference["cooldown_steps"], reference["max_steps"])
    total = curriculum["planned_training_flops"]
    return {
        "warmup_fraction": float(warmup),
        "cooldown_fraction": float(cooldown),
        "warmup_steps": _steps_at_flops(curriculum, total * warmup, round_up=True),
        "cooldown_steps": curriculum["max_steps"] - _steps_at_flops(
            curriculum, total * (1 - cooldown), round_up=False
        ),
    }


def apply_length_curriculum(settings):
    """Derive one curriculum from a newly generated full-8K baseline, in place."""
    training, budget, scheduler = settings["training"], settings["budget"], settings["scheduler"]
    if "length_curriculum" in training:
        raise ValueError("Generate a fresh baseline before applying the curriculum")
    if scheduler["name"] != "linear_cooldown":
        raise ValueError("The curriculum requires the baseline linear cooldown schedule")
    if training["micro_batch_size"] != 1:
        raise ValueError("Expected the baseline micro_batch_size=1")
    reference = {
        "context_length": settings["model"]["context_length"],
        "max_steps": training["max_steps"],
        "planned_training_flops": budget["planned_training_flops"],
        "planned_training_tokens": budget["planned_training_tokens"],
        "flops_per_token_numerator": budget["flops_per_token_numerator"],
        "flops_per_token_denominator": budget["flops_per_token_denominator"],
        "warmup_steps": scheduler["warmup_steps"],
        "cooldown_steps": scheduler["cooldown_steps"],
        "final_lr_ratio": scheduler["final_lr_ratio"],
    }
    curriculum = _build_curriculum(settings, reference)
    accumulation = curriculum["stages"][-1]["grad_accum_steps"]
    if training["gradient_accumulation_steps"] != accumulation:
        raise ValueError("Baseline gradient accumulation disagrees with tokens/update")
    training["length_curriculum"] = curriculum
    training["fp32_master_weights"] = True
    training["max_steps"] = curriculum["max_steps"]
    budget["planned_training_flops"] = curriculum["planned_training_flops"]
    budget["planned_training_tokens"] = curriculum["planned_training_tokens"]
    budget.pop("flops_per_update", None)
    budget["flops_per_token_basis"] = "8192-token reference; use curriculum stage costs at runtime"
    settings["scaling"]["flops_accounting"] = curriculum["accounting"]["kind"]
    scheduler.update(_scheduler_fields(curriculum))
    logging = settings["logging"]
    maximum = curriculum["max_steps"]
    logging["eval_every_steps"] = maximum
    logging["checkpoint_every_steps"] = maximum
    logging["checkpoint_steps"] = sorted({round(maximum * part / 10) for part in range(1, 10)})
    if "eval_steps" in logging:
        logging["eval_steps"] = [maximum]
    return settings


def curriculum_stage(settings, step):
    curriculum = settings["training"].get("length_curriculum")
    if curriculum is None:
        return None
    step = _integer(step, "curriculum step")
    for stage in curriculum["stages"]:
        if stage["first_step"] <= step <= stage["last_step"]:
            return stage
    raise ValueError("Curriculum step exceeds the training plan")


def curriculum_completed_flops(settings, completed_steps):
    curriculum = settings["training"].get("length_curriculum")
    if curriculum is None:
        return None
    completed_steps = _integer(completed_steps, "completed_steps", 0)
    if completed_steps > curriculum["max_steps"]:
        raise ValueError("Completed steps exceed the curriculum plan")
    return sum(
        min(stage["steps"], max(0, completed_steps - stage["first_step"] + 1)) * stage["flops_per_update"]
        for stage in curriculum["stages"]
    )


def curriculum_lr_factor(settings, completed_steps):
    """LambdaLR factor for the next update, using its post-update FLOPs fraction."""
    curriculum = settings["training"].get("length_curriculum")
    if curriculum is None:
        return None
    completed_steps = _integer(completed_steps, "completed_steps", 0)
    if completed_steps > curriculum["max_steps"]:
        raise ValueError("Completed steps exceed the curriculum plan")
    next_completed = min(completed_steps + 1, curriculum["max_steps"])
    final_ratio = curriculum["reference"]["final_lr_ratio"]
    if next_completed == curriculum["max_steps"]:
        return float(final_ratio)
    fraction = curriculum_completed_flops(settings, next_completed) / curriculum["planned_training_flops"]
    fields = _scheduler_fields(curriculum)
    warmup, cooldown = fields["warmup_fraction"], fields["cooldown_fraction"]
    if warmup > 0 and fraction < warmup:
        return fraction / warmup
    if fraction > 1 - cooldown:
        progress = (fraction - (1 - cooldown)) / cooldown
        return max(final_ratio, 1 - (1 - final_ratio) * progress)
    return 1.0


def validate_curriculum(settings, world_size, micro_batch_size, grad_accum_steps, max_steps):
    curriculum = settings["training"].get("length_curriculum")
    if curriculum is None:
        return
    if not isinstance(curriculum, dict) or not isinstance(curriculum.get("reference"), dict):
        raise ValueError("Malformed length curriculum")
    expected = _build_curriculum(settings, curriculum["reference"])
    if curriculum != expected:
        raise ValueError("Curriculum stages, coverage or accounting differ from the derived plan")
    training, budget, scheduler = settings["training"], settings["budget"], settings["scheduler"]
    accumulation = expected["stages"][-1]["grad_accum_steps"]
    values = (world_size, micro_batch_size, grad_accum_steps, max_steps)
    for value, label in zip(values, ("world_size", "micro_batch_size", "grad_accum_steps", "max_steps")):
        _integer(value, label)
    if values != (expected["world_size"], 1, accumulation, expected["max_steps"]):
        raise ValueError("CLI topology, batch or steps differ from the curriculum plan")
    if (
        training["micro_batch_size"] != 1
        or training["gradient_accumulation_steps"] != accumulation
        or training["max_steps"] != expected["max_steps"]
        or training.get("fp32_master_weights") is not True
    ):
        raise ValueError("Curriculum training settings require the declared batches and FP32 masters")
    if "flops_per_update" in budget:
        raise ValueError("Variable-length curriculum cannot declare fixed flops_per_update")
    for field in ("planned_training_flops", "planned_training_tokens"):
        if budget[field] != expected[field]:
            raise ValueError(f"Curriculum budget.{field} disagrees with stages")
    for field in ("flops_per_token_numerator", "flops_per_token_denominator"):
        if budget[field] != curriculum["reference"][field]:
            raise ValueError("Curriculum full-length FLOPs reference differs")
    if scheduler["name"] != "linear_cooldown" or scheduler["final_lr_ratio"] != curriculum["reference"]["final_lr_ratio"]:
        raise ValueError("Curriculum learning-rate schedule differs from its baseline")
    if any(scheduler.get(key) != value for key, value in _scheduler_fields(expected).items()):
        raise ValueError("Curriculum learning-rate fractions or step counts disagree")
    for stage in curriculum["stages"]:
        if world_size * stage["length"] * stage["micro_batch_size"] * stage["grad_accum_steps"] != expected["tokens_per_update"]:
            raise ValueError("Curriculum stage changes global tokens/update")
    for field in ("checkpoint_steps", "eval_steps"):
        for step in settings["logging"].get(field, []):
            if not 1 <= _integer(step, field) <= expected["max_steps"]:
                raise ValueError("Logging checkpoint/evaluation step exceeds the curriculum")
    for field in ("eval_every_steps", "checkpoint_every_steps"):
        if settings["logging"][field] != expected["max_steps"]:
            raise ValueError("Curriculum final evaluation/checkpoint frequency differs")
