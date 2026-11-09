"""Generate and run the eight Dense/Qwen3-MoE overnight experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
from qwen3_moe.plan import HYBRID_VARIANTS, make_recipe, validate_family_settings
from training.audit_qwen3_moe_flops import audit
from packed_cache import load_packed_cache, validate_packed_identity
from training_data import normalize_data_metadata


def positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def write_json(path, record):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_plans(output, target_flops):
    plans = {}
    for name in HYBRID_VARIANTS:
        config, settings, budget = make_recipe("3e17", hybrid_variant=name, target_flops=target_flops)
        directory = output / name
        directory.mkdir()
        write_json(directory / "config.json", config)
        write_json(directory / "budget.json", budget)
        (directory / "training.yaml").write_text(
            yaml.safe_dump(settings, sort_keys=False), encoding="utf-8",
        )
        plans[name] = settings
    return plans


def preflight(args, settings, devices):
    validate_family_settings(settings, check_runtime=True)
    model = settings["model"]["hf_config"]
    tokenizer_hash = settings["data"]["tokenizer_json_sha256"]
    metadata = json.loads((args.train_cache / "metadata.json").read_text())
    metadata = normalize_data_metadata(metadata, model["max_position_embeddings"])
    if metadata.get("tokenizer_details", {}).get("tokenizer_json_sha256") != tokenizer_hash:
        raise ValueError("Training cache tokenizer does not match the configured tokenizer")
    if hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest() != tokenizer_hash:
        raise ValueError("Export tokenizer does not match the configured tokenizer")
    for name in ("train.npy", "train_is_text.npy", "validation.npy", "validation_is_text.npy"):
        if not (args.train_cache / name).is_file():
            raise FileNotFoundError(args.train_cache / name)
    tokens, _, cache = load_packed_cache(args.eval_cache)
    if cache["format"] != "npy":
        raise ValueError("The overnight run requires a packed NPY evaluation cache")
    validate_packed_identity(cache, dict(vocab_size=model["vocab_size"], eos_token_id=model["eos_token_id"],
                                        tokenizer_json_sha256=tokenizer_hash), model["vocab_size"], tokens=tokens)
    if shutil.which("timeout") is None:
        raise RuntimeError("GNU timeout is required for the training and evaluation watchdogs")
    # Exit the probe process before DDP starts so it holds no GPU contexts.
    subprocess.run(
        [sys.executable, "-c",
         "import torch; n = torch.cuda.device_count(); "
         "assert n == 8, f'Expected eight visible GPUs, got {n}'; "
         "assert all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(n)), "
         "'All eight GPUs must support BF16'"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices)), timeout=60, check=True,
    )


def run_command(command, log, environment, timeout_seconds):
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["timeout", "--kill-after=30s", f"{timeout_seconds}s", *map(str, command)],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"Exit {completed.returncode}. See {log}")


def train_variant(args, directory, settings, environment, train_seconds, setup_grace):
    env = dict(environment, TRAIN_CONFIG=str(directory / "training.yaml"), NPROC_PER_NODE="8")
    run_command(
        ["bash", ROOT / "qwen3_moe_a30m/train.sh", args.train_cache, directory / "train",
         "--time-limit-seconds", train_seconds],
        directory / "train.log", env, train_seconds + setup_grace,
    )
    metrics_path = directory / "train/metrics.jsonl"
    report = audit(settings, metrics_path, allow_time_limit=True)
    step = report["completed_optimizer_steps"]
    checkpoint = directory / "train" / f"step-{step}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    report.update(
        config=str(directory / "training.yaml"), metrics=str(metrics_path),
        config_sha256=hashlib.sha256((directory / "training.yaml").read_bytes()).hexdigest(),
        metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    )
    write_json(directory / "flops-report.json", report)
    return checkpoint, report


def evaluate_variant(args, directory, checkpoint, device, environment, batch_size, eval_timeout):
    model = directory / "hf-model"
    run_command(
        [sys.executable, ROOT / "training/export_hf.py", "--checkpoint", checkpoint,
         "--tokenizer-dir", args.tokenizer, "--output-dir", model, "--validation-device", "cpu"],
        directory / "export.log", dict(environment, CUDA_VISIBLE_DEVICES=""), eval_timeout,
    )
    path = directory / "eval-result.json"
    env = dict(environment, CUDA_VISIBLE_DEVICES=device, NPROC_PER_NODE="1", BATCH_SIZE=str(batch_size),
               SEQUENCE_LENGTH="8192", FUSION_MODE=os.environ.get("EVAL_FUSION_MODE", "rmsnorm"))
    run_command(["bash", ROOT / "evaluation/eval.sh", args.eval_cache, model, path],
                directory / "eval.log", env, eval_timeout)
    evaluation = json.loads(path.read_text())
    return {name: evaluation[name] for name in
            ("bpb", "forward_seconds", "evaltime", "active_parameters", "scored_tokens")}


def run_queue(args, output, plans, devices, train_seconds, setup_grace, batch_size, eval_timeout):
    environment = dict(
        os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES=",".join(devices), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKEN_CACHE_VALIDATION_DIR=str(output / "cache-validation"),
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("TRITON_F32_DEFAULT", "ieee")
    results = {
        name: dict(variant=name, eval_gpu=device, train_status="pending", eval_status="pending",
                   planned_flops=settings["budget"]["planned_training_flops"])
        for (name, settings), device in zip(plans.items(), devices)
    }
    summary = output / "summary.json"

    def save():
        write_json(summary, dict(train_seconds=train_seconds, variants=list(results.values())))

    save()
    ready = {}
    for name, settings in plans.items():
        print(f"[train] {name} on {environment['CUDA_VISIBLE_DEVICES']}", flush=True)
        directory = output / name
        result = results[name]
        result["train_status"] = "running"
        save()
        try:
            checkpoint, report = train_variant(args, directory, settings, environment, train_seconds, setup_grace)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            result.update(train_status="failed", eval_status="skipped", train_error=str(error))
            print(f"[train failed] {name} {error}", flush=True)
        else:
            complete = report["status"] == "passed_log_path_audit"
            result.update(train_status="complete" if complete else "time_limit",
                          checkpoint=str(checkpoint), completed_steps=report["completed_optimizer_steps"],
                          actual_training_flops=report["actual_training_flops"],
                          training_tokens=report["training_tokens"],
                          final_context_length=report["final_context_length"],
                          elapsed_seconds=report["elapsed_seconds"])
            ready[name] = checkpoint
        save()

    # Each variant retains its GPU slot even if an earlier training run failed.
    # Do not begin any export or GPU evaluation until all training slots finish.
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {}
        for name, checkpoint in ready.items():
            result = results[name]
            device = result["eval_gpu"]
            result["eval_status"] = "running"
            print(f"[eval] {name} on GPU {device}", flush=True)
            future = pool.submit(evaluate_variant, args, output / name, checkpoint, device,
                                 environment, batch_size, eval_timeout)
            futures[future] = name
        save()
        for future in as_completed(futures):
            name = futures[future]
            result = results[name]
            try:
                evaluation = future.result()
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                result.update(eval_status="failed", eval_error=str(error))
                print(f"[eval failed] {name} {error}", flush=True)
            else:
                result.update(eval_status="passed", **evaluation)
            save()
    passed = sum(result["eval_status"] == "passed" for result in results.values())
    print(f"[summary] {passed}/{len(plans)} evaluated. {summary}", flush=True)
    return 0 if passed == len(plans) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_cache", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("eval_cache", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--plan-only", action="store_true", help="Generate the eight configurations without CUDA")
    args = parser.parse_args(argv)
    args.train_cache = args.train_cache.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.eval_cache = args.eval_cache.resolve()
    try:
        train_seconds = positive_env("TRAIN_SECONDS", 3600)
        setup_grace = positive_env("TRAIN_SETUP_GRACE_SECONDS", 600)
        target_flops = positive_env("TRAIN_FLOPS", 3 * 10**17)
        batch_size = positive_env("EVAL_BATCH_SIZE", 64)
        eval_timeout = positive_env("EVAL_TIMEOUT_SECONDS", 1800)
        devices = [device.strip() for device in os.environ.get(
            "CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7",
        ).split(",")]
        if not args.plan_only and (len(devices) != 8 or len(set(devices)) != 8 or not all(devices)):
            raise ValueError("CUDA_VISIBLE_DEVICES must name exactly eight distinct GPUs")
        if os.environ.get("EVAL_FUSION_MODE", "rmsnorm") not in ("none", "rmsnorm"):
            raise ValueError("EVAL_FUSION_MODE must be none or rmsnorm for Qwen3 MoE")
        output = (args.output or ROOT / "runs" / (
            time.strftime("qwen3-hybrid-3e17-%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        )).resolve()
        output.mkdir(parents=True, exist_ok=False)
        plans = make_plans(output, target_flops)
        print(f"Output directory {output}", flush=True)
        if args.plan_only:
            return 0
        preflight(args, next(iter(plans.values())), devices)
        return run_queue(args, output, plans, devices, train_seconds, setup_grace, batch_size, eval_timeout)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
