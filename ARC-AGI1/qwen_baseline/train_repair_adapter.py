"""Train and evaluate an offline ARC error-mask repair LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--dev-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--solve-replay-fraction", type=float, default=0.15)
    parser.add_argument("--noop-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--diagnostic-examples", type=int, default=64)
    parser.add_argument("--rollout-examples", type=int, default=16)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completion_collator(tokenizer: Any):
    import torch

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), maximum), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[index, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


def select_diagnostics(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    return rng.sample(records, min(count, len(records)))


def summarize_lora_b(model: Any) -> dict[str, float | int]:
    """Cheap proof that optimization moved the zero-initialized LoRA-B weights."""
    import torch

    tensors = [
        parameter.detach()
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    ]
    if not tensors:
        raise RuntimeError("No LoRA-B parameters found")
    with torch.no_grad():
        return {
            "tensors": len(tensors),
            "elements": sum(tensor.numel() for tensor in tensors),
            "nonzero_elements": sum(int(torch.count_nonzero(tensor).item()) for tensor in tensors),
            "max_abs": max(float(tensor.float().abs().max().item()) for tensor in tensors),
            "l2": float(sum(tensor.float().square().sum().item() for tensor in tensors) ** 0.5),
        }


def distributed_metadata(
    *, world_size: int, rank: int, local_rank: int, per_device_batch: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_size": (
            world_size * per_device_batch * gradient_accumulation_steps
        ),
    }


def prepare_unsloth_offload(
    model: Any,
    temporary_location: Path,
    *,
    writable_root: Path = Path("/kaggle/working"),
) -> tuple[str, Path]:
    """Make Unsloth's embedding offload remain inside a writable directory.

    Unsloth 2026.7.5 constructs its destination with
    ``os.path.join(temporary_location, model.config._name_or_path)``.  Kaggle
    models have an absolute ``_name_or_path`` under ``/kaggle/input``; Python
    therefore discards ``temporary_location`` and Unsloth writes to the
    read-only input mount.  Temporarily replacing the config value with a safe
    relative component preserves Unsloth's implementation while keeping the
    actual files under ``/kaggle/working``.
    """
    root = writable_root.resolve()
    temporary_location = temporary_location.resolve()
    try:
        temporary_location.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"Unsloth offload directory must be under {root}: {temporary_location}"
        ) from error

    original_name_or_path = str(model.config._name_or_path)
    safe_component = "base_model"
    resolved_destination = (temporary_location / safe_component).resolve()
    try:
        resolved_destination.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"Resolved Unsloth offload escaped {root}: {resolved_destination}"
        ) from error
    resolved_destination.mkdir(parents=True, exist_ok=True)
    write_probe = resolved_destination / ".write_probe"
    write_probe.write_bytes(b"")
    write_probe.unlink()
    model.config._name_or_path = safe_component
    return original_name_or_path, resolved_destination


def evaluate_model(
    model: Any,
    tokenizer: Any,
    repair_records: list[dict[str, Any]],
    *,
    rollout_examples: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from repair_mining import (
        length_bucket_batches,
        parse_rollout_grid,
        restricted_greedy_rollout_batch,
        stabilize_inference_state,
        teacher_forced_metrics,
    )
    from repair_sft import build_solve_replay_example, reply_grid

    stabilize_inference_state(model)
    tasks = {
        "repair": [dict(record) for record in repair_records],
        "ordinary_solve": [build_solve_replay_example(record) for record in repair_records],
    }
    result: dict[str, Any] = {}
    for name, examples in tasks.items():
        metrics = [
            teacher_forced_metrics(
                model,
                tokenizer,
                example["input"],
                example["reply"],
            )
            for example in examples
        ]
        task_result: dict[str, Any] = {
            "examples": len(examples),
            "teacher_forced_restricted_exact": sum(
                metric["restricted_greedy_exact"] for metric in metrics
            ),
            "mean_gold_nll": float(np.mean([metric["gold_nll"] for metric in metrics])),
            "mean_gold_token_nll": float(
                np.mean([metric["gold_mean_nll"] for metric in metrics])
            ),
        }

        rollout_subset = examples[: min(rollout_examples, len(examples))]
        batches = length_bucket_batches(
            rollout_subset,
            batch_size=4,
            key=lambda example: len(tokenizer.encode(example["reply"])),
        )
        exact = 0
        valid = 0
        for batch_indices in batches:
            batch = [rollout_subset[index] for index in batch_indices]
            prompts = [example["input"] for example in batch]
            maximum_prompt = max(len(tokenizer.encode(prompt)) for prompt in prompts)
            outputs = restricted_greedy_rollout_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=min(930, 8192 - maximum_prompt),
            )
            for example, token_ids in zip(batch, outputs):
                prediction, _reason = parse_rollout_grid(tokenizer, token_ids)
                valid += prediction is not None
                exact += prediction == reply_grid(example["reply"])
            torch.cuda.empty_cache()
        task_result["rollout_examples"] = len(rollout_subset)
        task_result["rollout_valid"] = valid
        task_result["rollout_exact"] = exact
        result[name] = task_result
    return result


def merge_evaluation_results(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine disjoint per-rank diagnostics into one weighted result."""
    merged: dict[str, Any] = {}
    for task_name in parts[0]:
        task_parts = [part[task_name] for part in parts]
        examples = sum(part["examples"] for part in task_parts)
        if examples <= 0:
            raise RuntimeError(f"No distributed diagnostics for {task_name}")
        merged[task_name] = {
            "examples": examples,
            "teacher_forced_restricted_exact": sum(
                part["teacher_forced_restricted_exact"] for part in task_parts
            ),
            "mean_gold_nll": sum(
                part["mean_gold_nll"] * part["examples"] for part in task_parts
            ) / examples,
            "mean_gold_token_nll": sum(
                part["mean_gold_token_nll"] * part["examples"] for part in task_parts
            ) / examples,
            "rollout_examples": sum(part["rollout_examples"] for part in task_parts),
            "rollout_valid": sum(part["rollout_valid"] for part in task_parts),
            "rollout_exact": sum(part["rollout_exact"] for part in task_parts),
        }
    return merged


def evaluate_distributed(
    model: Any,
    tokenizer: Any,
    repair_records: list[dict[str, Any]],
    *,
    rollout_examples: int,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    import torch

    indices = list(range(rank, len(repair_records), world_size))
    local_records = [repair_records[index] for index in indices]
    local_rollouts = sum(index < rollout_examples for index in indices)
    local_result = evaluate_model(
        model,
        tokenizer,
        local_records,
        rollout_examples=local_rollouts,
    )
    if world_size == 1:
        return local_result
    gathered: list[dict[str, Any] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    torch.distributed.gather_object(local_result, gathered, dst=0)
    if rank != 0:
        return None
    if gathered is None or any(part is None for part in gathered):
        raise RuntimeError("Distributed diagnostics were not gathered from every rank")
    return merge_evaluation_results(gathered)  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()

    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    compile_root = Path(tempfile.gettempdir()) / f"unsloth_repair_sft_pid{os.getpid()}"
    compile_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(compile_root))
    ptxas_path = Path(os.environ["TRITON_PTXAS_PATH"])
    if not ptxas_path.is_file():
        raise FileNotFoundError(f"Missing Triton ptxas binary: {ptxas_path}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    offload_root = (
        Path("/kaggle/working")
        / "unsloth_repair_offload"
        / f"rank{local_rank}_pid{os.getpid()}"
    )
    offload_root.mkdir(parents=True, exist_ok=True)
    working_root = Path("/kaggle/working").resolve()
    resolved_output_dir = args.output_dir.resolve()
    try:
        resolved_output_dir.relative_to(working_root)
    except ValueError as error:
        raise RuntimeError(
            f"Repair output directory must be under {working_root}: "
            f"{resolved_output_dir}"
        ) from error
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected world_size={args.expected_world_size}, observed {world_size}"
        )

    # Modern Unsloth must patch Transformers before Torch/Transformers/PEFT are
    # imported by this process. torchrun has already populated the rank env vars,
    # which the modern loader uses for distributed device placement.
    import unsloth  # noqa: F401
    import torch

    unsloth_version = importlib.metadata.version("unsloth")
    unsloth_zoo_version = importlib.metadata.version("unsloth_zoo")
    if unsloth_version != "2026.7.5" or unsloth_zoo_version != "2026.7.6":
        raise RuntimeError(
            "Repair DDP requires the pinned modern utility layer: "
            f"unsloth={unsloth_version} unsloth_zoo={unsloth_zoo_version}"
        )
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

    from datasets import Dataset
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    from arc_solver import _make_unsloth_fixed_trainer_class
    from repair_mining import model_execution_device
    from repair_sft import (
        REPAIR_TOKEN,
        add_and_initialize_repair_token,
        build_training_mixture,
        tokenize_completion_only,
    )

    is_main = rank == 0
    output_exists = torch.tensor(
        [int(args.output_dir.exists())], device=f"cuda:{local_rank}", dtype=torch.int32
    )
    if world_size > 1:
        torch.distributed.broadcast(output_exists, src=0)
    if output_exists.item():
        raise FileExistsError(args.output_dir)
    if is_main:
        args.output_dir.mkdir(parents=True)
    if world_size > 1:
        torch.distributed.barrier()

    distributed = distributed_metadata(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        per_device_batch=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if args.expected_world_size == 4 and distributed["effective_global_batch_size"] != 4:
        raise RuntimeError(f"Expected effective global batch 4, observed {distributed}")
    print(
        f"[rank {rank}] local_rank={local_rank} world_size={world_size} "
        f"device={torch.cuda.get_device_name(local_rank)} global_batch="
        f"{distributed['effective_global_batch_size']}",
        flush=True,
    )

    train_records = load_jsonl(args.train_path)
    dev_records = load_jsonl(args.dev_path)
    if args.max_train_examples is not None:
        train_records = train_records[: args.max_train_examples]
    mixture, mixture_manifest = build_training_mixture(
        train_records,
        solve_replay_fraction=args.solve_replay_fraction,
        noop_fraction=args.noop_fraction,
        seed=args.seed,
    )
    expected_fractions = {
        "repair_failure": 0.84,
        "solve_replay": 0.15,
        "repair_noop": 0.01,
    }
    if mixture_manifest["requested_fractions"] != expected_fractions:
        raise RuntimeError(
            f"Repair training mixture changed unexpectedly: {mixture_manifest}"
        )

    load_started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        device_map={"": f"cuda:{local_rank}"},
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=args.max_seq_length,
    )
    model_device = next(model.parameters()).device
    if model_device.type != "cuda" or model_device.index != local_rank:
        raise RuntimeError(
            f"Rank {rank} loaded model on {model_device}, expected cuda:{local_rank}"
        )
    old_vocab_size = len(tokenizer)
    repair_token_id = add_and_initialize_repair_token(model, tokenizer)
    if repair_token_id != old_vocab_size or len(tokenizer) != old_vocab_size + 1:
        raise RuntimeError(
            f"Unexpected repair token ID: id={repair_token_id}, old_vocab={old_vocab_size}"
        )

    original_model_config = model.config
    original_model_name, resolved_offload_dir = prepare_unsloth_offload(
        model, offload_root
    )
    free_working_gib = shutil.disk_usage("/kaggle/working").free / 2**30
    print(
        f"[rank {rank}] unsloth_offload_dir={resolved_offload_dir} "
        f"working_free_gib={free_working_gib:.3f}",
        flush=True,
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            # Repair prompts can approach the 8k context limit.  Rank-256 LoRA plus
            # uncheckpointed activations exceeded a 22 GiB L4 on the first forward
            # pass, so use Unsloth's offloaded checkpointing implementation.
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=True,
            loftq_config=None,
            # Each DDP rank gets a writable and collision-free target.  The
            # model config is temporarily made relative above because Unsloth
            # 2026.7.5 otherwise discards this prefix for absolute model paths.
            temporary_location=str(offload_root),
        )
    finally:
        # get_peft_model may replace ``model`` with a PEFT wrapper, so restore
        # the exact base config object that prepare_unsloth_offload modified.
        original_model_config._name_or_path = original_model_name
    execution_device = model_execution_device(model)
    if execution_device.type != "cuda" or execution_device.index != local_rank:
        raise RuntimeError(
            f"Rank {rank} PEFT execution device is {execution_device}, "
            f"expected cuda:{local_rank}"
        )
    print(
        f"[rank {rank}] first_parameter_device={next(model.parameters()).device} "
        f"execution_device={execution_device}",
        flush=True,
    )
    for _name, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    adapter_before = summarize_lora_b(model)
    model_load_seconds = time.perf_counter() - load_started

    tokenized = []
    overlong = Counter()
    for example in mixture:
        try:
            tokenized.append(
                tokenize_completion_only(
                    example,
                    tokenizer,
                    max_seq_length=args.max_seq_length,
                )
            )
        except ValueError as error:
            if "exceeding max_seq_length" not in str(error):
                raise
            overlong[example["record_type"]] += 1
    if not tokenized:
        raise RuntimeError("Every training example exceeded the context limit")
    distributed["tokenized_dataset_examples"] = len(tokenized)
    distributed["sampler_padding_examples_per_epoch"] = (-len(tokenized)) % world_size

    diagnostics = select_diagnostics(dev_records, args.diagnostic_examples, args.seed)
    if len(diagnostics) < world_size:
        raise RuntimeError(
            f"Need at least one diagnostic per rank: diagnostics={len(diagnostics)} "
            f"world_size={world_size}"
        )
    model = FastLanguageModel.for_inference(model)
    before = evaluate_distributed(
        model,
        tokenizer,
        diagnostics,
        rollout_examples=args.rollout_examples,
        rank=rank,
        world_size=world_size,
    )

    model = FastLanguageModel.for_training(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=f"cuda:{local_rank}")
    FixedTrainer = _make_unsloth_fixed_trainer_class(UnslothTrainer)
    trainer = FixedTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=completion_collator(tokenizer),
        train_dataset=Dataset.from_list(tokenized),
        max_seq_length=args.max_seq_length,
        args=UnslothTrainingArguments(
            output_dir=str(args.output_dir / "trainer_output"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            warmup_ratio=0.05,
            max_grad_norm=1.0,
            learning_rate=args.learning_rate,
            optim="adamw_torch",
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            save_strategy="steps",
            save_steps=100,
            # Keep one resumable checkpoint.  Four rank-local embedding
            # offloads, optimizer state, checkpoints, and the final adapter all
            # share Kaggle's 20 GiB writable volume.
            save_total_limit=1,
            eval_strategy="no",
            logging_steps=10,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
        ),
    )
    if trainer.accelerator.num_processes != world_size:
        raise RuntimeError(
            "Trainer did not enter the requested distributed world: "
            f"accelerator={trainer.accelerator.num_processes} expected={world_size}"
        )
    print(
        f"[rank {rank}] accelerator_process_index={trainer.accelerator.process_index} "
        f"num_processes={trainer.accelerator.num_processes}",
        flush=True,
    )
    print(
        f"[rank {rank}] pre_train_cuda_gib "
        f"allocated={torch.cuda.memory_allocated(local_rank) / 2**30:.3f} "
        f"reserved={torch.cuda.memory_reserved(local_rank) / 2**30:.3f}",
        flush=True,
    )
    training_started = time.perf_counter()
    stats = trainer.train()
    training_seconds = time.perf_counter() - training_started
    peak_cuda_gib = torch.cuda.max_memory_allocated(local_rank) / 2**30
    print(f"[rank {rank}] peak_train_cuda_gib={peak_cuda_gib:.3f}", flush=True)
    model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    del trainer
    adapter_after = summarize_lora_b(model)
    if adapter_after["nonzero_elements"] <= adapter_before["nonzero_elements"]:
        raise RuntimeError(
            "Training completed without changing any zero-initialized LoRA-B weights: "
            f"before={adapter_before} after={adapter_after}"
        )

    # Persist the trained artifact before optional autoregressive diagnostics.
    # A diagnostic failure must not discard a completed multi-hour run.
    adapter_dir = args.output_dir / "adapter"
    if is_main:
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
    if world_size > 1:
        torch.distributed.barrier()

    model = FastLanguageModel.for_inference(model)
    after = evaluate_distributed(
        model,
        tokenizer,
        diagnostics,
        rollout_examples=args.rollout_examples,
        rank=rank,
        world_size=world_size,
    )

    manifest = {
        "config": vars(args) | {
            "train_path": str(args.train_path),
            "dev_path": str(args.dev_path),
            "output_dir": str(args.output_dir),
        },
        "tokenizer": {
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": len(tokenizer),
            "repair_token": REPAIR_TOKEN,
            "repair_token_id": repair_token_id,
            "initialization": "mean(user,assistant,<|im_start|>,<|im_end|>)",
        },
        "mixture": mixture_manifest,
        "overlong": dict(sorted(overlong.items())),
        "tokenized_examples": len(tokenized),
        "distributed": distributed,
        "environment": {
            "unsloth": unsloth_version,
            "unsloth_zoo": unsloth_zoo_version,
            "torch": torch.__version__,
        },
        "adapter_update": {"before": adapter_before, "after": adapter_after},
        "diagnostics": {"before": before, "after": after},
        "timings": {
            "model_load_s": model_load_seconds,
            "training_s": training_seconds,
        },
        "train_metrics": dict(stats.metrics),
    }
    if is_main:
        (args.output_dir / "repair_sft_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str)
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)
    if world_size > 1:
        torch.distributed.barrier()
    # These are runtime-only buffers. Remove their directory entries after all
    # diagnostics so Kaggle persists the trained artifact rather than several
    # GiB of duplicate embedding offloads.
    shutil.rmtree(offload_root, ignore_errors=True)
    if world_size > 1:
        torch.distributed.barrier()
    if is_main:
        # Retain one checkpoint during training for recovery, then remove it
        # once the final adapter and manifest have been safely written.
        shutil.rmtree(args.output_dir / "trainer_output", ignore_errors=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
