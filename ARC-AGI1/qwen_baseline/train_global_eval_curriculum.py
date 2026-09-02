"""Train a global LoRA on reverse-NLL-ordered ARC evaluation training pairs.

The input challenge file supplies only the public per-task demonstrations.  No
evaluation solution file is accepted or opened.  An optional frozen starting
adapter may be merged before pre-training target NLL determines a hard-to-easy
curriculum for a fresh global LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repair-adapter-path", type=Path, default=None)
    parser.add_argument("--challenges-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views-per-task", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--canon-ac",
        action="store_true",
        help="Add zero-initialized residual Canon modules at A and C.",
    )
    parser.add_argument("--canon-kernel-size", type=int, default=4)
    parser.add_argument(
        "--canon-only-warmup-fraction",
        type=float,
        default=0.30,
        help="Fraction of the one-epoch curriculum trained with only Canon unfrozen.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nll_summary(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def parameter_l2(parameters) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for parameter in parameters:
            total += parameter.detach().double().cpu().square().sum()
    return float(total.sqrt().item())


def score_tokenized_examples(
    model: Any,
    tokenizer: Any,
    tokenized: list[dict[str, list[int]]],
    *,
    rank: int,
    world_size: int,
    batch_size: int,
) -> list[float] | None:
    """Return target-token mean NLLs on rank zero, preserving record indices."""
    import torch
    import torch.nn.functional as functional

    from repair_mining import model_execution_device

    if batch_size <= 0:
        raise ValueError("score batch size must be positive")
    device = model_execution_device(model)
    local_indices = list(range(rank, len(tokenized), world_size))
    # Length buckets reduce padding without changing the final index mapping.
    local_indices.sort(key=lambda index: len(tokenized[index]["input_ids"]))
    local_scores: dict[int, float] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(local_indices), batch_size):
            indices = local_indices[start : start + batch_size]
            maximum = max(len(tokenized[index]["input_ids"]) for index in indices)
            input_ids = torch.full(
                (len(indices), maximum), tokenizer.pad_token_id,
                dtype=torch.long, device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            for row, index in enumerate(indices):
                example = tokenized[index]
                length = len(example["input_ids"])
                input_ids[row, :length] = torch.tensor(example["input_ids"], device=device)
                attention_mask[row, :length] = 1
                labels[row, :length] = torch.tensor(example["labels"], device=device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            losses = functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(len(indices), -1)
            valid = shift_labels.ne(-100)
            totals = (losses * valid).sum(dim=1)
            counts = valid.sum(dim=1)
            if bool((counts == 0).any()):
                raise RuntimeError("A scored record has no target labels")
            for index, total, count in zip(indices, totals, counts):
                local_scores[index] = float((total / count).item())
            del logits, shift_logits, losses, input_ids, attention_mask, labels

    if world_size == 1:
        return [local_scores[index] for index in range(len(tokenized))]
    gathered: list[dict[int, float] | None] | None = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(local_scores, gathered, dst=0)
    if rank != 0:
        return None
    combined: dict[int, float] = {}
    assert gathered is not None
    for part in gathered:
        if part is None:
            raise RuntimeError("Missing a rank's curriculum scores")
        overlap = set(combined).intersection(part)
        if overlap:
            raise RuntimeError(f"Duplicate distributed score indices: {sorted(overlap)[:5]}")
        combined.update(part)
    if set(combined) != set(range(len(tokenized))):
        raise RuntimeError("Distributed scoring did not cover the complete curriculum")
    return [combined[index] for index in range(len(tokenized))]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    compile_root = Path(tempfile.gettempdir()) / f"unsloth_global_eval_pid{os.getpid()}"
    compile_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(compile_root))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected world_size={args.expected_world_size}, observed {world_size}"
        )
    working_root = Path("/kaggle/working").resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(working_root)
    except ValueError as error:
        raise RuntimeError(f"Output must be under {working_root}: {output_dir}") from error

    import unsloth  # noqa: F401  # must patch before Transformers/PEFT imports
    import torch

    if importlib.metadata.version("unsloth") != "2026.7.5":
        raise RuntimeError("Expected Unsloth 2026.7.5 utility")
    if importlib.metadata.version("unsloth_zoo") != "2026.7.6":
        raise RuntimeError("Expected Unsloth Zoo 2026.7.6 utility")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

    from datasets import Dataset
    from peft import PeftModel
    from torch.utils.data import SequentialSampler
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    from arc_solver import _make_unsloth_fixed_trainer_class
    from arc_canon import (
        add_canon_ac_modules,
        canon_parameters,
        install_canon_ac_training_hooks,
        save_canon_state,
    )
    from global_eval_curriculum import (
        build_exact_curriculum_records,
        format_completion_record,
        load_evaluation_training_tasks,
        summarize_records,
    )
    from repair_sft import (
        REPAIR_TOKEN,
        add_and_initialize_repair_token,
        tokenize_completion_only,
    )
    from train_repair_adapter import (
        completion_collator,
        prepare_unsloth_offload,
        summarize_lora_b,
    )

    # The Kaggle working volume is shared by every DDP rank.  Only rank zero
    # may decide whether this is a stale pre-existing output; otherwise rank
    # zero's mkdir races with the same check on ranks 1..N.
    output_exists = torch.tensor(
        [int(args.output_dir.exists())],
        device=f"cuda:{local_rank}",
        dtype=torch.int32,
    )
    if world_size > 1:
        torch.distributed.broadcast(output_exists, src=0)
    if output_exists.item():
        raise FileExistsError(args.output_dir)
    if is_main:
        args.output_dir.mkdir(parents=True)
    if world_size > 1:
        torch.distributed.barrier()

    raw_challenges = json.loads(args.challenges_path.read_text())
    if not isinstance(raw_challenges, dict):
        raise ValueError("ARC challenges must be a task-id mapping")
    observed_full_task_count = len(raw_challenges)
    tasks = load_evaluation_training_tasks(args.challenges_path)
    skipped_task_ids = sorted(set(raw_challenges) - set(tasks))
    if not tasks:
        raise RuntimeError("No tasks have at least two training pairs for global SFT")
    if is_main:
        print(
            f"global curriculum eligible_tasks={len(tasks)} "
            f"skipped_lt2_train_pairs={len(skipped_task_ids)}",
            flush=True,
        )
    if args.max_tasks is not None:
        tasks = dict(list(tasks.items())[: args.max_tasks])
    raw_records = build_exact_curriculum_records(
        tasks, views_per_task=args.views_per_task, seed=args.seed
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
    old_vocab_size = len(tokenizer)
    repair_token_id = None
    repair_adapter = None
    if args.repair_adapter_path is not None:
        repair_token_id = add_and_initialize_repair_token(model, tokenizer)
        if repair_token_id != old_vocab_size:
            raise RuntimeError(f"Unexpected {REPAIR_TOKEN} ID {repair_token_id}")

        repair_adapter = args.repair_adapter_path.resolve()
        for required in ("adapter_config.json", "adapter_model.safetensors"):
            if not (repair_adapter / required).is_file():
                raise FileNotFoundError(repair_adapter / required)
        model = PeftModel.from_pretrained(
            model, str(repair_adapter), is_trainable=False, local_files_only=True
        )
        model = model.merge_and_unload(safe_merge=True)
        remaining_repair_lora = [
            name for name, _parameter in model.named_parameters() if "lora_" in name
        ]
        if remaining_repair_lora:
            raise RuntimeError(
                f"Starting LoRA tensors remained after merge: {remaining_repair_lora[:5]}"
            )
    model_load_seconds = time.perf_counter() - load_started
    print(
        f"[rank {rank}] starting_adapter={repair_adapter}; vocab={len(tokenizer)} "
        f"device={next(model.parameters()).device}",
        flush=True,
    )

    canon_hooks = None
    if args.canon_ac:
        if args.epochs != 1.0:
            raise ValueError("The first staged Canon experiment requires --epochs 1.0")
        if not 0.0 < args.canon_only_warmup_fraction < 1.0:
            raise ValueError("canon-only warmup fraction must lie strictly between zero and one")
        add_canon_ac_modules(
            model,
            kernel_size=args.canon_kernel_size,
            zero_init=True,
        )
        canon_hooks = install_canon_ac_training_hooks(model)
        print(
            f"[rank {rank}] installed residual Canon-AC "
            f"kernel={args.canon_kernel_size} activation=none zero_init=true",
            flush=True,
        )

    formatted = [
        format_completion_record(record, tokenizer, max_seq_length=args.max_seq_length)
        for record in raw_records
    ]
    tokenized = [
        tokenize_completion_only(record, tokenizer, max_seq_length=args.max_seq_length)
        for record in formatted
    ]
    if len(tokenized) != len(tasks) * args.views_per_task:
        raise RuntimeError("Formatting changed the exact curriculum record count")

    scoring_started = time.perf_counter()
    model = FastLanguageModel.for_inference(model)
    scores = score_tokenized_examples(
        model,
        tokenizer,
        tokenized,
        rank=rank,
        world_size=world_size,
        batch_size=args.score_batch_size,
    )
    scoring_seconds = time.perf_counter() - scoring_started
    if world_size > 1:
        torch.distributed.barrier()

    order_payload: list[int] | None = None
    if is_main:
        assert scores is not None
        order_payload = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    objects = [order_payload]
    if world_size > 1:
        torch.distributed.broadcast_object_list(objects, src=0)
    order = objects[0]
    if order is None or len(order) != len(tokenized):
        raise RuntimeError("Failed to broadcast the complete curriculum order")
    ordered_tokenized = [tokenized[index] for index in order]

    curriculum_rows: list[dict[str, Any]] | None = None
    if is_main:
        assert scores is not None
        curriculum_rows = [
            {
                "curriculum_rank": rank_index,
                "mean_target_nll": scores[source_index],
                "task_id": formatted[source_index]["task_id"],
                "view_index": formatted[source_index]["view_index"],
                "target_index": formatted[source_index]["target_index"],
                "demonstration_indices": formatted[source_index]["demonstration_indices"],
                "kept_demonstration_indices": formatted[source_index]["kept_demonstration_indices"],
                "dropped_demonstration_indices": formatted[source_index]["dropped_demonstration_indices"],
                "transform_id": formatted[source_index]["transform_id"],
                "color_mapping": formatted[source_index]["color_mapping"],
                "sequence_tokens": formatted[source_index]["sequence_tokens"],
            }
            for rank_index, source_index in enumerate(order)
        ]
        with (args.output_dir / "curriculum.jsonl").open("w") as handle:
            for row in curriculum_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    model = FastLanguageModel.for_training(model)
    offload_root = (
        Path("/kaggle/working")
        / "unsloth_global_eval_offload"
        / f"rank{local_rank}_pid{os.getpid()}"
    )
    offload_root.mkdir(parents=True, exist_ok=True)
    original_config = model.config
    original_name, resolved_offload = prepare_unsloth_offload(model, offload_root)
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            # Match Vanilla V2: update the compact ARC embedding and output
            # head alongside the seven projection-family LoRAs.
            modules_to_save=["embed_tokens", "lm_head"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=True,
            loftq_config=None,
            temporary_location=str(offload_root),
        )
    finally:
        original_config._name_or_path = original_name
    print(f"[rank {rank}] global LoRA offload={resolved_offload}", flush=True)
    adapter_before = summarize_lora_b(model)
    joint_trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if args.canon_ac:
        for parameter in canon_parameters(model):
            parameter.requires_grad = True
        joint_trainable_names.update(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
    canon_l2_before = parameter_l2(canon_parameters(model)) if args.canon_ac else None
    for _name, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)

    BaseTrainer = _make_unsloth_fixed_trainer_class(UnslothTrainer)

    class SequentialCurriculumTrainer(BaseTrainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return SequentialSampler(dataset)

    def train_stage(stage_name: str, rows: list[dict[str, Any]]):
        trainer = SequentialCurriculumTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=completion_collator(tokenizer),
            train_dataset=Dataset.from_list(rows),
            max_seq_length=args.max_seq_length,
            args=UnslothTrainingArguments(
                output_dir=str(args.output_dir / f"trainer_output_{stage_name}"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=1.0 if args.canon_ac else args.epochs,
                warmup_ratio=0.05,
                max_grad_norm=1.0,
                learning_rate=args.learning_rate,
                optim="adamw_torch",
                weight_decay=0.0,
                lr_scheduler_type="cosine",
                seed=args.seed,
                data_seed=args.seed,
                report_to="none",
                save_strategy="no",
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
                f"Trainer world size {trainer.accelerator.num_processes} != {world_size}"
            )
        stage_started = time.perf_counter()
        stage_stats = trainer.train()
        elapsed = time.perf_counter() - stage_started
        unwrapped = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        del trainer
        return unwrapped, stage_stats, elapsed

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=f"cuda:{local_rank}")
    training_started = time.perf_counter()
    stage_metrics = []
    if args.canon_ac:
        split = max(
            1,
            min(
                len(ordered_tokenized) - 1,
                round(len(ordered_tokenized) * args.canon_only_warmup_fraction),
            ),
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".canonA." in name or ".canonC." in name
        model, warmup_stats, warmup_seconds = train_stage(
            "canon_only",
            ordered_tokenized[:split],
        )
        stage_metrics.append(
            {
                "name": "canon_only",
                "records": split,
                "seconds": warmup_seconds,
                "metrics": dict(warmup_stats.metrics),
            }
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in joint_trainable_names
        model, joint_stats, joint_seconds = train_stage(
            "canon_plus_lora",
            ordered_tokenized[split:],
        )
        stage_metrics.append(
            {
                "name": "canon_plus_lora",
                "records": len(ordered_tokenized) - split,
                "seconds": joint_seconds,
                "metrics": dict(joint_stats.metrics),
            }
        )
        stats = joint_stats
    else:
        model, stats, vanilla_seconds = train_stage("global_lora", ordered_tokenized)
        stage_metrics.append(
            {
                "name": "global_lora",
                "records": len(ordered_tokenized),
                "seconds": vanilla_seconds,
                "metrics": dict(stats.metrics),
            }
        )
    training_seconds = time.perf_counter() - training_started
    peak_cuda_gib = torch.cuda.max_memory_allocated(local_rank) / 2**30
    adapter_after = summarize_lora_b(model)
    if not math.isfinite(float(adapter_after["l2"])) or abs(
        float(adapter_after["l2"]) - float(adapter_before["l2"])
    ) <= 1e-8:
        raise RuntimeError(
            f"Global LoRA did not update: before={adapter_before} after={adapter_after}"
        )
    if is_main:
        adapter_dir = args.output_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        if args.canon_ac:
            save_canon_state(model, adapter_dir / "canon_ac.pt")
    if world_size > 1:
        torch.distributed.barrier()

    if is_main:
        assert scores is not None and curriculum_rows is not None
        dropped_counts = [len(record["dropped_demonstration_indices"]) for record in formatted]
        manifest = {
            "config": {
                **vars(args),
                "repair_adapter_path": (
                    str(args.repair_adapter_path)
                    if args.repair_adapter_path is not None else None
                ),
                "challenges_path": str(args.challenges_path),
                "output_dir": str(args.output_dir),
            },
            "data": {
                "challenge_sha256": file_sha256(args.challenges_path),
                "full_task_count": observed_full_task_count,
                "used_task_count": len(tasks),
                "skipped_lt2_train_pair_task_count": len(skipped_task_ids),
                "skipped_lt2_train_pair_task_ids": skipped_task_ids,
                "summary": summarize_records(raw_records),
                "formatted_records": len(formatted),
                "records_dropping_demonstrations": sum(count > 0 for count in dropped_counts),
                "total_dropped_demonstrations": sum(dropped_counts),
                "loss_scope": "final_target_reply_only",
                "test_pairs_loaded": False,
            },
            "curriculum": {
                "metric": "pre_update_mean_target_token_nll",
                "direction": "descending_hard_to_easy",
                "nll": nll_summary(scores),
                "first": curriculum_rows[:10],
                "last": curriculum_rows[-10:],
            },
            "model": {
                "base_vocab_size": old_vocab_size,
                "merged_vocab_size": len(tokenizer),
                "repair_token": REPAIR_TOKEN if repair_token_id is not None else None,
                "repair_token_id": repair_token_id,
                "repair_adapter_merged_before_scoring": repair_adapter is not None,
                "fresh_global_lora_rank": args.lora_rank,
                "fresh_global_modules_to_save": ["embed_tokens", "lm_head"],
                "canon_ac": args.canon_ac,
                "canon_kernel_size": args.canon_kernel_size if args.canon_ac else None,
                "canon_activation": None,
                "canon_residual": args.canon_ac,
                "canon_l2_before": canon_l2_before,
                "canon_l2_after": (
                    parameter_l2(canon_parameters(model)) if args.canon_ac else None
                ),
            },
            "environment": {
                "torch": torch.__version__,
                "unsloth": importlib.metadata.version("unsloth"),
                "unsloth_zoo": importlib.metadata.version("unsloth_zoo"),
                "world_size": world_size,
            },
            "adapter_update": {"before": adapter_before, "after": adapter_after},
            "timings": {
                "model_load_s": model_load_seconds,
                "curriculum_scoring_s": scoring_seconds,
                "training_s": training_seconds,
            },
            "peak_train_cuda_gib": peak_cuda_gib,
            "train_metrics": dict(stats.metrics),
            "train_stages": stage_metrics,
        }
        (args.output_dir / "global_eval_curriculum_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str)
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)

    shutil.rmtree(offload_root, ignore_errors=True)
    if is_main:
        for path in args.output_dir.glob("trainer_output_*"):
            shutil.rmtree(path, ignore_errors=True)
    if canon_hooks is not None:
        canon_hooks.remove()
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
