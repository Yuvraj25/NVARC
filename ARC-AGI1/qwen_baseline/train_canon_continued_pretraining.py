"""Time-bounded Canon-AC continued pretraining on released NVARC data.

This program intentionally has no repair, leave-one-out, validation-global,
candidate-generation, or reverse-NLL curriculum path.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-clock-hours", type=float, default=11.5)
    parser.add_argument("--canon-only-fraction", type=float, default=0.30)
    parser.add_argument("--checkpoint-hours", type=float, default=3.0)
    parser.add_argument("--virtual-records", type=int, default=3_255_481)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def padded_collator(tokenizer: Any):
    import torch

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), maximum), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def prepare_unsloth_offload(model: Any, offload_root: Path) -> tuple[str, Path]:
    """Force Unsloth's temporary embedding files into writable Kaggle storage."""
    original_name = str(model.config._name_or_path)
    offload_root.mkdir(parents=True, exist_ok=True)
    model.config._name_or_path = str(offload_root)
    return original_name, offload_root


def fixed_unsloth_trainer_class(unsloth_trainer):
    """Avoid Unsloth's in-place loss mutation without importing solver code."""

    class FixedUnslothTrainer(unsloth_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            del kwargs
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            if labels is not None:
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if hasattr(loss, "clone"):
                loss = loss.clone()
            return (loss, outputs) if return_outputs else loss

    return FixedUnslothTrainer


def atomic_model_snapshot(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one recoverable snapshot while retaining at most two copies briefly."""
    from arc_canon import save_canon_state

    started = time.perf_counter()
    temporary = output_dir / ".checkpoint_next"
    previous = output_dir / ".checkpoint_previous"
    current = output_dir / "checkpoint"
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(previous, ignore_errors=True)
    temporary.mkdir(parents=True)

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    unwrapped.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    save_canon_state(unwrapped, temporary / "canon_ac.pt")
    (temporary / "progress.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    if current.exists():
        current.rename(previous)
    temporary.rename(current)
    shutil.rmtree(previous, ignore_errors=True)
    elapsed = time.perf_counter() - started
    size_bytes = sum(path.stat().st_size for path in current.rglob("*") if path.is_file())
    return {"seconds": elapsed, "bytes": size_bytes, "path": str(current)}


def main() -> None:
    script_started_monotonic = time.monotonic()
    script_started_epoch = time.time()
    args = parse_args()
    if args.wall_clock_hours <= 0:
        raise ValueError("wall-clock hours must be positive")
    if not 0 < args.canon_only_fraction < 1:
        raise ValueError("canon-only fraction must lie between zero and one")
    if args.checkpoint_hours <= 0:
        raise ValueError("checkpoint hours must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != args.expected_world_size:
        raise RuntimeError(f"Expected world_size={args.expected_world_size}, got {world_size}")
    is_main = rank == 0

    working_root = Path("/kaggle/working").resolve()
    output_dir = args.output_dir.resolve()
    corpus_root = args.corpus_root.resolve()
    try:
        output_dir.relative_to(working_root)
    except ValueError as error:
        raise RuntimeError(f"Output must be below {working_root}: {output_dir}") from error
    if not corpus_root.is_dir():
        raise FileNotFoundError(corpus_root)

    rank_work = working_root / "canon_cpt_runtime" / f"rank{local_rank}"
    rank_work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ["HF_HOME"] = str(working_root / "hf_cache")
    os.environ["XDG_CACHE_HOME"] = str(working_root / "xdg_cache")
    os.environ["TMPDIR"] = str(rank_work / "tmp")
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(rank_work / "unsloth_compile")
    os.environ["TRITON_CACHE_DIR"] = str(rank_work / "triton_cache")
    for variable in ("TMPDIR", "UNSLOTH_COMPILE_LOCATION", "TRITON_CACHE_DIR"):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)

    import unsloth  # noqa: F401 - must patch before Transformers/PEFT
    import torch

    if importlib.metadata.version("unsloth") != "2026.7.5":
        raise RuntimeError("Expected Unsloth 2026.7.5")
    if importlib.metadata.version("unsloth_zoo") != "2026.7.6":
        raise RuntimeError("Expected Unsloth Zoo 2026.7.6")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    from transformers import TrainerCallback
    from torch.utils.data import SequentialSampler
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    from arc_canon import add_canon_ac_modules, canon_parameters, install_canon_ac_training_hooks
    from nvarc_continued_pretraining import (
        SOURCE_WEIGHTS,
        WeightedNVARCCorpus,
        build_wall_clock_plan,
        load_released_sources,
    )

    output_exists = torch.tensor(
        [int(output_dir.exists())], dtype=torch.int32, device=f"cuda:{local_rank}"
    )
    if world_size > 1:
        torch.distributed.broadcast(output_exists, src=0)
    if output_exists.item():
        raise FileExistsError(output_dir)
    if is_main:
        output_dir.mkdir(parents=True)
    if world_size > 1:
        torch.distributed.barrier()

    sources = load_released_sources(corpus_root)
    source_sizes = {source: len(dataset) for source, dataset in sources.items()}
    if is_main:
        print(f"source sizes = {source_sizes}", flush=True)
        print(f"source weights = {SOURCE_WEIGHTS}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        device_map={"": f"cuda:{local_rank}"},
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=args.max_seq_length,
    )
    if len(tokenizer) != 16:
        raise RuntimeError(f"Expected the released 16-token tokenizer, got {len(tokenizer)}")
    add_canon_ac_modules(model, kernel_size=4, zero_init=True)
    canon_hooks = install_canon_ac_training_hooks(model)

    offload_root = rank_work / "unsloth_offload"
    original_config = model.config
    original_name, _ = prepare_unsloth_offload(model, offload_root)
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=None,
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
    for parameter in canon_parameters(model):
        parameter.requires_grad = True
    joint_trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not any("lora_" in name for name in joint_trainable_names):
        raise RuntimeError("Fresh global LoRA was not installed")
    for _name, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)

    corpus_stage1 = WeightedNVARCCorpus(
        sources,
        tokenizer,
        max_seq_length=args.max_seq_length,
        virtual_length=args.virtual_records,
        seed=args.seed,
    )
    corpus_stage2 = WeightedNVARCCorpus(
        sources,
        tokenizer,
        max_seq_length=args.max_seq_length,
        virtual_length=args.virtual_records,
        seed=args.seed + 1,
    )
    sample = corpus_stage1[0]
    if sample["assistant_outputs"] < 1 or sample["supervised_tokens"] < 1:
        raise RuntimeError("Corpus plumbing produced no assistant-token supervision")
    if is_main:
        print(
            "first record =",
            {key: sample[key] for key in (
                "source", "puzzle_name", "sequence_tokens", "assistant_outputs",
                "supervised_tokens", "dropped_leading_pairs",
            )},
            flush=True,
        )

    global_batch = world_size * args.gradient_accumulation_steps
    clock_plan = build_wall_clock_plan(
        started=script_started_monotonic,
        budget_seconds=args.wall_clock_hours * 3600,
        canon_only_fraction=args.canon_only_fraction,
        checkpoint_seconds=args.checkpoint_hours * 3600,
    )
    absolute_deadline = clock_plan.deadline
    canon_stage_deadline = clock_plan.stage_boundary
    checkpoint_deadlines = clock_plan.periodic_checkpoints

    save_history: list[dict[str, Any]] = []
    completed_steps_before_stage = 0

    class ClockCallback(TrainerCallback):
        def __init__(self, *, stage_name: str, stop_at: float, final_stage: bool) -> None:
            self.stage_name = stage_name
            self.stop_at = stop_at
            self.final_stage = final_stage
            self.next_checkpoint = bisect.bisect_right(
                checkpoint_deadlines, time.monotonic()
            )
            self.saved_final = False

        def on_step_end(self, args_, state, control, model=None, **kwargs):
            del args_, kwargs
            now = time.monotonic()
            decision = torch.zeros(2, dtype=torch.int32, device=f"cuda:{local_rank}")
            if is_main:
                due = (
                    self.next_checkpoint < len(checkpoint_deadlines)
                    and now >= checkpoint_deadlines[self.next_checkpoint]
                )
                stopping = now >= self.stop_at
                decision[0] = int(due)
                decision[1] = int(stopping)
            if world_size > 1:
                torch.distributed.broadcast(decision, src=0)
            due, stopping = (bool(value) for value in decision.tolist())
            if due or (stopping and self.final_stage):
                if world_size > 1:
                    torch.distributed.barrier()
                if is_main:
                    total_steps = completed_steps_before_stage + int(state.global_step)
                    reason = "deadline" if stopping and self.final_stage else "periodic"
                    metadata = {
                        "reason": reason,
                        "stage": self.stage_name,
                        "stage_optimizer_steps": int(state.global_step),
                        "total_optimizer_steps": total_steps,
                        "estimated_records_seen": total_steps * global_batch,
                        "saved_epoch_time": time.time(),
                        "elapsed_hours": (time.monotonic() - script_started_monotonic) / 3600,
                        "source_weights": SOURCE_WEIGHTS,
                    }
                    saved = atomic_model_snapshot(
                        model, tokenizer, output_dir, metadata=metadata
                    )
                    save_history.append({**metadata, **saved})
                    print(f"checkpoint = {save_history[-1]}", flush=True)
                if world_size > 1:
                    torch.distributed.barrier()
                if due:
                    self.next_checkpoint += 1
                if stopping and self.final_stage:
                    self.saved_final = True
            if stopping:
                control.should_training_stop = True
            return control

    BaseTrainer = fixed_unsloth_trainer_class(UnslothTrainer)

    class SequentialCorpusTrainer(BaseTrainer):
        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return SequentialSampler(dataset)

    def train_stage(
        stage_name: str,
        dataset: Any,
        stop_at: float,
        *,
        final_stage: bool,
        seed_offset: int,
    ):
        callback = ClockCallback(stage_name=stage_name, stop_at=stop_at, final_stage=final_stage)
        trainer = SequentialCorpusTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=padded_collator(tokenizer),
            train_dataset=dataset,
            max_seq_length=args.max_seq_length,
            callbacks=[callback],
            args=UnslothTrainingArguments(
                output_dir=str(output_dir / f"trainer_{stage_name}"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_steps=1_000_000_000,
                warmup_steps=0,
                learning_rate=args.learning_rate,
                optim="adamw_torch",
                weight_decay=0.1,
                lr_scheduler_type="constant",
                seed=args.seed + seed_offset,
                data_seed=args.seed + seed_offset,
                report_to="none",
                save_strategy="no",
                eval_strategy="no",
                logging_steps=25,
                fp16=False,
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                ddp_find_unused_parameters=False,
                dataloader_num_workers=0,
                remove_unused_columns=True,
            ),
        )
        stats = trainer.train()
        steps = int(trainer.state.global_step)
        del trainer
        return stats, steps, callback

    # Stage 1: Canon only.  The installed LoRA remains exactly at initialization.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".canonA." in name or ".canonC." in name
    stage1_stats, stage1_steps, _ = train_stage(
        "canon_only",
        corpus_stage1,
        canon_stage_deadline,
        final_stage=False,
        seed_offset=0,
    )
    completed_steps_before_stage += stage1_steps

    # Stage 2: train Canon and the fresh rank-256 global LoRA jointly.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in joint_trainable_names
    stage2_stats, stage2_steps, final_callback = train_stage(
        "canon_plus_lora",
        corpus_stage2,
        absolute_deadline,
        final_stage=True,
        seed_offset=1,
    )
    completed_steps_before_stage += stage2_steps

    if is_main and not final_callback.saved_final:
        metadata = {
            "reason": "trainer_returned_before_deadline",
            "stage": "canon_plus_lora",
            "total_optimizer_steps": completed_steps_before_stage,
            "estimated_records_seen": completed_steps_before_stage * global_batch,
            "saved_epoch_time": time.time(),
            "elapsed_hours": (time.monotonic() - script_started_monotonic) / 3600,
            "source_weights": SOURCE_WEIGHTS,
        }
        saved = atomic_model_snapshot(model, tokenizer, output_dir, metadata=metadata)
        save_history.append({**metadata, **saved})
    if world_size > 1:
        torch.distributed.barrier()

    if is_main:
        manifest = {
            "config": {**vars(args), "corpus_root": str(corpus_root), "output_dir": str(output_dir)},
            "script_started_epoch": script_started_epoch,
            "elapsed_hours": (time.monotonic() - script_started_monotonic) / 3600,
            "source_sizes": source_sizes,
            "source_weights": SOURCE_WEIGHTS,
            "loss_scope": "every_assistant_output",
            "reverse_nll_scoring": False,
            "repair_data": False,
            "stages": [
                {"name": "canon_only", "steps": stage1_steps, "metrics": dict(stage1_stats.metrics)},
                {"name": "canon_plus_lora", "steps": stage2_steps, "metrics": dict(stage2_stats.metrics)},
            ],
            "total_optimizer_steps": completed_steps_before_stage,
            "estimated_records_seen": completed_steps_before_stage * global_batch,
            "checkpoint_history": save_history,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)

    canon_hooks.remove()
    shutil.rmtree(rank_work, ignore_errors=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
