"""Train and evaluate an offline ARC error-mask repair LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
import random
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
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
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


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

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

    import unsloth  # noqa: F401 - must precede transformers/PEFT imports in the pinned stack
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

    from arc_solver import _make_unsloth_fixed_trainer_class
    from repair_sft import (
        REPAIR_TOKEN,
        add_and_initialize_repair_token,
        build_training_mixture,
        tokenize_completion_only,
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

    load_started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=args.max_seq_length,
    )
    old_vocab_size = len(tokenizer)
    repair_token_id = add_and_initialize_repair_token(model, tokenizer)
    if repair_token_id != old_vocab_size or len(tokenizer) != old_vocab_size + 1:
        raise RuntimeError(
            f"Unexpected repair token ID: id={repair_token_id}, old_vocab={old_vocab_size}"
        )

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
        use_gradient_checkpointing=False,
        random_state=args.seed,
        use_rslora=True,
        loftq_config=None,
    )
    for _name, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
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

    diagnostics = select_diagnostics(dev_records, args.diagnostic_examples, args.seed)
    model = FastLanguageModel.for_inference(model)
    before = evaluate_model(
        model,
        tokenizer,
        diagnostics,
        rollout_examples=args.rollout_examples,
    )

    model = FastLanguageModel.for_training(model)
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
            save_strategy="no",
            eval_strategy="no",
            logging_steps=10,
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
        ),
    )
    training_started = time.perf_counter()
    stats = trainer.train()
    training_seconds = time.perf_counter() - training_started
    model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    del trainer

    # Persist the trained artifact before optional autoregressive diagnostics.
    # A diagnostic failure must not discard a completed multi-hour run.
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model = FastLanguageModel.for_inference(model)
    after = evaluate_model(
        model,
        tokenizer,
        diagnostics,
        rollout_examples=args.rollout_examples,
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
        "diagnostics": {"before": before, "after": after},
        "timings": {
            "model_load_s": model_load_seconds,
            "training_s": training_seconds,
        },
        "train_metrics": dict(stats.metrics),
    }
    (args.output_dir / "repair_sft_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
