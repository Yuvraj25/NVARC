"""Mine materialized ARC repair failures from the clean raw NVARC pool.

Run one process per GPU.  Every process deterministically reconstructs the same
global path sample, takes its ``global_index % world_size == rank`` shard, and
writes only expensive wrong-grid repair records.  Ordinary solve replay and
zero-mask repair examples are intentionally reconstructed from the original
clean corpus by the later trainer instead of being duplicated here.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-probes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--repair-token", type=str, default="<REPAIR>")
    return parser.parse_args()


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    if args.num_probes < 1:
        raise ValueError("num_probes must be positive")
    if not (0 <= args.rank < args.world_size):
        raise ValueError("rank must be in [0, world_size)")
    if args.rollout_batch_size < 1:
        raise ValueError("rollout_batch_size must be positive")

    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    compile_root = Path(tempfile.gettempdir()) / f"unsloth_repair_rank{args.rank}_pid{os.getpid()}"
    compile_root.mkdir(parents=True, exist_ok=True)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(compile_root)

    import unsloth  # noqa: F401 - must precede transformers/Unsloth model imports
    import torch
    from unsloth import FastLanguageModel

    from repair_mining import (
        build_repair_training_record,
        deterministic_sample_paths,
        discover_subset_root,
        format_prompt,
        format_reply,
        length_bucket_batches,
        load_probe_from_path,
        parse_rollout_grid,
        restricted_greedy_rollout_batch,
        stabilize_inference_state,
        teacher_forced_metrics,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / f"repair_failures.rank{args.rank}.jsonl"
    invalid_path = args.output_dir / f"invalid_rollouts.rank{args.rank}.jsonl"
    summary_path = args.output_dir / f"summary.rank{args.rank}.json"
    for path in (records_path, invalid_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    validation_ids = set(json.loads(args.validation_path.read_text()))
    training_root = discover_subset_root(args.input_root, "nvarc_training")
    global_paths = deterministic_sample_paths(
        training_root,
        count=args.num_probes,
        seed=args.seed,
        excluded_anchor_ids=validation_ids,
    )
    if len(global_paths) != args.num_probes:
        raise RuntimeError(f"Requested {args.num_probes} paths but found {len(global_paths)}")

    indexed_paths = [
        (global_index, path)
        for global_index, path in enumerate(global_paths)
        if global_index % args.world_size == args.rank
    ]
    probes = [
        load_probe_from_path(path, subset="nvarc_training", seed=args.seed)
        for _, path in indexed_paths
    ]
    prompts = [format_prompt(probe) for probe in probes]
    replies = [format_reply(probe.gold_output) for probe in probes]

    load_started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=args.max_seq_length,
    )
    model = FastLanguageModel.for_inference(model)
    stabilize_inference_state(model)
    if len(tokenizer) != 16:
        raise RuntimeError(f"Expected the 16-token ARC tokenizer, found {len(tokenizer)}")
    model_load_seconds = time.perf_counter() - load_started

    metrics = [None] * len(probes)
    sequence_too_long = []
    screen_started = time.perf_counter()
    # Batch 8 was slower than batch 1 in the validated 16-probe benchmark.
    for index, (prompt, reply) in enumerate(zip(prompts, replies)):
        prompt_tokens = len(tokenizer.encode(prompt))
        gold_tokens = len(tokenizer.encode(reply))
        if prompt_tokens + gold_tokens > args.max_seq_length:
            sequence_too_long.append(index)
            continue
        metrics[index] = teacher_forced_metrics(model, tokenizer, prompt, reply)
    screen_seconds = time.perf_counter() - screen_started

    failure_indices = [
        index
        for index, item in enumerate(metrics)
        if item is not None and not item["restricted_greedy_exact"]
    ]
    failure_prompts = [prompts[index] for index in failure_indices]
    rollout_batches = length_bucket_batches(
        failure_prompts,
        batch_size=args.rollout_batch_size,
        key=lambda prompt: len(tokenizer.encode(prompt)),
    )

    counts = {
        "assigned_probes": len(probes),
        "sequence_too_long": len(sequence_too_long),
        "teacher_forced_exact": sum(
            item is not None and item["restricted_greedy_exact"] for item in metrics
        ),
        "teacher_forced_failures": len(failure_indices),
        "valid_rollouts": 0,
        "rollout_exact_noops": 0,
        "usable_repair_failures": 0,
        "invalid_rollouts": 0,
    }
    rollout_started = time.perf_counter()
    for batch_number, local_failure_indices in enumerate(rollout_batches, 1):
        batch_probe_indices = [failure_indices[index] for index in local_failure_indices]
        batch_prompts = [prompts[index] for index in batch_probe_indices]
        batch_global_indices = [indexed_paths[index][0] for index in batch_probe_indices]
        maximum_prompt = max(len(tokenizer.encode(prompt)) for prompt in batch_prompts)
        token_batches = restricted_greedy_rollout_batch(
            model,
            tokenizer,
            batch_prompts,
            max_new_tokens=min(930, args.max_seq_length - maximum_prompt),
        )
        for probe_index, token_ids in zip(batch_probe_indices, token_batches):
            global_index, source_path = indexed_paths[probe_index]
            probe = probes[probe_index]
            prediction, invalid_reason = parse_rollout_grid(tokenizer, token_ids)
            if prediction is None:
                counts["invalid_rollouts"] += 1
                append_jsonl(
                    invalid_path,
                    {
                        "global_index": global_index,
                        "puzzle_id": probe.puzzle_id,
                        "source_relpath": str(source_path.relative_to(training_root)),
                        "reason": invalid_reason,
                        "rollout_tokens": len(token_ids),
                        "teacher_forced": metrics[probe_index],
                    },
                )
                continue

            record = build_repair_training_record(
                probe,
                prediction,
                repair_token=args.repair_token,
            )
            counts["valid_rollouts"] += 1
            if record["record_type"] == "repair_noop":
                counts["rollout_exact_noops"] += 1
                continue
            record["source_path"] = str(source_path.relative_to(training_root))
            record.update(
                {
                    "global_index": global_index,
                    "source_relpath": str(source_path.relative_to(training_root)),
                    "rollout_tokens": len(token_ids),
                    "teacher_forced": metrics[probe_index],
                    "decoder": {
                        "name": "restricted_greedy",
                        "rollout_batch_size": args.rollout_batch_size,
                        "batch_number": batch_number,
                        "batch_member_global_indices": batch_global_indices,
                        "rank": args.rank,
                        "world_size": args.world_size,
                        "max_new_tokens": min(930, args.max_seq_length - maximum_prompt),
                    },
                }
            )
            append_jsonl(records_path, record)
            counts["usable_repair_failures"] += 1

        print(
            f"rank={args.rank} rollout_batch={batch_number}/{len(rollout_batches)} "
            f"usable={counts['usable_repair_failures']} invalid={counts['invalid_rollouts']}",
            flush=True,
        )
        torch.cuda.empty_cache()

    rollout_seconds = time.perf_counter() - rollout_started
    summary = {
        "config": {
            "num_probes": args.num_probes,
            "seed": args.seed,
            "rank": args.rank,
            "world_size": args.world_size,
            "rollout_batch_size": args.rollout_batch_size,
            "max_seq_length": args.max_seq_length,
            "repair_token": args.repair_token,
        },
        "counts": counts,
        "timings": {
            "model_load_s": model_load_seconds,
            "teacher_forced_screen_s": screen_seconds,
            "rollout_s": rollout_seconds,
        },
        "outputs": {
            "repair_failures": str(records_path),
            "invalid_rollouts": str(invalid_path) if invalid_path.exists() else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
