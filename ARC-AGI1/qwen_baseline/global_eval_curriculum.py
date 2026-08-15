"""Build an exact-count ARC evaluation-train curriculum without test labels.

Each record treats one provided training pair as the final query.  Every other
provided training pair is a completed demonstration in the causal prefix.  The
record is intended for completion-only SFT: only the final query's output is a
label, although all demonstration outputs remain visible to the model.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from arc_loader import QwenFormatter
from repair_mining import transform_grid, validate_grid


DEFAULT_SEED = 20260815


def _stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def load_evaluation_training_tasks(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load only the public training pairs from an ARC challenges file."""
    raw = json.loads(path.read_text())
    tasks: dict[str, list[dict[str, Any]]] = {}
    for task_id, task in sorted(raw.items()):
        pairs = task.get("train")
        if not isinstance(pairs, list) or len(pairs) < 2:
            raise ValueError(f"{task_id}: expected at least two training pairs")
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or not validate_grid(pair.get("input")):
                raise ValueError(f"{task_id} train[{pair_index}]: invalid input grid")
            if not validate_grid(pair.get("output")):
                raise ValueError(f"{task_id} train[{pair_index}]: invalid output grid")
        # Deliberately do not retain ``test``.  This module never needs it.
        tasks[task_id] = pairs
    return tasks


def _color_mapping(task_id: str, view_index: int, seed: int) -> tuple[int, ...]:
    rng = random.Random(_stable_seed(seed, task_id, view_index, "colors"))
    values = list(range(10))
    rng.shuffle(values)
    return tuple(values)


def _demo_order(
    task_id: str,
    view_index: int,
    target_index: int,
    pair_count: int,
    seed: int,
) -> list[int]:
    indices = [index for index in range(pair_count) if index != target_index]
    rng = random.Random(_stable_seed(seed, task_id, view_index, "demo-order"))
    rng.shuffle(indices)
    return indices


def build_augmented_record(
    *,
    task_id: str,
    pairs: Sequence[dict[str, Any]],
    view_index: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build one deterministic leave-one-pair-as-query augmented record."""
    if len(pairs) < 2:
        raise ValueError("At least two training pairs are required")
    # A task-specific offset avoids always using pair zero for view zero while
    # retaining balanced target counts (difference at most one).
    target_offset = _stable_seed(seed, task_id, "target-offset") % len(pairs)
    target_index = int((target_offset + view_index) % len(pairs))
    transform_id = int(view_index % 8)
    color_mapping = _color_mapping(task_id, view_index, seed)
    demonstration_indices = _demo_order(
        task_id, view_index, target_index, len(pairs), seed
    )

    def transformed(index: int) -> dict[str, Any]:
        pair = pairs[index]
        return {
            "input": transform_grid(pair["input"], transform_id, color_mapping),
            "output": transform_grid(pair["output"], transform_id, color_mapping),
        }

    demonstrations = [transformed(index) for index in demonstration_indices]
    target = transformed(target_index)
    return {
        "record_type": "global_eval_train_sft",
        "task_id": task_id,
        "view_index": view_index,
        "target_index": target_index,
        "demonstration_indices": demonstration_indices,
        "transform_id": transform_id,
        "color_mapping": list(color_mapping),
        "demonstrations": demonstrations,
        "target_input": target["input"],
        "target_output": target["output"],
    }


def build_exact_curriculum_records(
    tasks: dict[str, list[dict[str, Any]]],
    *,
    views_per_task: int = 20,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if views_per_task <= 0:
        raise ValueError("views_per_task must be positive")
    records = [
        build_augmented_record(
            task_id=task_id,
            pairs=tasks[task_id],
            view_index=view_index,
            seed=seed,
        )
        for task_id in sorted(tasks)
        for view_index in range(views_per_task)
    ]
    expected = len(tasks) * views_per_task
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} records, built {len(records)}")
    return records


def format_completion_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    """Format a record, dropping earliest demonstrations only if necessary."""
    formatter = QwenFormatter(tokenizer)
    demonstrations = list(record["demonstrations"])
    demonstration_indices = list(record["demonstration_indices"])
    dropped: list[int] = []
    while True:
        prompt = formatter.fmt_train(demonstrations)
        prompt += formatter.fmt_query([{"input": record["target_input"]}])
        reply = formatter.fmt_reply([record["target_output"]])
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False)) + len(
            tokenizer.encode(reply, add_special_tokens=False)
        )
        if token_count <= max_seq_length:
            return {
                **record,
                "input": prompt,
                "reply": reply,
                "kept_demonstration_indices": demonstration_indices,
                "dropped_demonstration_indices": dropped,
                "sequence_tokens": token_count,
            }
        if not demonstrations:
            raise ValueError(
                f"{record['task_id']} view {record['view_index']}: target alone "
                f"requires {token_count} tokens (limit {max_seq_length})"
            )
        demonstrations.pop(0)
        dropped.append(demonstration_indices.pop(0))


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_task = Counter(record["task_id"] for record in records)
    by_target = Counter(
        (record["task_id"], record["target_index"]) for record in records
    )
    return {
        "records": len(records),
        "tasks": len(by_task),
        "records_per_task": dict(Counter(by_task.values())),
        "target_count_range_per_task": {
            task_id: [
                min(count for (key, _), count in by_target.items() if key == task_id),
                max(count for (key, _), count in by_target.items() if key == task_id),
            ]
            for task_id in sorted(by_task)
        },
    }
