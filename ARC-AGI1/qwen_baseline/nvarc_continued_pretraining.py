"""Data plumbing for time-bounded continued pretraining on NVARC.

The released corpus is a collection of Hugging Face ``save_to_disk`` datasets.
Each row contains alternating user/assistant messages.  Sampling is lazy and
deterministic so a long Kaggle run does not materialize or tokenize millions of
records before the first optimizer step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_WEIGHTS: dict[str, int] = {
    "nvarc_full": 40,
    "nvarc_training": 35,
    "arc2_training": 15,
    "rearc": 5,
    "concept": 3,
    "mini": 2,
}


def validate_source_weights(weights: Mapping[str, int]) -> None:
    if set(weights) != set(SOURCE_WEIGHTS):
        raise ValueError(f"Expected sources {sorted(SOURCE_WEIGHTS)}, got {sorted(weights)}")
    if any(not isinstance(value, int) or value <= 0 for value in weights.values()):
        raise ValueError("Source weights must be positive integers")
    if sum(weights.values()) != 100:
        raise ValueError(f"Source weights must sum to 100, got {sum(weights.values())}")


def _stable_u64(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def weighted_source_schedule(
    weights: Mapping[str, int] = SOURCE_WEIGHTS,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return a deterministic shuffled 100-record source schedule."""
    validate_source_weights(weights)
    schedule = [source for source, count in sorted(weights.items()) for _ in range(count)]
    # Fisher-Yates with stable hash-derived swaps avoids Python hash/random drift.
    for index in range(len(schedule) - 1, 0, -1):
        swap = _stable_u64(seed, "source-schedule", index) % (index + 1)
        schedule[index], schedule[swap] = schedule[swap], schedule[index]
    return tuple(schedule)


def discover_augmented_corpus_root(input_root: Path) -> Path:
    candidates = [
        input_root / "nvarc-augmented-puzzles",
        input_root / "datasets" / "sorokin" / "nvarc-augmented-puzzles",
    ]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / source).is_dir() for source in SOURCE_WEIGHTS):
            return candidate
    matches = [
        path
        for path in input_root.rglob("nvarc-augmented-puzzles")
        if path.is_dir() and all((path / source).is_dir() for source in SOURCE_WEIGHTS)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one NVARC augmented corpus root, found {matches}")
    return matches[0]


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError("A corpus record must contain complete user/assistant pairs")
    result: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index} is not a mapping")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            raise ValueError(
                f"Message {index} has role {message.get('role')!r}, expected {expected_role!r}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"Message {index} has empty/non-string content")
        result.append({"role": expected_role, "content": content})
    return result


def tokenize_all_assistant_outputs(
    messages: Sequence[dict[str, str]],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize ARC chat turns and supervise every assistant grid.

    As in the released NVARC preprocessing, the assistant generation prefix is
    part of the user turn and receives no loss.  The assistant grid and its
    closing ``<|im_end|>`` receive loss.  If needed, complete earliest pairs are
    removed until the record fits; no turn is truncated in the middle.
    """
    checked = validate_messages(list(messages))
    dropped_pairs = 0
    while checked:
        input_ids: list[int] = []
        labels: list[int] = []
        assistant_token_counts: list[int] = []
        for pair_index in range(0, len(checked), 2):
            user = checked[pair_index]["content"]
            assistant = checked[pair_index + 1]["content"]
            prompt = (
                "<|im_start|>user\n"
                + user
                + "<|im_end|><|im_start|>assistant\n"
            )
            reply = assistant + "<|im_end|>"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            reply_ids = tokenizer.encode(reply, add_special_tokens=False)
            if not prompt_ids or not reply_ids:
                raise ValueError("A user prompt or assistant reply tokenized to empty")
            input_ids.extend(prompt_ids)
            labels.extend([-100] * len(prompt_ids))
            input_ids.extend(reply_ids)
            labels.extend(reply_ids)
            assistant_token_counts.append(len(reply_ids))
        if len(input_ids) <= max_seq_length:
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "assistant_outputs": len(assistant_token_counts),
                "assistant_token_counts": assistant_token_counts,
                "supervised_tokens": sum(assistant_token_counts),
                "sequence_tokens": len(input_ids),
                "dropped_leading_pairs": dropped_pairs,
            }
        checked = checked[2:]
        dropped_pairs += 1
    raise ValueError("No complete input/output pair fits inside max_seq_length")


@dataclass(frozen=True)
class CorpusSelection:
    virtual_index: int
    source: str
    row_index: int


@dataclass(frozen=True)
class WallClockPlan:
    started: float
    stage_boundary: float
    deadline: float
    periodic_checkpoints: tuple[float, ...]


def build_wall_clock_plan(
    *,
    started: float,
    budget_seconds: float,
    canon_only_fraction: float,
    checkpoint_seconds: float,
) -> WallClockPlan:
    if budget_seconds <= 0 or checkpoint_seconds <= 0:
        raise ValueError("Time budget and checkpoint interval must be positive")
    if not 0 < canon_only_fraction < 1:
        raise ValueError("Canon-only fraction must lie between zero and one")
    deadline = started + budget_seconds
    checkpoints = []
    current = started + checkpoint_seconds
    while current < deadline:
        checkpoints.append(current)
        current += checkpoint_seconds
    return WallClockPlan(
        started=started,
        stage_boundary=started + budget_seconds * canon_only_fraction,
        deadline=deadline,
        periodic_checkpoints=tuple(checkpoints),
    )


class WeightedNVARCCorpus:
    """Lazy map-style dataset with exact 100-record source proportions."""

    def __init__(
        self,
        datasets_by_source: Mapping[str, Any],
        tokenizer: Any,
        *,
        max_seq_length: int,
        virtual_length: int,
        seed: int,
        weights: Mapping[str, int] = SOURCE_WEIGHTS,
    ) -> None:
        validate_source_weights(weights)
        if virtual_length <= 0:
            raise ValueError("virtual_length must be positive")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        missing = set(weights) - set(datasets_by_source)
        if missing:
            raise ValueError(f"Missing source datasets: {sorted(missing)}")
        empty = [source for source in weights if len(datasets_by_source[source]) == 0]
        if empty:
            raise ValueError(f"Empty source datasets: {empty}")
        self.datasets_by_source = dict(datasets_by_source)
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.virtual_length = int(virtual_length)
        self.seed = int(seed)
        self.schedule = weighted_source_schedule(weights, seed=seed)

    def __len__(self) -> int:
        return self.virtual_length

    def selection(self, index: int) -> CorpusSelection:
        if index < 0:
            index += self.virtual_length
        if not 0 <= index < self.virtual_length:
            raise IndexError(index)
        source = self.schedule[index % len(self.schedule)]
        source_length = len(self.datasets_by_source[source])
        row_index = _stable_u64(self.seed, source, index) % source_length
        return CorpusSelection(index, source, int(row_index))

    def __getitem__(self, index: int) -> dict[str, Any]:
        selection = self.selection(index)
        row = self.datasets_by_source[selection.source][selection.row_index]
        tokenized = tokenize_all_assistant_outputs(
            row["messages"], self.tokenizer, max_seq_length=self.max_seq_length
        )
        # Trainer removes non-model columns.  Keep provenance scalar-only so a
        # CPU plumbing check can inspect it before Trainer handoff.
        tokenized["source"] = selection.source
        tokenized["source_row_index"] = selection.row_index
        tokenized["puzzle_name"] = str(row.get("puzzle_name", ""))
        return tokenized


def load_released_sources(corpus_root: Path) -> dict[str, Any]:
    from datasets import load_from_disk

    return {
        source: load_from_disk(str(corpus_root / source))
        for source in SOURCE_WEIGHTS
    }
