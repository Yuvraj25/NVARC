"""Two-stage repair-aware test-time fine-tuning utilities for ARC."""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_opsd import OpsdPairSplit, build_opsd_examples
from arc_search import ASSISTANT_TOKEN_ID, EOS_ID, USER_TOKEN_ID
from repair_mining import (
    format_reply,
    gold_shape_error_mask,
    grid_to_string,
    parse_rollout_grid,
    restricted_greedy_rollout_batch,
)


REPAIR_TOKEN = "<REPAIR>"
REPAIR_TTFT_METHODS = {"loo_repair_mix", "warm_repair_mix"}


def deterministic_rows(rows: Sequence[Any], count: int, seed: int) -> list[Any]:
    """Select or repeat rows deterministically to obtain exactly ``count`` rows."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    if not rows:
        raise ValueError("Cannot select from an empty row collection")
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected = []
    while len(selected) < count:
        for index in order:
            selected.append(copy.deepcopy(rows[index]))
            if len(selected) == count:
                break
        rng.shuffle(order)
    return selected


def split_for_known_pair(
    puzzle_ds: ArcDataset,
    puzzle_key: str,
    reserved_pair_index: int,
) -> OpsdPairSplit:
    query = copy.deepcopy(puzzle_ds.queries[puzzle_key])
    train = query["train"]
    if not 0 <= reserved_pair_index < len(train):
        raise IndexError(reserved_pair_index)
    sft_pair_indices = [index for index in range(len(train)) if index != reserved_pair_index]
    reserved_pair = copy.deepcopy(train[reserved_pair_index])
    query["train"] = [copy.deepcopy(train[index]) for index in sft_pair_indices]
    reduced = ArcDataset(
        queries={puzzle_key: query},
        replies={},
        keys=[puzzle_key],
        is_orig=puzzle_ds.is_orig,
    )
    return OpsdPairSplit(
        puzzle_key=puzzle_key,
        reserved_pair_index=reserved_pair_index,
        sft_pair_indices=sft_pair_indices,
        reduced_dataset=reduced,
        reserved_pair=reserved_pair,
    )


def build_pair_probe_views(
    puzzle_ds: ArcDataset,
    puzzle_key: str,
    pair_index: int,
    formatter: QwenFormatter,
    view_count: int,
    seed: int,
):
    if view_count < 1:
        raise ValueError("view_count must be positive")
    split = split_for_known_pair(puzzle_ds, puzzle_key, pair_index)
    color_permutations = max(1, math.ceil(view_count / 8))
    examples = build_opsd_examples(
        split,
        formatter=formatter,
        color_permutations=color_permutations,
        cross_view_probability=0.0,
        seed=seed,
    )
    return examples[:view_count]


def repair_prompt(student_prompt: str, prediction: Any, gold: Any) -> str:
    mask = gold_shape_error_mask(prediction, gold)
    return (
        student_prompt
        + format_reply(prediction)
        + "<|im_start|>user\n"
        + REPAIR_TOKEN
        + "\n"
        + grid_to_string(mask)
        + "<|im_end|><|im_start|>assistant\n"
    )


def tokenize_ordinary_text(text: str, tokenizer, max_seq_length: int):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if not input_ids or len(input_ids) > max_seq_length:
        return None
    labels = [-100] * len(input_ids)
    starts = sorted(
        [index for index, token in enumerate(input_ids) if token == USER_TOKEN_ID]
        + [index for index, token in enumerate(input_ids) if token == ASSISTANT_TOKEN_ID]
    )
    ends = [index for index, token in enumerate(input_ids) if token == EOS_ID]
    for turn_index, (start, end) in enumerate(zip(starts, ends)):
        if turn_index % 2 == 1 and start < end:
            labels[start + 2 : end + 1] = input_ids[start + 2 : end + 1]
    if all(label == -100 for label in labels):
        raise RuntimeError("Ordinary TTFT row has no assistant labels")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def tokenize_repair_reply(prompt: str, reply: str, tokenizer, max_seq_length: int):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    reply_ids = tokenizer.encode(reply, add_special_tokens=False)
    input_ids = prompt_ids + reply_ids
    if not prompt_ids or not reply_ids or len(input_ids) > max_seq_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + reply_ids,
    }


def mixed_completion_collator(tokenizer):
    def collate(features):
        maximum = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, maximum), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for index, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[index, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def mine_repair_examples(
    *,
    model,
    tokenizer,
    formatter: QwenFormatter,
    puzzle_ds: ArcDataset,
    puzzle_key: str,
    pair_view_counts: dict[int, int],
    max_seq_length: int,
    max_new_tokens: int,
    seed: int,
    rollout_batch_size: int = 4,
):
    probes = []
    for pair_index, view_count in sorted(pair_view_counts.items()):
        views = build_pair_probe_views(
            puzzle_ds,
            puzzle_key,
            pair_index,
            formatter,
            view_count,
            seed + 1009 * pair_index,
        )
        probes.extend((pair_index, example) for example in views)

    requested_probe_count = len(probes)
    fitting_probes = []
    overlong_probes = []
    for pair_index, example in probes:
        prompt_tokens = len(tokenizer.encode(example.student_prompt, add_special_tokens=False))
        if prompt_tokens + max_new_tokens <= max_seq_length:
            fitting_probes.append((pair_index, example))
        else:
            overlong_probes.append(
                {
                    "pair_index": pair_index,
                    "view_key": example.h_key,
                    "prompt_tokens": prompt_tokens,
                }
            )
    probes = fitting_probes

    rollout_tokens = []
    for offset in range(0, len(probes), rollout_batch_size):
        batch = probes[offset : offset + rollout_batch_size]
        rollout_tokens.extend(
            restricted_greedy_rollout_batch(
                model,
                tokenizer,
                [example.student_prompt for _pair_index, example in batch],
                max_new_tokens=max_new_tokens,
            )
        )

    stats = {
        "requested_views": requested_probe_count,
        "fitting_views": len(probes),
        "overlong_views": len(overlong_probes),
        "overlong_diagnostics": overlong_probes,
        "views_by_pair": {str(key): value for key, value in pair_view_counts.items()},
        "valid_wrong": 0,
        "exact": 0,
        "invalid": 0,
        "wrong_shape": 0,
        "repair_rows_by_pair": Counter(),
        "diagnostics": [],
    }
    repair_rows = []
    for (pair_index, example), tokens in zip(probes, rollout_tokens):
        prediction, invalid_reason = parse_rollout_grid(tokenizer, tokens)
        diagnostic = {
            "pair_index": pair_index,
            "view_key": example.h_key,
            "invalid_reason": invalid_reason,
            "prediction_shape": None,
            "gold_shape": list(np.asarray(example.transformed_output).shape),
            "exact": False,
        }
        if prediction is None:
            stats["invalid"] += 1
            stats["diagnostics"].append(diagnostic)
            continue
        prediction_array = np.asarray(prediction)
        gold_array = np.asarray(example.transformed_output)
        diagnostic["prediction_shape"] = list(prediction_array.shape)
        diagnostic["exact"] = bool(np.array_equal(prediction_array, gold_array))
        if diagnostic["exact"]:
            stats["exact"] += 1
            stats["diagnostics"].append(diagnostic)
            continue
        stats["valid_wrong"] += 1
        if prediction_array.shape != gold_array.shape:
            stats["wrong_shape"] += 1
        tokenized = tokenize_repair_reply(
            repair_prompt(example.student_prompt, prediction, example.transformed_output),
            example.gold_reply,
            tokenizer,
            max_seq_length,
        )
        if tokenized is None:
            diagnostic["invalid_reason"] = "repair_sequence_too_long"
            stats["diagnostics"].append(diagnostic)
            continue
        tokenized["record_type"] = "repair"
        tokenized["pair_index"] = pair_index
        repair_rows.append(tokenized)
        stats["repair_rows_by_pair"][pair_index] += 1
        stats["diagnostics"].append(diagnostic)

    stats["repair_rows_by_pair"] = {
        str(key): value for key, value in sorted(stats["repair_rows_by_pair"].items())
    }
    stats["usable_repair_rows"] = len(repair_rows)
    return repair_rows, stats


def build_stage_two_mixture(
    *,
    ordinary_rows: Sequence[dict[str, Any]],
    repair_rows: Sequence[dict[str, Any]],
    total_steps: int,
    repair_fraction: float,
    seed: int,
):
    if not 0.0 <= repair_fraction <= 1.0:
        raise ValueError("repair_fraction must be in [0, 1]")
    repair_steps = round(total_steps * repair_fraction) if repair_rows else 0
    ordinary_steps = total_steps - repair_steps
    mixture = deterministic_rows(ordinary_rows, ordinary_steps, seed)
    if repair_steps:
        mixture.extend(deterministic_rows(repair_rows, repair_steps, seed + 1))
    random.Random(seed + 2).shuffle(mixture)
    return mixture, {"ordinary_steps": ordinary_steps, "repair_steps": repair_steps}
