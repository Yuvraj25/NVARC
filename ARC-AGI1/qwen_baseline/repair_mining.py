"""Utilities for mining ARC error-mask repair examples from raw NVARC puzzles.

The raw NVARC synthetic-puzzle dataset stores one JSON file per underlying
puzzle.  This module deliberately samples those files rather than treating the
24/32 augmented SFT records as independent puzzles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ARC_TOKENS = list(range(11)) + [15]
PAD_ID = 13
EOS_ID = 15


def _stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def validate_grid(grid: Any) -> bool:
    try:
        array = np.asarray(grid)
    except Exception:
        return False
    return (
        array.ndim == 2
        and 1 <= array.shape[0] <= 30
        and 1 <= array.shape[1] <= 30
        and np.issubdtype(array.dtype, np.integer)
        and bool(np.all((0 <= array) & (array <= 9)))
    )


def validate_pairs(pairs: Any, minimum: int = 3) -> bool:
    return (
        isinstance(pairs, list)
        and len(pairs) >= minimum
        and all(
            isinstance(pair, dict)
            and validate_grid(pair.get("input"))
            and validate_grid(pair.get("output"))
            for pair in pairs
        )
    )


def dihedral_transform(grid: Any, transform_id: int) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int8)
    if transform_id == 0:
        return array.copy()
    if transform_id == 1:
        return np.rot90(array, 1)
    if transform_id == 2:
        return np.rot90(array, 2)
    if transform_id == 3:
        return np.rot90(array, 3)
    if transform_id == 4:
        return np.fliplr(array)
    if transform_id == 5:
        return np.flipud(array)
    if transform_id == 6:
        return array.T
    if transform_id == 7:
        return np.fliplr(np.rot90(array, 1))
    raise ValueError(f"Invalid dihedral transform: {transform_id}")


def transform_grid(grid: Any, transform_id: int, color_mapping: Sequence[int]) -> list[list[int]]:
    if sorted(color_mapping) != list(range(10)):
        raise ValueError("color_mapping must be a permutation of 0..9")
    transformed = dihedral_transform(grid, transform_id)
    return np.asarray(color_mapping, dtype=np.int8)[transformed].tolist()


@dataclass(frozen=True)
class LeaveOneOutProbe:
    subset: str
    puzzle_id: str
    anchor_id: str
    source_path: str
    demonstration_indices: tuple[int, ...]
    query_index: int
    transform_id: int
    color_mapping: tuple[int, ...]
    demonstrations: tuple[dict[str, Any], ...]
    query_input: list[list[int]]
    gold_output: list[list[int]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def build_leave_one_out_probe(
    pairs: list[dict[str, Any]],
    *,
    subset: str,
    puzzle_id: str,
    anchor_id: str,
    source_path: str,
    seed: int,
    num_demonstrations: int = 2,
) -> LeaveOneOutProbe:
    if not validate_pairs(pairs, minimum=num_demonstrations + 1):
        raise ValueError(f"Need at least {num_demonstrations + 1} valid pairs")

    rng = np.random.default_rng(_stable_u64(f"{seed}:{subset}:{puzzle_id}"))
    selected = rng.choice(len(pairs), size=num_demonstrations + 1, replace=False).tolist()
    demonstration_indices = tuple(int(index) for index in selected[:-1])
    query_index = int(selected[-1])
    transform_id = int(rng.integers(0, 8))
    color_mapping = tuple(int(value) for value in rng.permutation(10))

    def transform_pair(index: int) -> dict[str, Any]:
        pair = pairs[index]
        return {
            "input": transform_grid(pair["input"], transform_id, color_mapping),
            "output": transform_grid(pair["output"], transform_id, color_mapping),
        }

    demonstrations = tuple(transform_pair(index) for index in demonstration_indices)
    query_pair = transform_pair(query_index)
    return LeaveOneOutProbe(
        subset=subset,
        puzzle_id=puzzle_id,
        anchor_id=anchor_id,
        source_path=source_path,
        demonstration_indices=demonstration_indices,
        query_index=query_index,
        transform_id=transform_id,
        color_mapping=color_mapping,
        demonstrations=demonstrations,
        query_input=query_pair["input"],
        gold_output=query_pair["output"],
    )


def load_probe_from_path(
    path: Path,
    *,
    subset: str,
    seed: int,
    num_demonstrations: int = 2,
) -> LeaveOneOutProbe:
    pairs = json.loads(path.read_text())
    return build_leave_one_out_probe(
        pairs,
        subset=subset,
        puzzle_id=path.stem,
        anchor_id=path.parent.name,
        source_path=str(path),
        seed=seed,
        num_demonstrations=num_demonstrations,
    )


def discover_subset_root(input_root: Path, subset: str) -> Path:
    # Kaggle mounts this release under its dataset slug.  Prefer the direct
    # path because recursively walking ~100k small JSON files is needlessly
    # slow on the read-only FUSE mount.
    direct_candidates = [
        input_root / "nvarc-synthetic-puzzles" / subset,
        input_root / "datasets" / "sorokin" / "nvarc-synthetic-puzzles" / subset,
    ]
    for direct in direct_candidates:
        if direct.is_dir() and any(direct.glob("*/*.json")):
            return direct

    candidates = []
    for path in input_root.rglob(subset):
        if path.is_dir() and any(path.glob("*/*.json")):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one raw {subset} root, found: {candidates}")
    return candidates[0]


def deterministic_sample_paths(
    subset_root: Path,
    *,
    count: int,
    seed: int,
    excluded_anchor_ids: Iterable[str] = (),
) -> list[Path]:
    """Choose stable files while spreading probes across source-anchor families.

    The raw release has roughly 100k small files.  Enumerating every file on a
    Kaggle mount is avoidable: rank the much smaller anchor directories, then
    draw one file per anchor before taking a second file from any anchor.
    """
    excluded = set(excluded_anchor_ids)
    anchors = [path for path in subset_root.iterdir() if path.is_dir() and path.name not in excluded]
    anchors.sort(key=lambda path: _stable_u64(f"{seed}:anchor:{path.name}"))

    selected = []
    round_index = 0
    while len(selected) < count:
        added = 0
        for anchor in anchors:
            files = list(anchor.glob("*.json"))
            files.sort(key=lambda path: _stable_u64(f"{seed}:file:{path.name}"))
            if round_index < len(files):
                selected.append(files[round_index])
                added += 1
                if len(selected) == count:
                    return selected
        if added == 0:
            break
        round_index += 1
    return selected


def grid_to_string(grid: Any) -> str:
    if not validate_grid(grid):
        raise ValueError("Invalid ARC grid")
    return "\n".join("".join(str(int(cell)) for cell in row) for row in grid)


def format_prompt(probe: LeaveOneOutProbe) -> str:
    text = ""
    for pair in probe.demonstrations:
        text += (
            "<|im_start|>user\n"
            + grid_to_string(pair["input"])
            + "<|im_end|><|im_start|>assistant\n"
            + grid_to_string(pair["output"])
            + "<|im_end|>"
        )
    return (
        text
        + "<|im_start|>user\n"
        + grid_to_string(probe.query_input)
        + "<|im_end|><|im_start|>assistant\n"
    )


def format_reply(grid: Any) -> str:
    return grid_to_string(grid) + "<|im_end|>"


def gold_shape_error_mask(prediction: Any, gold: Any) -> list[list[int]]:
    """Return a gold-shaped mask; absent cells are wrong and extras are cropped."""
    if not validate_grid(prediction) or not validate_grid(gold):
        raise ValueError("prediction and gold must be valid rectangular ARC grids")
    prediction_array = np.asarray(prediction)
    gold_array = np.asarray(gold)
    mask = np.ones(gold_array.shape, dtype=np.int8)
    rows = min(prediction_array.shape[0], gold_array.shape[0])
    cols = min(prediction_array.shape[1], gold_array.shape[1])
    mask[:rows, :cols] = prediction_array[:rows, :cols] != gold_array[:rows, :cols]
    return mask.tolist()


def error_mask_diagnostics(prediction: Any, gold: Any) -> dict[str, Any]:
    """Describe cell errors while keeping the repair mask at the gold shape.

    Missing predicted cells are marked in the gold-shaped mask.  Extra cells
    cannot be represented inside that mask, so they are counted separately;
    the target shape tells the repair model which suffix rows/columns to drop.
    """
    mask = gold_shape_error_mask(prediction, gold)
    prediction_array = np.asarray(prediction)
    gold_array = np.asarray(gold)
    overlap_rows = min(prediction_array.shape[0], gold_array.shape[0])
    overlap_cols = min(prediction_array.shape[1], gold_array.shape[1])
    wrong_or_missing = int(np.asarray(mask).sum())
    extra = int(prediction_array.size - overlap_rows * overlap_cols)
    return {
        "error_mask": mask,
        "prediction_shape": list(prediction_array.shape),
        "gold_shape": list(gold_array.shape),
        "shape_equal": prediction_array.shape == gold_array.shape,
        "wrong_or_missing_gold_cells": wrong_or_missing,
        "extra_prediction_cells": extra,
        "total_wrong_missing_or_extra_cells": wrong_or_missing + extra,
    }


def parse_rollout_grid(tokenizer: Any, token_ids: Sequence[int]) -> tuple[list[list[int]] | None, str | None]:
    if not token_ids:
        return None, "empty_rollout"
    if token_ids[-1] != EOS_ID:
        return None, "missing_eos"
    text = tokenizer.decode(list(token_ids[:-1]))
    try:
        rows = [[int(character) for character in line] for line in text.strip().split("\n")]
    except ValueError:
        return None, "non_digit_token"
    if not validate_grid(rows):
        return None, "malformed_grid"
    return rows, None


def stabilize_inference_state(model: Any) -> None:
    """Undo pinned-Unsloth generate() state changes before the next forward.

    In the 2025-09 Kaggle stack, the patched generate path can leave decoder
    layers with ``gradient_checkpointing=True`` even though their checkpoint
    callback is absent.  The following ordinary forward then fails.  This is
    an inference-only pipeline, so checkpointing must remain disabled.
    """
    model.eval()
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False


def teacher_forced_metrics(model: Any, tokenizer: Any, prompt: str, gold_reply: str) -> dict[str, Any]:
    import torch

    stabilize_inference_state(model)
    prompt_ids = tokenizer.encode(prompt)
    gold_ids = tokenizer.encode(gold_reply)
    if not prompt_ids or not gold_ids:
        raise ValueError("Prompt and gold reply must tokenize to non-empty sequences")
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + gold_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids, return_dict=True, use_cache=False).logits[0]
    completion_logits = logits[len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(gold_ids)].float()
    targets = torch.tensor(gold_ids, device=device, dtype=torch.long)
    legal_ids = torch.tensor(ARC_TOKENS, device=device, dtype=torch.long)
    legal_argmax = legal_ids[completion_logits[:, legal_ids].argmax(dim=-1)]
    log_prob = torch.log_softmax(completion_logits, dim=-1)
    positions = torch.arange(len(gold_ids), device=device)
    token_nll = -log_prob[positions, targets]
    wrong = (legal_argmax != targets).nonzero(as_tuple=False).flatten()
    first_wrong = int(wrong[0].cpu()) if len(wrong) else None
    result = {
        "prompt_tokens": len(prompt_ids),
        "gold_tokens": len(gold_ids),
        "gold_nll": float(token_nll.sum().cpu()),
        "gold_mean_nll": float(token_nll.mean().cpu()),
        "restricted_greedy_exact": len(wrong) == 0,
        "wrong_argmax_tokens": int(len(wrong)),
        "first_wrong_token": first_wrong,
    }
    del input_ids, logits, completion_logits, targets, legal_ids, legal_argmax, log_prob, token_nll, wrong
    return result


def restricted_greedy_rollout(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
) -> list[int]:
    import torch
    from transformers import LogitsProcessorList

    class ArcOnlyLogitsProcessor:
        def __call__(self, input_ids, scores):
            masked = torch.full_like(scores, -torch.inf)
            masked[:, ARC_TOKENS] = scores[:, ARC_TOKENS]
            return masked

    stabilize_inference_state(model)
    prompt_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=EOS_ID,
                pad_token_id=PAD_ID,
                logits_processor=LogitsProcessorList([ArcOnlyLogitsProcessor()]),
                use_cache=True,
            )
    finally:
        stabilize_inference_state(model)
    result = generated[0, len(prompt_ids) :].tolist()
    del input_ids, generated
    return result
