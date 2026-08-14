"""Dataset and tokenizer utilities for offline ARC error-mask repair SFT."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from repair_mining import format_reply, grid_to_string, zero_mask


REPAIR_TOKEN = "<REPAIR>"
STRUCTURAL_TOKENS = ("user", "assistant", "<|im_start|>", "<|im_end|>")


def merge_repair_adapter_into_base(
    model: Any,
    tokenizer: Any,
    adapter_path: str | Path,
    *,
    auto_tokenizer_cls: Any,
    peft_model_cls: Any,
) -> tuple[Any, Any]:
    """Merge the offline repair adapter before creating a fresh TTFT adapter.

    The repair checkpoint owns a seventeenth token plus full saved copies of
    ``embed_tokens`` and ``lm_head``.  The base model must therefore be resized
    before PEFT loads the checkpoint; stacking an old 16-token TTFT adapter on
    top would be shape-incompatible and would not test the repaired
    initialization fairly.
    """
    adapter_path = Path(adapter_path)
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (adapter_path / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Repair adapter is incomplete at {adapter_path}: missing={missing}"
        )

    repaired_tokenizer = auto_tokenizer_cls.from_pretrained(
        str(adapter_path),
        local_files_only=True,
    )
    base_vocab_size = len(tokenizer)
    repaired_vocab_size = len(repaired_tokenizer)
    repair_token_id = repaired_tokenizer.convert_tokens_to_ids(REPAIR_TOKEN)
    if base_vocab_size != 16:
        raise RuntimeError(f"Expected 16-token ARC base vocabulary, got {base_vocab_size}")
    if repaired_vocab_size != 17 or repair_token_id != 16:
        raise RuntimeError(
            "Unexpected repair tokenizer layout: "
            f"vocab={repaired_vocab_size} repair_token_id={repair_token_id}"
        )

    try:
        model.resize_token_embeddings(repaired_vocab_size, mean_resizing=False)
    except TypeError:
        # Older Transformers releases do not expose ``mean_resizing``.
        model.resize_token_embeddings(repaired_vocab_size)

    repaired_model = peft_model_cls.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )
    repaired_model = repaired_model.merge_and_unload(safe_merge=True)
    input_rows = repaired_model.get_input_embeddings().weight.shape[0]
    output_rows = repaired_model.get_output_embeddings().weight.shape[0]
    if input_rows != repaired_vocab_size or output_rows != repaired_vocab_size:
        raise RuntimeError(
            "Merged repair model has inconsistent vocabulary matrices: "
            f"input={input_rows} output={output_rows} tokenizer={repaired_vocab_size}"
        )
    return repaired_model, repaired_tokenizer


def ordinary_solve_prompt(record: dict[str, Any]) -> str:
    """Recover the ordinary query prompt preceding a mined repair trajectory."""
    candidate_reply = format_reply(record["prediction"])
    repair_suffix_start = candidate_reply + "<|im_start|>user\n" + REPAIR_TOKEN + "\n"
    position = record["input"].rfind(repair_suffix_start)
    if position < 0:
        raise ValueError("Could not locate candidate-to-repair boundary")
    return record["input"][:position]


def reply_grid(reply: str) -> list[list[int]]:
    suffix = "<|im_end|>"
    if not reply.endswith(suffix):
        raise ValueError("Reply is missing <|im_end|>")
    body = reply[: -len(suffix)]
    rows = [[int(character) for character in row] for row in body.splitlines()]
    if not rows or any(not row for row in rows):
        raise ValueError("Reply does not contain a rectangular grid")
    if len({len(row) for row in rows}) != 1:
        raise ValueError("Reply grid is ragged")
    return rows


def build_solve_replay_example(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "solve_replay",
        "anchor_id": record["anchor_id"],
        "puzzle_id": record["puzzle_id"],
        "input": ordinary_solve_prompt(record),
        "reply": record["reply"],
    }


def build_zero_mask_noop_example(record: dict[str, Any]) -> dict[str, Any]:
    solve_prompt = ordinary_solve_prompt(record)
    gold = reply_grid(record["reply"])
    return {
        "record_type": "repair_noop",
        "anchor_id": record["anchor_id"],
        "puzzle_id": record["puzzle_id"],
        "input": (
            solve_prompt
            + format_reply(gold)
            + "<|im_start|>user\n"
            + REPAIR_TOKEN
            + "\n"
            + grid_to_string(zero_mask(gold))
            + "<|im_end|><|im_start|>assistant\n"
        ),
        "reply": record["reply"],
    }


def build_training_mixture(
    repair_records: Sequence[dict[str, Any]],
    *,
    solve_replay_fraction: float = 0.15,
    noop_fraction: float = 0.01,
    seed: int = 20260812,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use every repair failure once and add replay/no-op examples to target fractions."""
    if not repair_records:
        raise ValueError("At least one repair record is required")
    if solve_replay_fraction < 0 or noop_fraction < 0:
        raise ValueError("Mixture fractions must be non-negative")
    repair_fraction = 1.0 - solve_replay_fraction - noop_fraction
    if repair_fraction <= 0:
        raise ValueError("Repair failures must retain positive mixture mass")
    if any(record.get("record_type") != "repair_failure" for record in repair_records):
        raise ValueError("Only repair_failure records may seed the mixture")

    rng = random.Random(seed)
    total_target = math.ceil(len(repair_records) / repair_fraction)
    replay_count = round(total_target * solve_replay_fraction)
    noop_count = max(1, round(total_target * noop_fraction)) if noop_fraction else 0

    def sample_records(count: int) -> list[dict[str, Any]]:
        if count <= len(repair_records):
            return rng.sample(list(repair_records), count)
        return [rng.choice(repair_records) for _ in range(count)]

    mixture = [dict(record) for record in repair_records]
    mixture.extend(build_solve_replay_example(record) for record in sample_records(replay_count))
    mixture.extend(build_zero_mask_noop_example(record) for record in sample_records(noop_count))
    rng.shuffle(mixture)
    counts = Counter(example["record_type"] for example in mixture)
    manifest = {
        "seed": seed,
        "requested_fractions": {
            "repair_failure": repair_fraction,
            "solve_replay": solve_replay_fraction,
            "repair_noop": noop_fraction,
        },
        "counts": dict(sorted(counts.items())),
        "actual_fractions": {
            key: value / len(mixture) for key, value in sorted(counts.items())
        },
        "total_examples": len(mixture),
    }
    return mixture, manifest


def add_and_initialize_repair_token(
    model: Any,
    tokenizer: Any,
    *,
    repair_token: str = REPAIR_TOKEN,
) -> int:
    """Add one control token and initialize it from structural-token embeddings."""
    import torch

    old_vocab_size = len(tokenizer)
    if repair_token in tokenizer.get_vocab():
        raise ValueError(f"{repair_token} is already present in the tokenizer")
    # Transformers versions differ here: some tokenizer classes expose
    # ``additional_special_tokens`` as a direct attribute, while newer Qwen2
    # tokenizers expose the same information only through ``special_tokens_map``.
    existing_special = getattr(tokenizer, "additional_special_tokens", None)
    if existing_special is None:
        special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
        existing_special = special_tokens_map.get("additional_special_tokens", [])
    existing_special = list(existing_special)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": existing_special + [repair_token]}
    )
    if added != 1 or len(tokenizer) != old_vocab_size + 1:
        raise RuntimeError(
            f"Expected exactly one new token: added={added}, old={old_vocab_size}, new={len(tokenizer)}"
        )
    repair_token_id = tokenizer.convert_tokens_to_ids(repair_token)
    structural_ids = tokenizer.convert_tokens_to_ids(list(STRUCTURAL_TOKENS))
    if any(token_id is None or token_id < 0 or token_id >= old_vocab_size for token_id in structural_ids):
        raise RuntimeError(f"Could not resolve structural token IDs: {structural_ids}")

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_weight = model.get_input_embeddings().weight
    output_layer = model.get_output_embeddings()
    with torch.no_grad():
        input_weight[repair_token_id].copy_(input_weight[structural_ids].mean(dim=0))
        if output_layer is not None and output_layer.weight.shape[0] == len(tokenizer):
            output_layer.weight[repair_token_id].copy_(
                output_layer.weight[structural_ids].mean(dim=0)
            )
    return repair_token_id


def tokenize_completion_only(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize one example while applying loss only to its final reply."""
    prompt_ids = tokenizer.encode(example["input"], add_special_tokens=False)
    reply_ids = tokenizer.encode(example["reply"], add_special_tokens=False)
    if not prompt_ids or not reply_ids:
        raise ValueError("Prompt and reply must tokenize to non-empty sequences")
    input_ids = prompt_ids + reply_ids
    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"Sequence has {len(input_ids)} tokens, exceeding max_seq_length={max_seq_length}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + reply_ids,
    }
