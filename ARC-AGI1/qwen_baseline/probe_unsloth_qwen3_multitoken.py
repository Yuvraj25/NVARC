#!/usr/bin/env python3
"""Parity and speed gate for cached multi-token Qwen3 inference on Kaggle."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch

from patch_unsloth_qwen3_multitoken import patch_unsloth


ARC_TOKENS = list(range(11)) + [15]
MAX_SEQ_LENGTH = 8192
INFERENCE_BUFFER_NAMES = (
    "paged_attention",
    "paged_attention_K",
    "paged_attention_V",
    "temp_QA",
    "temp_KV",
    "RH_Q",
    "temp_O",
    "attention",
    "scalar",
    "half_head_dim",
)


def reset_fast_inference_buffers(model) -> None:
    for module in model.modules():
        for name in INFERENCE_BUFFER_NAMES:
            if hasattr(module, name):
                delattr(module, name)


def load_model(model_path: str, adapter_path: Optional[str]):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def build_prompt_tokens(
    tokenizer,
    test_path: str,
    puzzle_key: str,
    subkey: Optional[str],
    batch_size: int,
):
    from arc_loader import ArcDataset, QwenFormatter

    formatter = QwenFormatter(tokenizer=tokenizer)
    puzzles = ArcDataset.from_file(test_path).change_keys([puzzle_key]).split_multi_replies()
    eval_ds = puzzles.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(
        formatter=formatter,
        name="input",
        max_len=MAX_SEQ_LENGTH - formatter.max_new_tokens(),
    )
    available = sorted(eval_ds.keys)
    if subkey:
        if subkey not in eval_ds.keys:
            raise ValueError(f"Unknown subkey {subkey!r}; available={available}")
        selected = [subkey]
    else:
        by_length = {}
        for key in available:
            token_ids = tokenizer.encode(eval_ds.get(key, formatter)["input"])
            by_length.setdefault(len(token_ids), []).append((key, token_ids))
        compatible = max(by_length.values(), key=len)
        if len(compatible) < batch_size:
            raise RuntimeError(
                f"Only {len(compatible)} prompts share a token length; requested batch {batch_size}"
            )
        selected = [key for key, _ in compatible[:batch_size]]
    prompt_tokens = [
        tokenizer.encode(eval_ds.get(key, formatter)["input"])
        for key in selected
    ]
    if len({len(tokens) for tokens in prompt_tokens}) != 1:
        raise RuntimeError("Selected prompts do not have equal token lengths")
    return selected, prompt_tokens


def position_ids(batch_size: int, start: int, length: int, device):
    positions = torch.arange(start, start + length, device=device, dtype=torch.long)
    return positions.unsqueeze(0).expand(batch_size, -1)


@torch.inference_mode()
def prefill(model, prefix_ids: torch.Tensor):
    return model(input_ids=prefix_ids, return_dict=True, use_cache=True)


@torch.inference_mode()
def make_arc_draft(model, prefix_ids: torch.Tensor, draft_len: int):
    reset_fast_inference_buffers(model)
    prefix = prefill(model, prefix_ids)
    cache = prefix.past_key_values
    logits = prefix.logits[:, -1]
    start = prefix_ids.shape[1]
    draft = []
    sequential_logits = []
    arc_ids = torch.tensor(ARC_TOKENS, device=logits.device)
    for offset in range(draft_len):
        token = arc_ids[logits.index_select(-1, arc_ids).argmax(-1)]
        draft.append(token)
        output = model(
            input_ids=token[:, None],
            position_ids=position_ids(prefix_ids.shape[0], start + offset, 1, prefix_ids.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        logits = output.logits[:, -1]
        sequential_logits.append(logits.float().log_softmax(-1).cpu())
        cache = output.past_key_values
    return torch.stack(draft, dim=1), torch.stack(sequential_logits, dim=1)


@torch.inference_mode()
def block_logprobs(model, prefix_ids: torch.Tensor, draft: torch.Tensor):
    reset_fast_inference_buffers(model)
    prefix = prefill(model, prefix_ids)
    output = model(
        input_ids=draft,
        position_ids=position_ids(prefix_ids.shape[0], prefix_ids.shape[1], draft.shape[1], prefix_ids.device),
        past_key_values=prefix.past_key_values,
        return_dict=True,
        use_cache=True,
    )
    return output.logits.float().log_softmax(-1).cpu(), output.past_key_values


def cache_length(cache) -> int:
    return int(cache[0][0].shape[-2])


@torch.inference_mode()
def full_forward_logprobs(model, prefix_ids: torch.Tensor, draft: torch.Tensor):
    reset_fast_inference_buffers(model)
    combined = torch.cat((prefix_ids, draft), dim=1)
    output = model(input_ids=combined, return_dict=True, use_cache=False)
    start = prefix_ids.shape[1]
    return output.logits[:, start : start + draft.shape[1]].float().log_softmax(-1).cpu()


def diff_report(left: torch.Tensor, right: torch.Tensor, prob_threshold: float = 0.2):
    left_arc = left[:, :, ARC_TOKENS]
    right_arc = right[:, :, ARC_TOKENS]
    diff = (left_arc - right_arc).abs()
    cutoff = math.log(prob_threshold)
    relevant = (left_arc >= cutoff) | (right_arc >= cutoff)
    relevant_diff = diff[relevant]
    position_rows = []
    for position in range(left.shape[1]):
        position_diff = diff[0, position]
        arc_index = int(position_diff.argmax())
        token_id = ARC_TOKENS[arc_index]
        left_top_index = int(left_arc[0, position].argmax())
        right_top_index = int(right_arc[0, position].argmax())
        position_rows.append(
            {
                "position": position,
                "max_diff_token": token_id,
                "max_abs_diff": float(position_diff[arc_index]),
                "left_logprob": float(left_arc[0, position, arc_index]),
                "right_logprob": float(right_arc[0, position, arc_index]),
                "left_top_token": ARC_TOKENS[left_top_index],
                "right_top_token": ARC_TOKENS[right_top_index],
                "threshold_membership_disagreements": int(
                    ((left_arc[0, position] >= cutoff) != (right_arc[0, position] >= cutoff)).sum()
                ),
            }
        )
    return {
        "max_arc_logprob_diff": float(diff.max()),
        "mean_arc_logprob_diff": float(diff.mean()),
        "search_relevant_values": int(relevant.sum()),
        "max_search_relevant_logprob_diff": (
            float(relevant_diff.max()) if relevant_diff.numel() else 0.0
        ),
        "arc_argmax_disagreements": int(
            (left_arc.argmax(-1) != right_arc.argmax(-1)).sum()
        ),
        "threshold_membership_disagreements": int(
            ((left_arc >= cutoff) != (right_arc >= cutoff)).sum()
        ),
        "positions": position_rows,
    }


@torch.inference_mode()
def benchmark(model, prefix_ids: torch.Tensor, draft: torch.Tensor, repeats: int):
    reset_fast_inference_buffers(model)
    prefix = prefill(model, prefix_ids)
    cache = prefix.past_key_values
    batch_size, draft_len = draft.shape
    start = prefix_ids.shape[1]

    # Allocate the paged buffers once; subsequent trials overwrite the suffix.
    model(
        input_ids=draft[:, :1],
        position_ids=position_ids(batch_size, start, 1, prefix_ids.device),
        past_key_values=cache,
        return_dict=True,
        use_cache=True,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sequential_times = []
    for _ in range(repeats):
        current = cache
        started = time.perf_counter()
        for offset in range(draft_len):
            output = model(
                input_ids=draft[:, offset : offset + 1],
                position_ids=position_ids(batch_size, start + offset, 1, prefix_ids.device),
                past_key_values=current,
                return_dict=True,
                use_cache=True,
            )
            current = output.past_key_values
        torch.cuda.synchronize()
        sequential_times.append(time.perf_counter() - started)
    sequential_peak = torch.cuda.max_memory_allocated()

    # Force the scratch tensors through q_len=K once before timing warm calls.
    model(
        input_ids=draft,
        position_ids=position_ids(batch_size, start, draft_len, prefix_ids.device),
        past_key_values=cache,
        return_dict=True,
        use_cache=True,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    block_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        model(
            input_ids=draft,
            position_ids=position_ids(batch_size, start, draft_len, prefix_ids.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        torch.cuda.synchronize()
        block_times.append(time.perf_counter() - started)
    block_peak = torch.cuda.max_memory_allocated()
    return {
        "sequential_median_ms": 1000.0 * float(torch.tensor(sequential_times).median()),
        "block_median_ms": 1000.0 * float(torch.tensor(block_times).median()),
        "speedup": float(torch.tensor(sequential_times).median() / torch.tensor(block_times).median()),
        "sequential_peak_bytes": sequential_peak,
        "block_peak_bytes": block_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth-package-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--puzzle-key", default="136b0064")
    parser.add_argument("--subkey")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--draft-len", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-arc-logprob-diff", type=float, default=0.02)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA and FlashAttention")
    if args.draft_len < 2:
        raise ValueError("--draft-len must be at least 2")

    changed = patch_unsloth(Path(args.unsloth_package_dir))
    print("PATCHED", json.dumps([str(path) for path in changed]))
    model, tokenizer = load_model(args.model_path, args.adapter_path)
    selected, prompt_tokens = build_prompt_tokens(
        tokenizer,
        args.test_path,
        args.puzzle_key,
        args.subkey,
        args.batch_size,
    )
    device = next(model.parameters()).device
    prefix_ids = torch.tensor(prompt_tokens, device=device, dtype=torch.long)

    draft, sequential = make_arc_draft(model, prefix_ids, args.draft_len)
    block_by_length = {}
    block_cache = None
    for length in sorted({2, min(5, args.draft_len), args.draft_len}):
        block_logprob, cache = block_logprobs(model, prefix_ids, draft[:, :length])
        block_by_length[length] = block_logprob
        if length == args.draft_len:
            block_cache = cache
    block = block_by_length[args.draft_len]
    full_forward = full_forward_logprobs(model, prefix_ids, draft)
    seq_vs_block = diff_report(sequential, block)
    seq_vs_full = diff_report(sequential, full_forward)
    block_vs_full = diff_report(block, full_forward)
    full_diff = (sequential - block).abs()
    expected_cache_len = prefix_ids.shape[1] + args.draft_len
    result = {
        "puzzle_key": args.puzzle_key,
        "adapter_path": args.adapter_path,
        "subkeys": selected,
        "batch_size": prefix_ids.shape[0],
        "prefix_tokens": prefix_ids.shape[1],
        "draft_tokens": draft.tolist(),
        "draft_len": args.draft_len,
        "max_arc_logprob_diff": seq_vs_block["max_arc_logprob_diff"],
        "mean_arc_logprob_diff": seq_vs_block["mean_arc_logprob_diff"],
        "max_full_logprob_diff": float(full_diff.max()),
        "continuation_arc_argmax_equal": bool(
            sequential[:, -1, ARC_TOKENS].argmax(-1).equal(block[:, -1, ARC_TOKENS].argmax(-1))
        ),
        "cache_len": cache_length(block_cache),
        "expected_cache_len": expected_cache_len,
        "seq_vs_block": seq_vs_block,
        "seq_vs_full": seq_vs_full,
        "block_vs_full": block_vs_full,
        "seq_vs_block_by_draft_len": {
            str(length): diff_report(sequential[:, :length], values)
            for length, values in block_by_length.items()
        },
    }
    result.update(benchmark(model, prefix_ids, draft, args.repeats))
    result["parity_pass"] = bool(
        seq_vs_block["max_search_relevant_logprob_diff"] <= args.max_arc_logprob_diff
        and result["continuation_arc_argmax_equal"]
        and seq_vs_block["threshold_membership_disagreements"] == 0
        and result["cache_len"] == result["expected_cache_len"]
    )
    result["speed_pass"] = bool(result["speedup"] >= args.min_speedup)
    print("MULTITOKEN_PROBE", json.dumps(result, sort_keys=True))
    if not result["parity_pass"]:
        raise SystemExit("FAIL: cached multi-token logprob/cache parity")
    if not result["speed_pass"]:
        raise SystemExit("FAIL: cached multi-token path did not meet the speed gate")


if __name__ == "__main__":
    main()
