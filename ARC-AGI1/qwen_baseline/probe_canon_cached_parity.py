"""Compare full-sequence and cached Canon-AC logits on identical forced tokens."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from arc_canon import (
    CanonDFSCache,
    add_canon_ac_modules,
    canon_cached_step,
    canon_parameters,
    canon_prefill,
    install_canon_ac_training_hooks,
    load_canon_state,
    prepare_canon_inference,
)
from arc_loader import ArcDataset, QwenFormatter
from arc_search import ARC_TOKENS, EOS_ID
from arc_search import default_max_score, inference_turbo_dfs
from arc_search_multitoken import inference_turbo_dfs_multitoken


def full_next_logits(model, prefix: torch.Tensor, reply: torch.Tensor) -> torch.Tensor:
    combined = torch.cat((prefix, reply), dim=1)
    outputs = model(input_ids=combined, return_dict=True, use_cache=False)
    prefix_length = prefix.shape[1]
    reply_length = reply.shape[1]
    return outputs.logits[
        0,
        prefix_length - 1 : prefix_length + reply_length - 1,
    ].float().cpu()


def manual_cached_next_logits(
    model,
    prefix: torch.Tensor,
    reply_ids: list[int],
) -> torch.Tensor:
    prefill, cache = canon_prefill(model, prefix)
    cached = [prefill.logits[0, -1].float().cpu()]
    prefix_length = prefix.shape[1]
    for offset, token_id in enumerate(reply_ids[:-1]):
        step = canon_cached_step(
            model,
            input_ids=torch.tensor([[token_id]], dtype=torch.long, device=model.device),
            position_ids=torch.tensor(
                [[prefix_length + offset]], dtype=torch.long, device=model.device
            ),
            cache=cache,
        )
        cache = step.past_key_values
        cached.append(step.logits[0, -1].float().cpu())
    return torch.stack(cached)


def native_cached_next_logits(
    model,
    prefix: torch.Tensor,
    reply_ids: list[int],
) -> torch.Tensor:
    """Use Unsloth's own cached-generation path; valid here with zero Canon."""
    prefill = model(input_ids=prefix, return_dict=True, use_cache=True)
    cache = prefill.past_key_values
    cached = [prefill.logits[0, -1].float().cpu()]
    prefix_length = prefix.shape[1]
    for offset, token_id in enumerate(reply_ids[:-1]):
        step = model(
            input_ids=torch.tensor([[token_id]], dtype=torch.long, device=model.device),
            position_ids=torch.tensor(
                [[prefix_length + offset]], dtype=torch.long, device=model.device
            ),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        cache = step.past_key_values
        cached.append(step.logits[0, -1].float().cpu())
    return torch.stack(cached)


def sibling_backtrack_logits(
    model,
    prefix: torch.Tensor,
    reply_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explore one branch, then decode a sibling from the shared parent."""
    if len(reply_ids) < 2:
        raise ValueError("Sibling diagnostic requires at least two reply tokens")
    first = int(reply_ids[0])
    sibling = (first + 1) % 10 if first in range(10) else 0
    prefix_length = prefix.shape[1]

    prefill, parent_cache = canon_prefill(model, prefix)
    first_branch = canon_cached_step(
        model,
        input_ids=torch.tensor([[first]], dtype=torch.long, device=model.device),
        position_ids=torch.tensor([[prefix_length]], dtype=torch.long, device=model.device),
        cache=parent_cache,
    )
    # Go one token deeper so the paged buffer contains a descendant branch.
    canon_cached_step(
        model,
        input_ids=torch.tensor(
            [[int(reply_ids[1])]], dtype=torch.long, device=model.device
        ),
        position_ids=torch.tensor(
            [[prefix_length + 1]], dtype=torch.long, device=model.device
        ),
        cache=first_branch.past_key_values,
    )
    sibling_step = canon_cached_step(
        model,
        input_ids=torch.tensor([[sibling]], dtype=torch.long, device=model.device),
        position_ids=torch.tensor([[prefix_length]], dtype=torch.long, device=model.device),
        cache=parent_cache,
    )
    full_sibling = model(
        input_ids=torch.cat(
            (
                prefix,
                torch.tensor([[sibling]], dtype=torch.long, device=model.device),
            ),
            dim=1,
        ),
        return_dict=True,
        use_cache=False,
    ).logits[0, -1].float().cpu()
    return full_sibling.unsqueeze(0), sibling_step.logits[0, -1].float().cpu().unsqueeze(0)


def multitoken_block_logits(
    model,
    prefix: torch.Tensor,
    reply_ids: list[int],
    block_length: int = 9,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Compare a q9 Canon cache call with the equivalent full forward."""
    length = min(block_length, len(reply_ids) - 1)
    draft = reply_ids[:length]
    expected = reply_ids[1 : length + 1]
    prefill, parent = canon_prefill(model, prefix)
    start = prefix.shape[1]
    block = canon_cached_step(
        model,
        input_ids=torch.tensor([draft], dtype=torch.long, device=model.device),
        position_ids=torch.arange(
            start, start + length, dtype=torch.long, device=model.device
        ).unsqueeze(0),
        cache=parent,
    )
    combined = torch.cat(
        (prefix, torch.tensor([draft], dtype=torch.long, device=model.device)), dim=1
    )
    full = model(input_ids=combined, return_dict=True, use_cache=False).logits[
        0, start : start + length
    ].float().cpu()
    return full, block.logits[0].float().cpu(), expected


def partial_accept_logits(
    model,
    prefix: torch.Tensor,
    reply_ids: list[int],
    block_length: int = 9,
    accepted: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Use the state for an accepted q9 prefix, discarding its draft suffix."""
    length = min(block_length, len(reply_ids) - 2)
    if accepted >= length:
        raise ValueError((accepted, length))
    draft = reply_ids[:length]
    start = prefix.shape[1]
    _, parent = canon_prefill(model, prefix)
    block = canon_cached_step(
        model,
        input_ids=torch.tensor([draft], dtype=torch.long, device=model.device),
        position_ids=torch.arange(
            start, start + length, dtype=torch.long, device=model.device
        ).unsqueeze(0),
        cache=parent,
    )
    cache_length = start + accepted
    sliced_kv = []
    for layer in block.past_key_values.kv:
        values = list(layer)
        values[0] = values[0][..., :cache_length, :]
        values[1] = values[1][..., :cache_length, :]
        sliced_kv.append(tuple(values) if isinstance(layer, tuple) else values)
    child = CanonDFSCache(
        kv=tuple(sliced_kv),
        canon=block.canon_prefix_states[accepted - 1],
    )
    accepted_next = int(reply_ids[accepted])
    step = canon_cached_step(
        model,
        input_ids=torch.tensor(
            [[accepted_next]], dtype=torch.long, device=model.device
        ),
        position_ids=torch.tensor(
            [[start + accepted]], dtype=torch.long, device=model.device
        ),
        cache=child,
    )
    full_input = torch.cat(
        (
            prefix,
            torch.tensor(
                [reply_ids[: accepted + 1]], dtype=torch.long, device=model.device
            ),
        ),
        dim=1,
    )
    full = model(input_ids=full_input, return_dict=True, use_cache=False).logits[
        0, -1
    ].float().cpu()
    return (
        full.unsqueeze(0),
        step.logits[0, -1].float().cpu().unsqueeze(0),
        [int(reply_ids[accepted + 1])],
    )


def parity_metrics(
    full_next: torch.Tensor,
    cached_next: torch.Tensor,
    reply_ids: list[int],
    threshold: float,
) -> dict:
    if full_next.shape != cached_next.shape:
        raise RuntimeError((full_next.shape, cached_next.shape))
    legal = torch.tensor(ARC_TOKENS, dtype=torch.long)
    full_logp = full_next.log_softmax(-1)
    cached_logp = cached_next.log_softmax(-1)
    full_legal = full_logp[:, legal]
    cached_legal = cached_logp[:, legal]
    abs_error = (full_legal - cached_legal).abs()
    full_argmax = legal[full_legal.argmax(-1)]
    cached_argmax = legal[cached_legal.argmax(-1)]
    argmax_mismatch = full_argmax.ne(cached_argmax)
    cutoff = math.log(threshold)
    threshold_mismatch = full_legal.ge(cutoff).ne(cached_legal.ge(cutoff)).any(-1)
    gold = torch.tensor(reply_ids, dtype=torch.long)
    full_gold_p = full_logp.gather(1, gold[:, None]).exp().squeeze(1)
    cached_gold_p = cached_logp.gather(1, gold[:, None]).exp().squeeze(1)
    return {
        "max_abs_legal_logprob_error": float(abs_error.max()),
        "mean_abs_legal_logprob_error": float(abs_error.mean()),
        "legal_argmax_mismatches": int(argmax_mismatch.sum()),
        "first_legal_argmax_mismatch": (
            int(argmax_mismatch.nonzero()[0]) if argmax_mismatch.any() else None
        ),
        "threshold_membership_mismatches": int(threshold_mismatch.sum()),
        "first_threshold_membership_mismatch": (
            int(threshold_mismatch.nonzero()[0])
            if threshold_mismatch.any()
            else None
        ),
        "full_gold_geomean_probability": float(
            full_gold_p.clamp_min(1e-30).log().mean().exp()
        ),
        "cached_gold_geomean_probability": float(
            cached_gold_p.clamp_min(1e-30).log().mean().exp()
        ),
        "full_eos_probability_at_expected_end": float(full_logp[-1, EOS_ID].exp()),
        "cached_eos_probability_at_expected_end": float(
            cached_logp[-1, EOS_ID].exp()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--canon-state", required=True)
    parser.add_argument("--challenges-path", required=True)
    parser.add_argument("--solutions-path", required=True)
    parser.add_argument("--puzzle-key", default="0934a4d8")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--color-permutations", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=8192,
    )
    add_canon_ac_modules(model, kernel_size=4, zero_init=True)
    load_canon_state(model, args.canon_state)
    hooks = install_canon_ac_training_hooks(model)
    model = FastLanguageModel.for_inference(model)
    prepare_canon_inference(model)

    formatter = QwenFormatter(tokenizer)
    dataset = ArcDataset.from_file(
        args.challenges_path,
        keys=[args.puzzle_key],
    ).load_replies(args.solutions_path)
    eval_dataset = dataset.split_multi_replies().augment(
        n=args.color_permutations,
        seed=2,
    )
    rows = [eval_dataset.get(key, formatter) for key in eval_dataset.keys]
    row = max(rows, key=lambda item: len(tokenizer.encode(item["input"])))
    prefix_ids = tokenizer.encode(row["input"])
    reply_ids = tokenizer.encode(row["reply"])
    if not reply_ids or reply_ids[-1] != EOS_ID:
        raise RuntimeError(
            f"Expected an EOS-terminated gold reply, got tail={reply_ids[-8:]}"
        )

    device = model.device
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    reply = torch.tensor([reply_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        trained_full = full_next_logits(model, prefix, reply)
        trained_manual = manual_cached_next_logits(model, prefix, reply_ids)
        trained_metrics = parity_metrics(
            trained_full, trained_manual, reply_ids, args.threshold
        )
        sibling_full, sibling_cached = sibling_backtrack_logits(
            model, prefix, reply_ids
        )
        sibling_metrics = parity_metrics(
            sibling_full,
            sibling_cached,
            [int(reply_ids[1])],
            args.threshold,
        )
        block_full, block_cached, block_expected = multitoken_block_logits(
            model, prefix, reply_ids
        )
        block_metrics = parity_metrics(
            block_full, block_cached, block_expected, args.threshold
        )
        partial_full, partial_cached, partial_expected = partial_accept_logits(
            model, prefix, reply_ids
        )
        partial_metrics = parity_metrics(
            partial_full, partial_cached, partial_expected, args.threshold
        )

        for parameter in canon_parameters(model):
            parameter.zero_()

        zero_full = full_next_logits(model, prefix, reply)
        zero_manual = manual_cached_next_logits(model, prefix, reply_ids)
        zero_manual_metrics = parity_metrics(
            zero_full, zero_manual, reply_ids, args.threshold
        )
        zero_native = native_cached_next_logits(model, prefix, reply_ids)
        zero_native_metrics = parity_metrics(
            zero_full, zero_native, reply_ids, args.threshold
        )

    # Candidate equality is checked with trained Canon, so restore the sidecar
    # after the zero-Canon diagnostic above.
    load_canon_state(model, args.canon_state)
    with torch.inference_mode():
        one_token = inference_turbo_dfs(
            model,
            [prefix_ids],
            max_new_tokens=len(reply_ids),
            max_score=default_max_score(args.threshold),
            end_time=time.time() + 300,
        )
        q9_stats = {}
        q9 = inference_turbo_dfs_multitoken(
            model,
            [prefix_ids],
            max_new_tokens=len(reply_ids),
            max_score=default_max_score(args.threshold),
            end_time=time.time() + 300,
            repeat_len=9,
            stats=q9_stats,
        )

    def normalized(result):
        return {
            lane: [tuple(tokens) for _, tokens in beams]
            for lane, beams in result
        }

    one_token_candidates = normalized(one_token)
    q9_candidates = normalized(q9)

    summary = {
        "puzzle_key": args.puzzle_key,
        "subkey": str(row["key"]),
        "prefix_tokens": len(prefix_ids),
        "reply_tokens": len(reply_ids),
        "positions_compared": len(reply_ids),
        "expected_final_token": int(reply_ids[-1]),
        "trained_canon_manual_cache": trained_metrics,
        "trained_canon_sibling_backtrack": sibling_metrics,
        "trained_canon_q9_block": block_metrics,
        "trained_canon_q9_partial_accept": partial_metrics,
        "zero_canon_manual_cache": zero_manual_metrics,
        "zero_canon_native_unsloth_cache": zero_native_metrics,
        "q9_candidate_equality": one_token_candidates == q9_candidates,
        "one_token_candidates": sum(map(len, one_token_candidates.values())),
        "q9_candidates": sum(map(len, q9_candidates.values())),
        "q9_stats": q9_stats,
    }
    Path(args.output_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    hooks.remove()


if __name__ == "__main__":
    main()
