#!/usr/bin/env python3
"""Shadow cached multi-token decisions against real q_len=1 DFS frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Optional

import torch

from patch_unsloth_qwen3_multitoken import patch_unsloth
from probe_unsloth_qwen3_multitoken import (
    ARC_TOKENS,
    build_prompt_tokens,
    load_model,
    position_ids,
    prefill,
    reset_fast_inference_buffers,
)


EOS_ID = 15
PAD_ID = 13


@dataclass
class TraceState:
    max_frames: int
    deadline: float
    frames: list[dict] = field(default_factory=list)
    completed: list[list[tuple[float, list[int]]]] = field(default_factory=list)

    @property
    def stopped(self) -> bool:
        return len(self.frames) >= self.max_frames or time.time() >= self.deadline


def arc_logprobs(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().log_softmax(-1)[:, ARC_TOKENS].cpu()


@torch.inference_mode()
def trace_turbo_dfs(
    model,
    logits,
    remaining: int,
    max_score: float,
    scores: list[float],
    suffixes: list[list[int]],
    position: int,
    cache,
    state: TraceState,
):
    if state.stopped:
        return
    rows = arc_logprobs(logits)
    lane_candidates = []
    for lane, (score, suffix, row) in enumerate(zip(scores, suffixes, rows)):
        candidates = []
        if score < max_score:
            row_dict = {token_id: float(row[index]) for index, token_id in enumerate(ARC_TOKENS)}
            state.frames.append(
                {
                    "lane": lane,
                    "suffix": list(suffix),
                    "score": float(score),
                    "remaining": remaining,
                    "arc_logprobs": row_dict,
                }
            )
            for token_id in ARC_TOKENS:
                next_score = score - row_dict[token_id]
                if next_score >= max_score:
                    continue
                if token_id == EOS_ID:
                    state.completed[lane].append((next_score, suffix + [token_id]))
                elif remaining > 1:
                    candidates.append((next_score, token_id))
        lane_candidates.append(sorted(candidates, key=lambda item: item[0]))
        if state.stopped:
            return

    while not state.stopped:
        batch_tokens = []
        batch_scores = []
        batch_suffixes = []
        alive = 0
        for suffix, candidates in zip(suffixes, lane_candidates):
            if candidates:
                score, token_id = candidates.pop(0)
                batch_tokens.append(token_id)
                batch_scores.append(score)
                batch_suffixes.append(suffix + [token_id])
                alive += 1
            else:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000.0)
                batch_suffixes.append(list(suffix))
        if alive == 0:
            return
        token_tensor = torch.tensor(batch_tokens, device=model.device, dtype=torch.long)[:, None]
        output = model(
            input_ids=token_tensor,
            position_ids=torch.full(
                (len(batch_tokens), 1), position, device=model.device, dtype=torch.long
            ),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        trace_turbo_dfs(
            model=model,
            logits=output.logits[:, -1],
            remaining=remaining - 1,
            max_score=max_score,
            scores=batch_scores,
            suffixes=batch_suffixes,
            position=position + 1,
            cache=output.past_key_values,
            state=state,
        )


@torch.inference_mode()
def collect_real_frames(model, prefix_tokens, max_new_tokens, max_score, max_frames, seconds):
    reset_fast_inference_buffers(model)
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    initial = model(input_ids=input_ids, return_dict=True, use_cache=True)
    state = TraceState(
        max_frames=max_frames,
        deadline=time.time() + seconds,
        completed=[[] for _ in prefix_tokens],
    )
    trace_turbo_dfs(
        model=model,
        logits=initial.logits[:, -1],
        remaining=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * len(prefix_tokens),
        suffixes=[[] for _ in prefix_tokens],
        position=input_ids.shape[1],
        cache=initial.past_key_values,
        state=state,
    )
    return state


def candidate_branches(frames, max_score):
    branches = []
    for frame_index, frame in enumerate(frames):
        if frame["remaining"] <= 2:
            continue
        for token_id in ARC_TOKENS:
            if token_id == EOS_ID:
                continue
            next_score = frame["score"] - frame["arc_logprobs"][token_id]
            if next_score >= max_score:
                continue
            branches.append(
                {
                    "frame_index": frame_index,
                    "lane": frame["lane"],
                    "branch_suffix": frame["suffix"] + [token_id],
                    "repeat_token": token_id,
                    "score": next_score,
                    "remaining": frame["remaining"] - 1,
                    "initial_margin": max_score - next_score,
                }
            )
    return sorted(branches, key=lambda branch: branch["initial_margin"])


@torch.inference_mode()
def replay_parent(model, prefix_tokens, suffix):
    reset_fast_inference_buffers(model)
    device = model.device
    prefix_ids = torch.tensor([prefix_tokens], device=device, dtype=torch.long)
    output = prefill(model, prefix_ids)
    cache = output.past_key_values
    logits = output.logits[:, -1]
    position = prefix_ids.shape[1]
    for token_id in suffix:
        token = torch.tensor([[token_id]], device=device, dtype=torch.long)
        output = model(
            input_ids=token,
            position_ids=position_ids(1, position, 1, device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1]
        position += 1
    return logits, cache, position


@torch.inference_mode()
def sequential_repeat_rows(model, prefix_tokens, suffix, repeat_token, count):
    logits, cache, position = replay_parent(model, prefix_tokens, suffix)
    rows = []
    for _ in range(count):
        rows.append(arc_logprobs(logits)[0])
        token = torch.tensor([[repeat_token]], device=model.device, dtype=torch.long)
        output = model(
            input_ids=token,
            position_ids=position_ids(1, position, 1, model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        logits = output.logits[:, -1]
        cache = output.past_key_values
        position += 1
    return torch.stack(rows)


@torch.inference_mode()
def block_repeat_rows(model, prefix_tokens, suffix, repeat_token, count):
    logits, cache, position = replay_parent(model, prefix_tokens, suffix)
    rows = [arc_logprobs(logits)[0]]
    if count > 1:
        draft = torch.full(
            (1, count - 1), repeat_token, device=model.device, dtype=torch.long
        )
        output = model(
            input_ids=draft,
            position_ids=position_ids(1, position, count - 1, model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        rows.extend(arc_logprobs(output.logits[0]).unbind(0))
    return torch.stack(rows)


def compare_branch(model, prefix_tokens, branch, max_score, repeat_len):
    extra_count = min(repeat_len - 1, branch["remaining"] - 1)
    if extra_count <= 0:
        return None
    sequential = sequential_repeat_rows(
        model,
        prefix_tokens,
        branch["branch_suffix"],
        branch["repeat_token"],
        extra_count,
    )
    block = block_repeat_rows(
        model,
        prefix_tokens,
        branch["branch_suffix"],
        branch["repeat_token"],
        extra_count,
    )
    seq_score = float(branch["score"])
    block_score = float(branch["score"])
    max_diff = 0.0
    min_seq_margin = float("inf")
    first_divergence: Optional[dict] = None
    compared_steps = 0
    for step, (seq_row, block_row) in enumerate(zip(sequential, block)):
        compared_steps += 1
        max_diff = max(max_diff, float((seq_row - block_row).abs().max()))
        seq_next = {
            token_id: seq_score - float(seq_row[index])
            for index, token_id in enumerate(ARC_TOKENS)
        }
        block_next = {
            token_id: block_score - float(block_row[index])
            for index, token_id in enumerate(ARC_TOKENS)
        }
        seq_retained = {token_id for token_id, score in seq_next.items() if score < max_score}
        block_retained = {token_id for token_id, score in block_next.items() if score < max_score}
        min_seq_margin = min(
            min_seq_margin,
            min((abs(max_score - score) for score in seq_next.values()), default=float("inf")),
        )
        if seq_retained != block_retained:
            first_divergence = {
                "step": step,
                "seq_score": seq_score,
                "block_score": block_score,
                "seq_only": sorted(seq_retained - block_retained),
                "block_only": sorted(block_retained - seq_retained),
                "seq_next": seq_next,
                "block_next": block_next,
            }
            break
        repeat_token = branch["repeat_token"]
        if repeat_token not in seq_retained:
            break
        seq_score = seq_next[repeat_token]
        block_score = block_next[repeat_token]
    return {
        **branch,
        "extra_count": extra_count,
        "compared_steps": compared_steps,
        "max_arc_logprob_diff": max_diff,
        "min_seq_boundary_margin": min_seq_margin,
        "final_seq_score": seq_score,
        "final_block_score": block_score,
        "first_divergence": first_divergence,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth-package-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--puzzle-key", default="136b0064")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prob-threshold", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=930)
    parser.add_argument("--max-trace-frames", type=int, default=128)
    parser.add_argument("--trace-seconds", type=float, default=45.0)
    parser.add_argument("--max-branch-comparisons", type=int, default=32)
    parser.add_argument("--repeat-len", type=int, default=9)
    args = parser.parse_args()

    changed = patch_unsloth(Path(args.unsloth_package_dir))
    print("PATCHED", json.dumps([str(path) for path in changed]))
    model, tokenizer = load_model(args.model_path, args.adapter_path)
    subkeys, prefix_tokens = build_prompt_tokens(
        tokenizer,
        args.test_path,
        args.puzzle_key,
        None,
        args.batch_size,
    )
    max_score = -math.log(args.prob_threshold)
    started = time.perf_counter()
    state = collect_real_frames(
        model,
        prefix_tokens,
        args.max_new_tokens,
        max_score,
        args.max_trace_frames,
        args.trace_seconds,
    )
    branches = candidate_branches(state.frames, max_score)
    selected = branches[: args.max_branch_comparisons]
    comparisons = []
    for index, branch in enumerate(selected):
        result = compare_branch(
            model,
            prefix_tokens[branch["lane"]],
            branch,
            max_score,
            args.repeat_len,
        )
        if result is not None:
            comparisons.append(result)
        print(
            "SHADOW_PROGRESS",
            json.dumps(
                {
                    "comparison": index + 1,
                    "selected": len(selected),
                    "divergences": sum(row["first_divergence"] is not None for row in comparisons),
                }
            ),
            flush=True,
        )
        if result is not None and result["first_divergence"] is not None:
            break

    divergences = [row for row in comparisons if row["first_divergence"] is not None]
    result = {
        "puzzle_key": args.puzzle_key,
        "subkeys": subkeys,
        "adapter_path": args.adapter_path,
        "prob_threshold": args.prob_threshold,
        "max_score": max_score,
        "trace_frames": len(state.frames),
        "completed_candidates_during_trace": sum(len(rows) for rows in state.completed),
        "candidate_branches": len(branches),
        "branches_compared": len(comparisons),
        "repeat_steps_compared": sum(row["compared_steps"] for row in comparisons),
        "minimum_tested_boundary_margin": min(
            (row["min_seq_boundary_margin"] for row in comparisons),
            default=None,
        ),
        "maximum_arc_logprob_diff": max(
            (row["max_arc_logprob_diff"] for row in comparisons),
            default=0.0,
        ),
        "decision_divergences": len(divergences),
        "first_divergence": divergences[0] if divergences else None,
        "wall_seconds": time.perf_counter() - started,
        "shadow_decision_pass": not divergences and bool(comparisons),
    }
    print("SHADOW_DFS_RESULT", json.dumps(result, sort_keys=True), flush=True)
    if not result["shadow_decision_pass"]:
        raise SystemExit("FAIL: shadow DFS retain/prune decisions diverged or no branches were compared")


if __name__ == "__main__":
    main()
