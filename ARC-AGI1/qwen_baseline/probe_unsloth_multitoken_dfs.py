#!/usr/bin/env python3
"""Compare complete vanilla and conservative multi-token DFS candidate pools."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from arc_search import inference_turbo_dfs
from arc_search_multitoken import inference_turbo_dfs_multitoken
from patch_unsloth_qwen3_multitoken import patch_unsloth
from probe_unsloth_qwen3_multitoken import (
    ARC_TOKENS,
    build_prompt_tokens,
    load_model,
    position_ids,
    prefill,
    reset_fast_inference_buffers,
)


def candidate_map(result):
    mapped = {}
    for lane, beams in result:
        lane_map = {}
        for score, tokens in beams:
            lane_map.setdefault(tuple(tokens), []).append(float(score))
        mapped[lane] = {
            tokens: sorted(scores)
            for tokens, scores in lane_map.items()
        }
    return mapped


def compare_candidates(left, right):
    left_map = candidate_map(left)
    right_map = candidate_map(right)
    lanes = sorted(set(left_map) | set(right_map))
    missing = []
    extra = []
    score_diffs = []
    multiplicity_equal = True
    for lane in lanes:
        left_lane = left_map.get(lane, {})
        right_lane = right_map.get(lane, {})
        for tokens in sorted(set(left_lane) - set(right_lane)):
            missing.append({"lane": lane, "tokens": list(tokens)})
        for tokens in sorted(set(right_lane) - set(left_lane)):
            extra.append({"lane": lane, "tokens": list(tokens)})
        for tokens in set(left_lane) & set(right_lane):
            left_scores = left_lane[tokens]
            right_scores = right_lane[tokens]
            if len(left_scores) != len(right_scores):
                multiplicity_equal = False
                continue
            score_diffs.extend(abs(a - b) for a, b in zip(left_scores, right_scores))
    return {
        "candidate_tokens_equal": not missing and not extra and multiplicity_equal,
        "missing_candidate_count": len(missing),
        "extra_candidate_count": len(extra),
        "missing_examples": missing[:3],
        "extra_examples": extra[:3],
        "max_matching_score_diff": max(score_diffs, default=0.0),
    }


def candidate_count(result):
    return sum(len(beams) for _lane, beams in result)


@torch.inference_mode()
def warm_paths(model, prefix_tokens, repeat_len):
    reset_fast_inference_buffers(model)
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    prefix = prefill(model, input_ids)
    arc_ids = torch.tensor(ARC_TOKENS, device=model.device)
    token = arc_ids[prefix.logits[:, -1].index_select(-1, arc_ids).argmax(-1)]
    model(
        input_ids=token[:, None],
        position_ids=position_ids(input_ids.shape[0], input_ids.shape[1], 1, model.device),
        past_key_values=prefix.past_key_values,
        return_dict=True,
        use_cache=True,
    )
    reset_fast_inference_buffers(model)
    prefix = prefill(model, input_ids)
    model(
        input_ids=token[:, None].expand(-1, repeat_len),
        position_ids=position_ids(
            input_ids.shape[0], input_ids.shape[1], repeat_len, model.device
        ),
        past_key_values=prefix.past_key_values,
        return_dict=True,
        use_cache=True,
    )
    reset_fast_inference_buffers(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth-package-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--puzzle-key", default="136b0064")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prob-threshold", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repeat-len", type=int, default=9)
    parser.add_argument("--path-seconds", type=float, default=240.0)
    args = parser.parse_args()

    changed = patch_unsloth(Path(args.unsloth_package_dir))
    print("PATCHED", json.dumps([str(path) for path in changed]))
    model, tokenizer = load_model(args.model_path, args.adapter_path)
    subkeys, prefix_tokens = build_prompt_tokens(
        tokenizer, args.test_path, args.puzzle_key, None, args.batch_size
    )
    warm_paths(model, prefix_tokens, args.repeat_len)
    max_score = -math.log(args.prob_threshold)

    reset_fast_inference_buffers(model)
    baseline_started = time.perf_counter()
    baseline = inference_turbo_dfs(
        model,
        prefix_tokens,
        args.max_new_tokens,
        max_score,
        time.time() + args.path_seconds,
    )
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started
    baseline_timed_out = baseline_seconds >= args.path_seconds - 1.0

    reset_fast_inference_buffers(model)
    speculative_stats = {}
    speculative_started = time.perf_counter()
    speculative = inference_turbo_dfs_multitoken(
        model,
        prefix_tokens,
        args.max_new_tokens,
        max_score,
        time.time() + args.path_seconds,
        repeat_len=args.repeat_len,
        stats=speculative_stats,
    )
    torch.cuda.synchronize()
    speculative_seconds = time.perf_counter() - speculative_started
    speculative_timed_out = speculative_seconds >= args.path_seconds - 1.0

    comparison = compare_candidates(baseline, speculative)
    complete = not baseline_timed_out and not speculative_timed_out
    result = {
        "puzzle_key": args.puzzle_key,
        "subkeys": subkeys,
        "batch_size": args.batch_size,
        "prob_threshold": args.prob_threshold,
        "max_new_tokens": args.max_new_tokens,
        "repeat_len": args.repeat_len,
        "baseline_seconds": baseline_seconds,
        "speculative_seconds": speculative_seconds,
        "speedup": baseline_seconds / speculative_seconds,
        "baseline_timed_out": baseline_timed_out,
        "speculative_timed_out": speculative_timed_out,
        "baseline_candidates": candidate_count(baseline),
        "speculative_candidates": candidate_count(speculative),
        "speculative_stats": speculative_stats,
        "complete_candidate_gate": complete and comparison["candidate_tokens_equal"],
        **comparison,
    }
    print("MULTITOKEN_DFS_RESULT", json.dumps(result, sort_keys=True), flush=True)
    if not result["complete_candidate_gate"]:
        raise SystemExit("FAIL: complete multi-token DFS candidate parity gate")


if __name__ == "__main__":
    main()
