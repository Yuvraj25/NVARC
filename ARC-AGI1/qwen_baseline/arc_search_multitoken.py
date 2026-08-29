"""Conservative cached multi-token acceleration for vanilla ARC DFS.

This module is deliberately not wired into ``arc_solver``.  It preserves the
recursive DFS order by accepting an extra repeated token only while every
active batch lane has exactly one viable non-EOS continuation and that
continuation is the repeated token.  Ambiguous frames are explored by the
ordinary recursive search logic.
"""

from __future__ import annotations

from collections import defaultdict
import time

import torch

from arc_search import ARC_TOKENS, EOS_ID, PAD_ID


def _position_ids(batch_size: int, start: int, length: int, device):
    values = torch.arange(start, start + length, device=device, dtype=torch.long)
    return values.unsqueeze(0).expand(batch_size, -1)


def _slice_cache(cache, length: int):
    """Return cache views ending at ``length`` without copying K/V tensors."""
    sliced = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError(f"Unsupported cache layer type: {type(layer)!r}")
        values = list(layer)
        values[0] = values[0][..., :length, :]
        values[1] = values[1][..., :length, :]
        sliced.append(tuple(values) if isinstance(layer, tuple) else values)
    return tuple(sliced) if isinstance(cache, tuple) else sliced


def _slice_canon_cache(cache, length: int, canon_state):
    from arc_canon import CanonDFSCache

    return CanonDFSCache(kv=_slice_cache(cache.kv, length), canon=canon_state)


def _classify_frame(logits, scores, remaining: int, max_score: float):
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    eos = defaultdict(list)
    candidates = {}
    for lane in range(n):
        candidates[lane] = []
        for token_id in ARC_TOKENS:
            score = nll[lane, token_id].item()
            if score >= max_score:
                continue
            if token_id == EOS_ID:
                eos[lane].append((score, [token_id]))
            elif remaining > 1:
                candidates[lane].append((score, token_id))
        candidates[lane].sort(key=lambda item: item[0])
    return eos, candidates


def _can_accept_repeat(logits, scores, remaining, repeat_tokens, active, max_score):
    if remaining <= 1:
        return None
    eos, candidates = _classify_frame(logits, scores, remaining, max_score)
    next_scores = list(scores)
    for lane, is_active in enumerate(active):
        if not is_active:
            continue
        lane_candidates = candidates[lane]
        if eos.get(lane) or len(lane_candidates) != 1:
            return None
        score, token_id = lane_candidates[0]
        if token_id != repeat_tokens[lane]:
            return None
        next_scores[lane] = score
    return next_scores


def _bump(stats, name, value=1):
    if stats is not None:
        stats[name] = int(stats.get(name, 0)) + int(value)


def turbo_dfs_multitoken(
    model,
    logits,
    max_new_tokens,
    max_score,
    scores,
    pos,
    cache,
    start_time,
    end_time,
    repeat_len=9,
    stats=None,
):
    """Run DFS with conservative repeated-token block verification."""
    n = logits.size(0)
    suffixes, candidates = _classify_frame(logits, scores, max_new_tokens, max_score)
    _bump(stats, "frames", sum(score < max_score for score in scores))

    while time.time() - start_time < 540 and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        active = []
        for lane in range(n):
            if candidates[lane]:
                score, token_id = candidates[lane].pop(0)
                batch_tokens.append(token_id)
                batch_scores.append(score)
                active.append(True)
            else:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000.0)
                active.append(False)
        if not any(active):
            break

        draft_len = min(max(1, repeat_len), max_new_tokens - 1)
        token_tensor = torch.tensor(batch_tokens, device=model.device, dtype=torch.long)
        if draft_len == 1:
            input_ids = token_tensor[:, None]
        else:
            input_ids = token_tensor[:, None].expand(-1, draft_len)

        call_started = time.perf_counter()
        position_ids = _position_ids(n, pos, draft_len, model.device)
        if hasattr(model, "_arc_canon_enabled"):
            from arc_canon import canon_cached_step

            outputs = canon_cached_step(
                model,
                input_ids=input_ids,
                position_ids=position_ids,
                cache=cache,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )
        _bump(stats, "model_calls")
        _bump(stats, "model_tokens", n * draft_len)
        if stats is not None:
            stats["model_time_s"] = stats.get("model_time_s", 0.0) + time.perf_counter() - call_started
        if draft_len > 1:
            _bump(stats, "block_calls")
            _bump(stats, "draft_tokens", n * draft_len)

        consumed = 1
        chain_scores = list(batch_scores)
        while consumed < draft_len:
            accepted_scores = _can_accept_repeat(
                outputs.logits[:, consumed - 1],
                chain_scores,
                max_new_tokens - consumed,
                batch_tokens,
                active,
                max_score,
            )
            if accepted_scores is None:
                break
            chain_scores = accepted_scores
            consumed += 1

        _bump(stats, "accepted_extra_tokens", sum(active) * (consumed - 1))
        if draft_len > 1 and consumed == 1:
            _bump(stats, "zero_extra_blocks")
        cache_len = pos + consumed
        if hasattr(model, "_arc_canon_enabled"):
            child_cache = _slice_canon_cache(
                outputs.past_key_values,
                cache_len,
                outputs.canon_prefix_states[consumed - 1],
            )
        else:
            child_cache = _slice_cache(outputs.past_key_values, cache_len)
        next_suffixes = turbo_dfs_multitoken(
            model=model,
            logits=outputs.logits[:, consumed - 1],
            max_new_tokens=max_new_tokens - consumed,
            max_score=max_score,
            scores=chain_scores,
            pos=pos + consumed,
            cache=child_cache,
            start_time=start_time,
            end_time=end_time,
            repeat_len=repeat_len,
            stats=stats,
        )

        for lane, beams in next_suffixes.items():
            prefix = [batch_tokens[lane]] * consumed
            for score, suffix_tokens in beams:
                suffixes[lane].append((score, prefix + suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs_multitoken(
    model,
    prefix_tokens,
    max_new_tokens,
    max_score,
    end_time,
    repeat_len=9,
    stats=None,
):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    started = time.time()
    if hasattr(model, "_arc_canon_enabled"):
        from arc_canon import canon_prefill

        outputs, initial_cache = canon_prefill(model, input_ids)
    else:
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        initial_cache = outputs.past_key_values
    _bump(stats, "prefill_calls")
    suffixes = turbo_dfs_multitoken(
        model=model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=initial_cache,
        start_time=started,
        end_time=end_time,
        repeat_len=repeat_len,
        stats=stats,
    )
    return [
        (lane, sorted(beams, key=lambda item: item[0]))
        for lane, beams in suffixes.items()
    ]
