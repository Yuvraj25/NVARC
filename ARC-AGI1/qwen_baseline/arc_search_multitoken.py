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


NEWLINE_ID = 10
MAX_GRID_SIDE = 30


class _GridState:
    """Branch-local structural state for one generated ARC grid."""

    __slots__ = ("width", "column", "completed_rows")

    def __init__(self, width=None, column=0, completed_rows=0):
        self.width = width
        self.column = column
        self.completed_rows = completed_rows

    def advance(self, token_id):
        if 0 <= token_id <= 9:
            return _GridState(self.width, self.column + 1, self.completed_rows)
        if token_id == NEWLINE_ID:
            width = self.column if self.width is None else self.width
            return _GridState(width, 0, self.completed_rows + 1)
        if token_id == EOS_ID:
            return self
        raise ValueError(f"Not an ARC grid token: {token_id}")


def _token_is_structurally_legal(state, token_id):
    """Whether appending ``token_id`` can still form a <=30x30 rectangle."""
    if 0 <= token_id <= 9:
        width_limit = MAX_GRID_SIDE if state.width is None else state.width
        return state.completed_rows < MAX_GRID_SIDE and state.column < width_limit
    row_complete = (
        state.column > 0
        and state.column <= MAX_GRID_SIDE
        and (state.width is None or state.column == state.width)
    )
    if token_id == NEWLINE_ID:
        return row_complete and state.completed_rows < MAX_GRID_SIDE - 1
    if token_id == EOS_ID:
        return row_complete
    return False


def _draft_len_for_token(state, token_id, repeat_len, remaining):
    if not 0 <= token_id <= 9:
        return 1
    width_limit = MAX_GRID_SIDE if state.width is None else state.width
    cells_left = width_limit - state.column
    return min(max(1, repeat_len), cells_left, remaining - 1)


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
    from arc_canon import CanonACState, CanonDFSCache

    # q-block prefix states are views into a shared rolling-state tensor.
    # Clone only the chosen prefix so recursive DFS does not retain the other
    # q-1 state snapshots after the full block output is released.
    selected_state = CanonACState(
        a=tuple(tensor.clone() for tensor in canon_state.a),
        c=tuple(tensor.clone() for tensor in canon_state.c),
    )
    return CanonDFSCache(kv=_slice_cache(cache.kv, length), canon=selected_state)


def _classify_frame(
    logits,
    scores,
    remaining: int,
    max_score: float,
    *,
    generated_prefixes=None,
    frontier=None,
    frontier_max_score=None,
    root_lanes=None,
):
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    eos = defaultdict(list)
    candidates = {}
    for lane in range(n):
        candidates[lane] = []
        for token_id in ARC_TOKENS:
            score = nll[lane, token_id].item()
            if score >= max_score:
                can_finish = token_id == EOS_ID
                can_continue = token_id != EOS_ID and remaining > 1
                if (
                    frontier is not None
                    and frontier_max_score is not None
                    and score < frontier_max_score
                    and (can_finish or can_continue)
                    and scores[lane] < max_score
                ):
                    frontier.append(
                        {
                            "lane": lane if root_lanes is None else root_lanes[lane],
                            "score": score,
                            "tokens": list(generated_prefixes[lane]) + [token_id],
                            "finished": can_finish,
                        }
                    )
                continue
            if token_id == EOS_ID:
                eos[lane].append((score, [token_id]))
            elif remaining > 1:
                candidates[lane].append((score, token_id))
        candidates[lane].sort(key=lambda item: item[0])
    return eos, candidates


def _classify_structured_frame(logits, scores, remaining, max_score, states):
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    eos = defaultdict(list)
    candidates = {}
    for lane in range(n):
        candidates[lane] = []
        for token_id in ARC_TOKENS:
            if not _token_is_structurally_legal(states[lane], token_id):
                continue
            score = nll[lane, token_id].item()
            if score >= max_score:
                continue
            if token_id == EOS_ID:
                eos[lane].append((score, [token_id]))
            elif remaining > 1:
                candidates[lane].append((score, token_id))
        candidates[lane].sort(key=lambda item: item[0])
    return eos, candidates


def _can_accept_repeat(
    logits,
    scores,
    remaining,
    repeat_tokens,
    active,
    max_score,
    *,
    generated_prefixes=None,
    frontier=None,
    frontier_max_score=None,
    root_lanes=None,
):
    if remaining <= 1:
        return None
    eos, candidates = _classify_frame(
        logits,
        scores,
        remaining,
        max_score,
        generated_prefixes=generated_prefixes,
        frontier=frontier,
        frontier_max_score=frontier_max_score,
        root_lanes=root_lanes,
    )
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


def _can_accept_structured_repeat(
    logits,
    scores,
    remaining,
    repeat_tokens,
    states,
    max_score,
):
    if remaining <= 1:
        return None
    eos, candidates = _classify_structured_frame(
        logits, scores, remaining, max_score, states
    )
    next_scores = list(scores)
    next_states = list(states)
    for lane in range(len(states)):
        lane_candidates = candidates[lane]
        if eos.get(lane) or len(lane_candidates) != 1:
            return None
        score, token_id = lane_candidates[0]
        if token_id != repeat_tokens[lane]:
            return None
        next_scores[lane] = score
        next_states[lane] = states[lane].advance(token_id)
    return next_scores, next_states


def _bump(stats, name, value=1):
    if stats is not None:
        stats[name] = int(stats.get(name, 0)) + int(value)


def _reset_unsloth_paged_attention(model):
    """Force the next cached step to import the newly prefetched KV cache."""
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
        if hasattr(module, "paged_attention"):
            for name in ("paged_attention_K", "paged_attention_V", "paged_attention"):
                if hasattr(module, name):
                    delattr(module, name)


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
    generated_prefixes=None,
    frontier=None,
    frontier_max_score=None,
    root_lanes=None,
):
    """Run DFS with conservative repeated-token block verification."""
    n = logits.size(0)
    if generated_prefixes is None:
        generated_prefixes = [[] for _ in range(n)]
    if root_lanes is None:
        root_lanes = list(range(n))
    suffixes, candidates = _classify_frame(
        logits,
        scores,
        max_new_tokens,
        max_score,
        generated_prefixes=generated_prefixes,
        frontier=frontier,
        frontier_max_score=frontier_max_score,
        root_lanes=root_lanes,
    )
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
            intermediate_prefixes = [
                list(generated_prefixes[lane])
                + ([batch_tokens[lane]] * consumed if active[lane] else [])
                for lane in range(n)
            ]
            accepted_scores = _can_accept_repeat(
                outputs.logits[:, consumed - 1],
                chain_scores,
                max_new_tokens - consumed,
                batch_tokens,
                active,
                max_score,
                generated_prefixes=intermediate_prefixes,
                frontier=frontier,
                frontier_max_score=frontier_max_score,
                root_lanes=root_lanes,
            )
            if accepted_scores is None:
                break
            chain_scores = accepted_scores
            consumed += 1

        _bump(stats, "accepted_extra_tokens", sum(active) * (consumed - 1))
        if draft_len > 1 and consumed == 1:
            _bump(stats, "zero_extra_blocks")
        cache_len = pos + consumed
        # A depth-first call can remain on the Python stack for hundreds of
        # output tokens.  Do not let that stack retain the complete q-token
        # result (full-vocabulary logits plus every Canon prefix snapshot) at
        # each depth.  Materialize only the accepted position required by the
        # child before releasing the block result.
        child_logits = outputs.logits[:, consumed - 1].clone()
        if hasattr(model, "_arc_canon_enabled"):
            child_cache = _slice_canon_cache(
                outputs.past_key_values,
                cache_len,
                outputs.canon_prefix_states[consumed - 1],
            )
        else:
            child_cache = _slice_cache(outputs.past_key_values, cache_len)
        del outputs
        next_suffixes = turbo_dfs_multitoken(
            model=model,
            logits=child_logits,
            max_new_tokens=max_new_tokens - consumed,
            max_score=max_score,
            scores=chain_scores,
            pos=pos + consumed,
            cache=child_cache,
            start_time=start_time,
            end_time=end_time,
            repeat_len=repeat_len,
            stats=stats,
            generated_prefixes=[
                list(generated_prefixes[lane])
                + ([batch_tokens[lane]] * consumed if active[lane] else [])
                for lane in range(n)
            ],
            frontier=frontier,
            frontier_max_score=frontier_max_score,
            root_lanes=root_lanes,
        )

        for lane, beams in next_suffixes.items():
            prefix = [batch_tokens[lane]] * consumed
            for score, suffix_tokens in beams:
                suffixes[lane].append((score, prefix + suffix_tokens))

    return suffixes


def turbo_dfs_multitoken_structured(
    model,
    logits,
    max_new_tokens,
    max_score,
    scores,
    pos,
    cache,
    states,
    start_time,
    end_time,
    repeat_len=9,
    stats=None,
    physical_batch_size=None,
    physical_lanes=None,
):
    """Rectangular-grid DFS with active lanes bucketed by safe draft length."""
    if hasattr(model, "_arc_canon_enabled"):
        raise NotImplementedError(
            "Structured length-bucketed DFS is currently a Vanilla-only path"
        )

    n = logits.size(0)
    if physical_batch_size is None:
        physical_batch_size = n
    if physical_lanes is None:
        physical_lanes = list(range(n))
    if n > physical_batch_size or len(physical_lanes) != n:
        raise ValueError(
            f"Invalid logical/physical lanes: logical={n} "
            f"mapped={len(physical_lanes)} physical={physical_batch_size}"
        )
    suffixes, candidates = _classify_structured_frame(
        logits, scores, max_new_tokens, max_score, states
    )
    _bump(stats, "structured_frames", n)

    while time.time() - start_time < 540 and time.time() < end_time:
        picked = []
        for lane in range(n):
            if candidates[lane]:
                score, token_id = candidates[lane].pop(0)
                picked.append((lane, score, token_id))
        if not picked:
            break

        buckets = defaultdict(list)
        for lane, score, token_id in picked:
            draft_len = _draft_len_for_token(
                states[lane], token_id, repeat_len, max_new_tokens
            )
            buckets[draft_len].append((lane, score, token_id))

        for draft_len in sorted(buckets, reverse=True):
            group = buckets[draft_len]
            logical_group_size = len(group)
            logical_lanes = [item[0] for item in group]
            target_physical_lanes = [
                physical_lanes[lane] for lane in logical_lanes
            ]
            batch_scores = [item[1] for item in group]
            target_tokens = [item[2] for item in group]
            chain_states = [
                states[lane].advance(token_id)
                for lane, _score, token_id in group
            ]
            # Unsloth's paged KV buffers retain the original prefill batch and
            # are mutated in place. Never reorder or copy their lane axis.
            # Run every bucket at that physical batch size, write disposable
            # filler trajectories into non-target lanes, and read only the
            # target physical lanes. Sibling calls overwrite positions after
            # this frame's prefix, so parent-prefix backtracking stays intact.
            batch_tokens = [0] * physical_batch_size
            for physical_lane, token_id in zip(
                target_physical_lanes, target_tokens
            ):
                batch_tokens[physical_lane] = token_id
            token_tensor = torch.tensor(
                batch_tokens, device=model.device, dtype=torch.long
            )
            input_ids = token_tensor[:, None].expand(-1, draft_len)
            position_ids = _position_ids(
                physical_batch_size, pos, draft_len, model.device
            )

            call_started = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )
            _bump(stats, "model_calls")
            _bump(stats, "model_tokens", physical_batch_size * draft_len)
            _bump(stats, "useful_model_tokens", logical_group_size * draft_len)
            _bump(
                stats,
                "padded_model_tokens",
                (physical_batch_size - logical_group_size) * draft_len,
            )
            _bump(stats, "length_bucket_calls")
            _bump(stats, f"q{draft_len}_calls")
            if stats is not None:
                stats["model_time_s"] = stats.get("model_time_s", 0.0) + time.perf_counter() - call_started
            if draft_len > 1:
                _bump(stats, "block_calls")
                _bump(stats, "draft_tokens", physical_batch_size * draft_len)

            consumed = 1
            chain_scores = list(batch_scores)
            while consumed < draft_len:
                accepted = _can_accept_structured_repeat(
                    outputs.logits[
                        target_physical_lanes, consumed - 1
                    ],
                    chain_scores,
                    max_new_tokens - consumed,
                    target_tokens,
                    chain_states,
                    max_score,
                )
                if accepted is None:
                    break
                chain_scores, chain_states = accepted
                consumed += 1

            _bump(
                stats,
                "accepted_extra_tokens",
                logical_group_size * (consumed - 1),
            )
            if draft_len > 1 and consumed == 1:
                _bump(stats, "zero_extra_blocks")

            child_logits = outputs.logits[
                target_physical_lanes, consumed - 1
            ].clone()
            child_cache = _slice_cache(
                outputs.past_key_values, pos + consumed
            )
            del outputs
            next_suffixes = turbo_dfs_multitoken_structured(
                model=model,
                logits=child_logits,
                max_new_tokens=max_new_tokens - consumed,
                max_score=max_score,
                scores=chain_scores[:logical_group_size],
                pos=pos + consumed,
                cache=child_cache,
                states=chain_states[:logical_group_size],
                start_time=start_time,
                end_time=end_time,
                repeat_len=repeat_len,
                stats=stats,
                physical_batch_size=physical_batch_size,
                physical_lanes=target_physical_lanes,
            )

            for local_lane, beams in next_suffixes.items():
                parent_lane = logical_lanes[local_lane]
                prefix = [target_tokens[local_lane]] * consumed
                for score, suffix_tokens in beams:
                    suffixes[parent_lane].append((score, prefix + suffix_tokens))

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
    structured_rows=False,
    frontier_max_score=None,
    return_frontier=False,
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
    common = dict(
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
    if structured_rows and frontier_max_score is not None:
        raise NotImplementedError(
            "Threshold-frontier capture is currently implemented for the "
            "production unstructured q9 path only"
        )
    frontier = [] if frontier_max_score is not None else None
    if structured_rows:
        suffixes = turbo_dfs_multitoken_structured(
            states=[_GridState() for _ in range(input_ids.size(0))],
            physical_batch_size=input_ids.size(0),
            **common,
        )
    else:
        suffixes = turbo_dfs_multitoken(
            frontier=frontier,
            frontier_max_score=frontier_max_score,
            **common,
        )
    result = [
        (lane, sorted(beams, key=lambda item: item[0]))
        for lane, beams in suffixes.items()
    ]
    if return_frontier:
        return result, sorted(
            frontier or [],
            key=lambda item: (item["lane"], item["score"], item["tokens"]),
        )
    return result


@torch.no_grad()
def resume_turbo_dfs_multitoken(
    model,
    prefix_tokens,
    frontier,
    max_new_tokens,
    max_score,
    end_time,
    repeat_len=9,
    stats=None,
):
    """Resume only branches captured at a stricter threshold boundary.

    KV tensors are deliberately not retained in ``frontier``: doing so would
    keep an entire cache per rejected sibling.  Entries of equal generated
    length are instead replayed together at the original physical batch size.
    """
    if hasattr(model, "_arc_canon_enabled"):
        raise NotImplementedError("Frontier resume is currently Vanilla-only")
    physical_batch_size = len(prefix_tokens)
    grouped = defaultdict(list)
    completed = defaultdict(list)
    for entry in frontier:
        lane = int(entry["lane"])
        tokens = list(entry["tokens"])
        if entry.get("finished", False):
            completed[lane].append((float(entry["score"]), tokens))
        else:
            grouped[len(tokens)].append(entry)

    started = time.time()
    for generated_length in sorted(grouped):
        entries = grouped[generated_length]
        for offset in range(0, len(entries), physical_batch_size):
            if time.time() >= end_time or time.time() - started >= 540:
                break
            active_entries = entries[offset : offset + physical_batch_size]
            padded_entries = list(active_entries)
            while len(padded_entries) < physical_batch_size:
                padded_entries.append(active_entries[0])

            replay_tokens = [
                list(prefix_tokens[int(entry["lane"])]) + list(entry["tokens"])
                for entry in padded_entries
            ]
            _reset_unsloth_paged_attention(model)
            input_ids = torch.tensor(
                replay_tokens, device=model.device, dtype=torch.long
            )
            call_started = time.perf_counter()
            outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
            if stats is not None:
                stats["resume_prefill_time_s"] = (
                    stats.get("resume_prefill_time_s", 0.0)
                    + time.perf_counter()
                    - call_started
                )
            _bump(stats, "resume_prefill_calls")
            _bump(stats, "resume_prefill_tokens", input_ids.numel())

            scores = [float(entry["score"]) for entry in active_entries]
            scores.extend([1000.0] * (physical_batch_size - len(scores)))
            suffixes = turbo_dfs_multitoken(
                model=model,
                logits=outputs.logits[:, -1],
                max_new_tokens=max_new_tokens - generated_length,
                max_score=max_score,
                scores=scores,
                pos=input_ids.size(1),
                cache=outputs.past_key_values,
                start_time=started,
                end_time=end_time,
                repeat_len=repeat_len,
                stats=stats,
            )
            for local_lane, beams in suffixes.items():
                if local_lane >= len(active_entries):
                    continue
                entry = active_entries[local_lane]
                root_lane = int(entry["lane"])
                entry_tokens = list(entry["tokens"])
                for score, suffix_tokens in beams:
                    completed[root_lane].append(
                        (score, entry_tokens + suffix_tokens)
                    )

    return [
        (lane, sorted(beams, key=lambda item: item[0]))
        for lane, beams in completed.items()
    ]
