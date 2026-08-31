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


def _select_cache_lanes(cache, lanes):
    """Copy selected batch lanes from the tuple cache used by this DFS path."""
    selected = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError(f"Unsupported cache layer type: {type(layer)!r}")
        values = list(layer)
        for index in (0, 1):
            lane_index = torch.tensor(lanes, device=values[index].device)
            values[index] = values[index].index_select(0, lane_index)
        selected.append(tuple(values) if isinstance(layer, tuple) else values)
    return tuple(selected) if isinstance(cache, tuple) else selected


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
):
    """Rectangular-grid DFS with active lanes bucketed by safe draft length."""
    if hasattr(model, "_arc_canon_enabled"):
        raise NotImplementedError(
            "Structured length-bucketed DFS is currently a Vanilla-only path"
        )

    n = logits.size(0)
    if physical_batch_size is None:
        physical_batch_size = n
    if n > physical_batch_size:
        raise ValueError(
            f"Logical batch {n} exceeds physical batch {physical_batch_size}"
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
            # Unsloth's paged-attention scratch buffers are allocated at the
            # prefill batch size and cannot change batch dimension mid-DFS.
            # Duplicate one real lane to fill unused physical slots, then
            # discard those duplicate outputs before recursing.
            padded_group = group + [group[-1]] * (
                physical_batch_size - logical_group_size
            )
            lanes = [item[0] for item in padded_group]
            batch_scores = [item[1] for item in padded_group]
            batch_tokens = [item[2] for item in padded_group]
            chain_states = [
                states[lane].advance(token_id)
                for lane, _score, token_id in padded_group
            ]
            token_tensor = torch.tensor(
                batch_tokens, device=model.device, dtype=torch.long
            )
            input_ids = token_tensor[:, None].expand(-1, draft_len)
            position_ids = _position_ids(
                physical_batch_size, pos, draft_len, model.device
            )
            group_cache = _select_cache_lanes(cache, lanes)

            call_started = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=group_cache,
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
                    outputs.logits[:, consumed - 1],
                    chain_scores,
                    max_new_tokens - consumed,
                    batch_tokens,
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
                :logical_group_size, consumed - 1
            ].clone()
            child_cache = _select_cache_lanes(
                _slice_cache(outputs.past_key_values, pos + consumed),
                list(range(logical_group_size)),
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
            )

            for local_lane, beams in next_suffixes.items():
                parent_lane = lanes[local_lane]
                prefix = [batch_tokens[local_lane]] * consumed
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
    if structured_rows:
        suffixes = turbo_dfs_multitoken_structured(
            states=[_GridState() for _ in range(input_ids.size(0))],
            physical_batch_size=input_ids.size(0),
            **common,
        )
    else:
        suffixes = turbo_dfs_multitoken(**common)
    return [
        (lane, sorted(beams, key=lambda item: item[0]))
        for lane, beams in suffixes.items()
    ]
