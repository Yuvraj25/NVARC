from collections import defaultdict
import os
import time

import numpy as np
import torch


ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15
ASSERT_IMMUTABLE_DFS_CACHE = os.environ.get("ARC_ASSERT_IMMUTABLE_DFS_CACHE") == "1"


def _iter_cache_tensors(cache):
    if cache is None:
        return
    if isinstance(cache, torch.Tensor):
        yield cache
        return
    if isinstance(cache, (tuple, list)):
        for item in cache:
            yield from _iter_cache_tensors(item)
        return
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            yield from _iter_cache_tensors(layer)
        return
    for attr in ("keys", "values", "key_cache", "value_cache"):
        if hasattr(cache, attr):
            yield from _iter_cache_tensors(getattr(cache, attr))


def _cache_fingerprint(cache):
    fp = []
    for tensor in _iter_cache_tensors(cache):
        if tensor is None:
            continue
        item = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
            tensor.data_ptr(),
        )
        if tensor.numel() > 0:
            sample = tensor.reshape(-1)[: min(8, tensor.numel())].detach().float().cpu()
            item = item + (tuple(sample.tolist()),)
        fp.append(item)
    return tuple(fp)


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)

    suffixes = defaultdict(list)
    candidates = {}

    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x: x[0])

    while time.time() - start_time < 540 and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        cache_fp = _cache_fingerprint(cache) if ASSERT_IMMUTABLE_DFS_CACHE else None

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        if ASSERT_IMMUTABLE_DFS_CACHE:
            updated_fp = _cache_fingerprint(cache)
            assert updated_fp == cache_fp, "DFS input cache mutated in-place across sibling branch expansion"

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens - 1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos + 1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        result.append((batch_id, sorted(beams, key=lambda x: x[0])))
    return result


def default_max_score(prob_threshold=0.2):
    return -np.log(prob_threshold)
