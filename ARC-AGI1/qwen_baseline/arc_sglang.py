import gc
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_rescoring import BaseRescorer, FullPassEntry
from arc_search import ARC_TOKENS, EOS_ID


@dataclass
class SglangConfig:
    model_path: str
    adapter_path: str
    tensor_parallel_size: int = 1
    mem_fraction_static: Optional[float] = None
    max_model_len: int = 8192
    lora_name: str = "arc_adapter"
    max_lora_rank: int = 256
    speculative_repeat_len: int = 4


def _as_batch(outputs):
    if isinstance(outputs, dict):
        return [outputs]
    return outputs


def _timed(label: str, fn):
    started_at = time.perf_counter()
    result = fn()
    print(f"[sglang] {label} took {time.perf_counter() - started_at:.2f}s")
    return result


def _bump_count(count_stats, key, amount=1):
    if count_stats is None:
        return
    count_stats[key] = count_stats.get(key, 0) + amount


def _patch_sglang_rslora(sglang):
    from sglang.srt.lora.lora import LoRAAdapter

    if getattr(LoRAAdapter, "_arc_rslora_patched", False):
        return

    original_init = LoRAAdapter.__init__

    def patched_init(self, uid, config, base_hf_config, load_config, lora_backend):
        original_init(self, uid, config, base_hf_config, load_config, lora_backend)
        if self.config.hf_config.get("use_rslora"):
            self.scaling = self.config.lora_alpha / math.sqrt(self.config.r)

    LoRAAdapter.__init__ = patched_init
    LoRAAdapter._arc_rslora_patched = True


def _iter_token_logprobs(row):
    if row is None:
        return
    if (
        len(row) in (2, 3)
        and isinstance(row[0], (list, tuple))
        and isinstance(row[1], (list, tuple))
        and all(isinstance(token_id, int) for token_id in row[1])
    ):
        for logprob, token_id in zip(row[0], row[1]):
            yield logprob, token_id
        return
    for item in row:
        logprob, token_id = item[:2]
        yield logprob, token_id


class ArcSglangBackend:
    def __init__(self, config: SglangConfig):
        self.config = config
        sglang = _timed("import", self._import_sglang)
        engine_kwargs = {
            "model_path": config.model_path,
            "skip_tokenizer_init": True,
            "trust_remote_code": True,
            "tp_size": config.tensor_parallel_size,
            "context_length": config.max_model_len,
            "enable_lora": True,
            "lora_paths": [f"{config.lora_name}={config.adapter_path}"],
            "max_lora_rank": config.max_lora_rank,
            "lora_target_modules": ["all"],
            "max_loras_per_batch": 1,
            "max_loaded_loras": 1,
        }
        if config.mem_fraction_static is not None:
            engine_kwargs["mem_fraction_static"] = config.mem_fraction_static
        self.engine = _timed("engine_init", lambda: sglang.Engine(**engine_kwargs))

    @staticmethod
    def _import_sglang():
        import sys

        arc_stack = "/kaggle/working/arc_stack"
        if arc_stack not in sys.path:
            sys.path.append(arc_stack)
        sglang = __import__("sglang")
        _patch_sglang_rslora(sglang)
        return sglang

    def close(self):
        if getattr(self, "engine", None) is not None:
            self.engine.shutdown()
            self.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def next_arc_logprobs(self, prefix_tokens: list[list[int]]) -> list[dict[int, float]]:
        if not prefix_tokens:
            return []
        outputs = self.engine.generate(
            input_ids=prefix_tokens,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=-1,
            token_ids_logprob=[ARC_TOKENS for _ in prefix_tokens],
            lora_path=[self.config.lora_name for _ in prefix_tokens],
        )
        result = []
        for output in _as_batch(outputs):
            meta = output.get("meta_info", {})
            rows = meta.get("output_token_ids_logprobs")
            if not rows or rows[0] is None:
                raise RuntimeError(f"SGLang did not return output_token_ids_logprobs; meta keys={sorted(meta.keys())}")
            result.append({int(token_id): float(logprob) for logprob, token_id in _iter_token_logprobs(rows[0])})
        return result

    def score_answers(self, query_tokens: list[list[int]], answer_tokens: list[list[int]]) -> list[float]:
        if len(query_tokens) != len(answer_tokens):
            raise ValueError("query_tokens and answer_tokens must have the same length")
        if not query_tokens:
            return []
        input_ids = [query + answer for query, answer in zip(query_tokens, answer_tokens)]
        logprob_start_lens = [max(len(query) - 1, 0) for query in query_tokens]
        outputs = self.engine.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=logprob_start_lens,
            token_ids_logprob=[ARC_TOKENS for _ in input_ids],
            lora_path=[self.config.lora_name for _ in input_ids],
        )
        scores = []
        for output, answer in zip(_as_batch(outputs), answer_tokens):
            meta = output.get("meta_info", {})
            token_id_logprobs = meta.get("input_token_ids_logprobs")
            if token_id_logprobs is None:
                raise RuntimeError(f"SGLang did not return input_token_ids_logprobs; meta keys={sorted(meta.keys())}")
            answer_rows = token_id_logprobs[1 : 1 + len(answer)] if answer else []
            if len(answer_rows) != len(answer):
                raise RuntimeError(f"SGLang returned {len(answer_rows)} answer logprobs for {len(answer)} answer tokens")
            total = 0.0
            for row, expected_token_id in zip(answer_rows, answer):
                logprobs = {int(token_id): float(logprob) for logprob, token_id in _iter_token_logprobs(row)}
                if expected_token_id not in logprobs:
                    raise RuntimeError(f"SGLang did not return logprob for ARC token {expected_token_id}")
                total += logprobs[expected_token_id]
            scores.append(-total)
        return scores


def inference_sglang_dfs(
    backend: ArcSglangBackend,
    prefix_tokens: list[list[int]],
    max_new_tokens: int,
    max_score: float,
    end_time: float,
):
    suffixes: list[list[tuple[float, list[int]]]] = [[] for _ in prefix_tokens]
    stack: list[tuple[int, list[int], float, int]] = [(i, [], 0.0, max_new_tokens) for i in range(len(prefix_tokens))]
    started_at = time.time()
    while stack and time.time() - started_at < 540 and time.time() < end_time:
        batch = stack[:64]
        del stack[:64]
        prompts = [prefix_tokens[batch_id] + suffix for batch_id, suffix, _score, _remaining in batch]
        logprob_rows = backend.next_arc_logprobs(prompts)
        for (batch_id, suffix, score, remaining), logprobs in zip(batch, logprob_rows):
            candidates = []
            for token_id in ARC_TOKENS:
                logprob = logprobs.get(token_id, float("-inf"))
                next_score = score - logprob
                if next_score >= max_score:
                    continue
                if token_id == EOS_ID:
                    suffixes[batch_id].append((next_score, suffix + [token_id]))
                elif remaining > 1:
                    candidates.append((next_score, token_id))
            for next_score, token_id in sorted(candidates, key=lambda item: item[0], reverse=True):
                stack.insert(0, (batch_id, suffix + [token_id], next_score, remaining - 1))
    return [(batch_id, sorted(beams, key=lambda item: item[0])) for batch_id, beams in enumerate(suffixes) if beams]


def _insert_frames(stack, frames):
    for batch_id, suffix, score, remaining in sorted(frames, key=lambda item: item[2], reverse=True):
        stack.insert(0, (batch_id, suffix, score, remaining))


def _branch_from_logprobs(
    batch_id: int,
    suffix: list[int],
    score: float,
    remaining: int,
    logprobs: dict[int, float],
    max_score: float,
):
    eos_beams = []
    candidates = []
    for token_id in ARC_TOKENS:
        logprob = logprobs.get(token_id, float("-inf"))
        next_score = score - logprob
        if next_score >= max_score:
            continue
        if token_id == EOS_ID:
            eos_beams.append((next_score, suffix + [token_id]))
        elif remaining > 1:
            candidates.append((next_score, token_id))
    return eos_beams, candidates


def _speculate_repeated_branch(
    backend: ArcSglangBackend,
    prefix_tokens: list[list[int]],
    batch_id: int,
    suffix: list[int],
    score: float,
    remaining: int,
    repeat_token_id: int,
    max_score: float,
    end_time: float,
    suffixes: list[list[tuple[float, list[int]]]],
    count_stats: dict[str, int] | None,
):
    _bump_count(count_stats, "spec_frames_started")

    if remaining <= 1 or backend.config.speculative_repeat_len <= 1:
        return [(batch_id, suffix, score, remaining)]

    active = [(batch_id, suffix, score, remaining, 1)]
    returned_frames = []
    while active and time.time() < end_time:
        prompts = [prefix_tokens[cur_batch_id] + cur_suffix for cur_batch_id, cur_suffix, _score, _remaining, _accepted in active]
        logprob_rows = backend.next_arc_logprobs(prompts)
        next_active = []
        for (cur_batch_id, cur_suffix, cur_score, cur_remaining, accepted_count), logprobs in zip(active, logprob_rows):
            eos_beams, candidates = _branch_from_logprobs(
                batch_id=cur_batch_id,
                suffix=cur_suffix,
                score=cur_score,
                remaining=cur_remaining,
                logprobs=logprobs,
                max_score=max_score,
            )
            if eos_beams:
                suffixes[cur_batch_id].extend(eos_beams)

            side_frames = []
            repeat_candidate = None
            for next_score, token_id in candidates:
                if token_id == repeat_token_id:
                    repeat_candidate = (next_score, token_id)
                else:
                    side_frames.append((cur_batch_id, cur_suffix + [token_id], next_score, cur_remaining - 1))
            _bump_count(count_stats, "spec_side_frames_enqueued", len(side_frames))
            returned_frames.extend(side_frames)

            if repeat_candidate is None:
                repeat_logprob = logprobs.get(repeat_token_id, float("-inf"))
                repeat_next_score = cur_score - repeat_logprob
                if repeat_next_score >= max_score:
                    _bump_count(count_stats, "spec_stop_threshold")
                else:
                    _bump_count(count_stats, "spec_stop_repeat_invalid")
                continue

            next_score, _ = repeat_candidate
            _bump_count(count_stats, "spec_tokens_attempted")
            _bump_count(count_stats, "spec_tokens_accepted")

            next_suffix = cur_suffix + [repeat_token_id]
            next_remaining = cur_remaining - 1
            next_accepted_count = accepted_count + 1

            if next_remaining <= 1:
                _bump_count(count_stats, "spec_stop_remaining_exhausted")
                returned_frames.append((cur_batch_id, next_suffix, next_score, next_remaining))
                continue

            if next_accepted_count >= backend.config.speculative_repeat_len:
                _bump_count(count_stats, "spec_stop_depth_limit")
                returned_frames.append((cur_batch_id, next_suffix, next_score, next_remaining))
                continue

            next_active.append((cur_batch_id, next_suffix, next_score, next_remaining, next_accepted_count))
        active = next_active

    if active:
        returned_frames.extend((cur_batch_id, cur_suffix, cur_score, cur_remaining) for cur_batch_id, cur_suffix, cur_score, cur_remaining, _accepted in active)
    return returned_frames


def inference_sglang_speculative_dfs(
    backend: ArcSglangBackend,
    prefix_tokens: list[list[int]],
    max_new_tokens: int,
    max_score: float,
    end_time: float,
    count_stats: dict[str, int] | None = None,
):
    suffixes: list[list[tuple[float, list[int]]]] = [[] for _ in prefix_tokens]
    stack: list[tuple[int, list[int], float, int]] = [(i, [], 0.0, max_new_tokens) for i in range(len(prefix_tokens))]
    started_at = time.time()
    while stack and time.time() - started_at < 540 and time.time() < end_time:
        batch = stack[:64]
        del stack[:64]
        prompts = [prefix_tokens[batch_id] + suffix for batch_id, suffix, _score, _remaining in batch]
        logprob_rows = backend.next_arc_logprobs(prompts)
        new_frames = []
        for (batch_id, suffix, score, remaining), logprobs in zip(batch, logprob_rows):
            eos_beams, candidates = _branch_from_logprobs(
                batch_id=batch_id,
                suffix=suffix,
                score=score,
                remaining=remaining,
                logprobs=logprobs,
                max_score=max_score,
            )
            if eos_beams:
                suffixes[batch_id].extend(eos_beams)
                _bump_count(count_stats, "spec_stop_eos", len(eos_beams))
            for next_score, token_id in candidates:
                branch_suffix = suffix + [token_id]
                branch_remaining = remaining - 1
                branch_frames = _speculate_repeated_branch(
                    backend=backend,
                    prefix_tokens=prefix_tokens,
                    batch_id=batch_id,
                    suffix=branch_suffix,
                    score=next_score,
                    remaining=branch_remaining,
                    repeat_token_id=token_id,
                    max_score=max_score,
                    end_time=end_time,
                    suffixes=suffixes,
                    count_stats=count_stats,
                )
                new_frames.extend(branch_frames)
        _insert_frames(stack, new_frames)
    return [(batch_id, sorted(beams, key=lambda item: item[0])) for batch_id, beams in enumerate(suffixes) if beams]


class SglangRescorer(BaseRescorer):
    def __init__(self, *args, backend: ArcSglangBackend, **kwargs):
        self.backend = backend
        super().__init__(*args, **kwargs)
        self.entries = self._build_entries()

    def _build_entries(self):
        entries = []
        for sample in self._build_template().as_list(self.formatter):
            entries.append(FullPassEntry(key=sample["key"], query_text=sample["input"]))
        return entries

    def score_solution(self, solution):
        started_at = time.perf_counter()
        query_tokens = []
        answer_tokens = []
        solution_list = solution.tolist()
        for entry in self.entries:
            augmented_solution = ArcDataset.forward_mod(solution_list, entry.key)
            answer_text = self.formatter.fmt_reply([augmented_solution])
            query_tokens.append(self.tokenizer.encode(entry.query_text))
            answer_tokens.append(self.tokenizer.encode(answer_text))
        scores = []
        for offset in range(0, len(query_tokens), 4):
            scores.extend(self.backend.score_answers(query_tokens[offset : offset + 4], answer_tokens[offset : offset + 4]))
        elapsed = time.perf_counter() - started_at
        self.stats["score_calls"] += 1
        self.stats["score_time_s"] += elapsed
        self.stats["answers_scored"] += len(answer_tokens)
        self.stats["query_tokens"] += sum(len(tokens) for tokens in query_tokens)
        self.stats["answer_tokens"] += sum(len(tokens) for tokens in answer_tokens)
        return scores
