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
    adapter_path: Optional[str] = None
    tensor_parallel_size: int = 1
    mem_fraction_static: Optional[float] = None
    max_model_len: int = 8192
    lora_name: str = "arc_adapter"
    max_lora_rank: int = 256
    max_loaded_loras: int = 1
    speculative_repeat_len: int = 5
    dynamic_repeat_enabled: bool = False
    dynamic_repeat_gap_low: float = 1.5
    dynamic_repeat_gap_mid: float = 3.0
    dynamic_repeat_gap_high: float = 5.0


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
        self.current_lora_name: Optional[str] = config.lora_name if config.adapter_path else None
        self.stats = {
            "adapter_load_calls": 0,
            "adapter_load_time_s": 0.0,
            "adapter_unload_calls": 0,
            "adapter_unload_time_s": 0.0,
            "next_arc_logprobs_calls": 0,
            "next_arc_logprobs_prompts": 0,
            "next_arc_logprobs_prompt_tokens": 0,
            "next_arc_logprobs_max_batch": 0,
            "next_arc_logprobs_time_s": 0.0,
            "draft_arc_logprobs_calls": 0,
            "draft_arc_logprobs_prompts": 0,
            "draft_arc_logprobs_prompt_tokens": 0,
            "draft_arc_logprobs_draft_tokens": 0,
            "draft_arc_logprobs_max_batch": 0,
            "draft_arc_logprobs_time_s": 0.0,
            "score_arc_logprobs_calls": 0,
            "score_arc_logprobs_prompts": 0,
            "score_arc_logprobs_prompt_tokens": 0,
            "score_arc_logprobs_answer_tokens": 0,
            "score_arc_logprobs_max_batch": 0,
            "score_arc_logprobs_time_s": 0.0,
        }
        sglang = _timed("import", self._import_sglang)
        engine_kwargs = {
            "model_path": config.model_path,
            "skip_tokenizer_init": True,
            "trust_remote_code": True,
            "tp_size": config.tensor_parallel_size,
            "context_length": config.max_model_len,
            "enable_lora": True,
            "max_lora_rank": config.max_lora_rank,
            "lora_target_modules": ["all"],
            "max_loras_per_batch": 1,
            "max_loaded_loras": config.max_loaded_loras,
        }
        if config.adapter_path:
            engine_kwargs["lora_paths"] = [f"{config.lora_name}={config.adapter_path}"]
        if config.mem_fraction_static is not None:
            engine_kwargs["mem_fraction_static"] = config.mem_fraction_static
        started_at = time.perf_counter()
        self.engine = _timed("engine_init", lambda: sglang.Engine(**engine_kwargs))
        self.engine_init_s = time.perf_counter() - started_at

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

    def reset_stats(self):
        for key, value in self.stats.items():
            self.stats[key] = 0.0 if isinstance(value, float) else 0

    def _require_active_lora(self) -> str:
        if self.current_lora_name is None:
            raise RuntimeError("No active SGLang LoRA adapter is loaded")
        return self.current_lora_name

    @staticmethod
    def _raise_for_lora_result(action: str, lora_name: str, result):
        success = getattr(result, "success", None)
        if success is False:
            raise RuntimeError(f"SGLang {action} failed for LoRA {lora_name}: {getattr(result, 'error_message', None)}")

    def load_adapter(self, lora_name: str, adapter_path: str, pinned: bool = False):
        if self.current_lora_name == lora_name and self.config.adapter_path == adapter_path:
            return
        if self.current_lora_name is not None:
            self.unload_adapter()
        started_at = time.perf_counter()
        result = self.engine.load_lora_adapter(lora_name=lora_name, lora_path=adapter_path, pinned=pinned)
        self.stats["adapter_load_calls"] += 1
        self.stats["adapter_load_time_s"] += time.perf_counter() - started_at
        self._raise_for_lora_result("load", lora_name, result)
        self.current_lora_name = lora_name
        self.config.adapter_path = adapter_path
        self.config.lora_name = lora_name

    def unload_adapter(self):
        if self.current_lora_name is None:
            return
        lora_name = self.current_lora_name
        started_at = time.perf_counter()
        result = self.engine.unload_lora_adapter(lora_name=lora_name)
        self.stats["adapter_unload_calls"] += 1
        self.stats["adapter_unload_time_s"] += time.perf_counter() - started_at
        self._raise_for_lora_result("unload", lora_name, result)
        self.current_lora_name = None
        self.config.adapter_path = None

    def next_arc_logprobs(self, prefix_tokens: list[list[int]]) -> list[dict[int, float]]:
        if not prefix_tokens:
            return []
        active_lora = self._require_active_lora()
        started_at = time.perf_counter()
        outputs = self.engine.generate(
            input_ids=prefix_tokens,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=-1,
            token_ids_logprob=[ARC_TOKENS for _ in prefix_tokens],
            lora_path=[active_lora for _ in prefix_tokens],
        )
        elapsed = time.perf_counter() - started_at
        self.stats["next_arc_logprobs_calls"] += 1
        self.stats["next_arc_logprobs_prompts"] += len(prefix_tokens)
        self.stats["next_arc_logprobs_prompt_tokens"] += sum(len(tokens) for tokens in prefix_tokens)
        self.stats["next_arc_logprobs_max_batch"] = max(self.stats["next_arc_logprobs_max_batch"], len(prefix_tokens))
        self.stats["next_arc_logprobs_time_s"] += elapsed
        result = []
        for output in _as_batch(outputs):
            meta = output.get("meta_info", {})
            rows = meta.get("output_token_ids_logprobs")
            if not rows or rows[0] is None:
                raise RuntimeError(f"SGLang did not return output_token_ids_logprobs; meta keys={sorted(meta.keys())}")
            result.append({int(token_id): float(logprob) for logprob, token_id in _iter_token_logprobs(rows[0])})
        return result

    def _input_arc_logprobs(
        self,
        query_tokens: list[list[int]],
        answer_tokens: list[list[int]],
        stats_prefix: str,
    ) -> list[list[dict[int, float]]]:
        if len(query_tokens) != len(answer_tokens):
            raise ValueError("query_tokens and answer_tokens must have the same length")
        if not query_tokens:
            return []
        active_lora = self._require_active_lora()
        input_ids = [query + answer for query, answer in zip(query_tokens, answer_tokens)]
        logprob_start_lens = [max(len(query) - 1, 0) for query in query_tokens]
        started_at = time.perf_counter()
        outputs = self.engine.generate(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 0},
            return_logprob=True,
            logprob_start_len=logprob_start_lens,
            token_ids_logprob=[ARC_TOKENS for _ in input_ids],
            lora_path=[active_lora for _ in input_ids],
        )
        elapsed = time.perf_counter() - started_at
        self.stats[f"{stats_prefix}_calls"] += 1
        self.stats[f"{stats_prefix}_prompts"] += len(query_tokens)
        self.stats[f"{stats_prefix}_prompt_tokens"] += sum(len(tokens) for tokens in query_tokens)
        token_key = "draft_tokens" if stats_prefix == "draft_arc_logprobs" else "answer_tokens"
        self.stats[f"{stats_prefix}_{token_key}"] += sum(len(tokens) for tokens in answer_tokens)
        self.stats[f"{stats_prefix}_max_batch"] = max(self.stats[f"{stats_prefix}_max_batch"], len(query_tokens))
        self.stats[f"{stats_prefix}_time_s"] += elapsed
        result = []
        for output, answer in zip(_as_batch(outputs), answer_tokens):
            meta = output.get("meta_info", {})
            token_id_logprobs = meta.get("input_token_ids_logprobs")
            if token_id_logprobs is None:
                raise RuntimeError(f"SGLang did not return input_token_ids_logprobs; meta keys={sorted(meta.keys())}")
            answer_rows = token_id_logprobs[1 : 1 + len(answer)] if answer else []
            if len(answer_rows) != len(answer):
                raise RuntimeError(f"SGLang returned {len(answer_rows)} answer logprobs for {len(answer)} answer tokens")
            result.append(
                [
                    {int(token_id): float(logprob) for logprob, token_id in _iter_token_logprobs(row)}
                    for row in answer_rows
                ]
            )
        return result

    def draft_arc_logprobs(
        self,
        query_tokens: list[list[int]],
        answer_tokens: list[list[int]],
    ) -> list[list[dict[int, float]]]:
        return self._input_arc_logprobs(query_tokens, answer_tokens, stats_prefix="draft_arc_logprobs")

    def score_answers(self, query_tokens: list[list[int]], answer_tokens: list[list[int]]) -> list[float]:
        rows_by_answer = self._input_arc_logprobs(query_tokens, answer_tokens, stats_prefix="score_arc_logprobs")
        scores = []
        for answer_rows, answer in zip(rows_by_answer, answer_tokens):
            total = 0.0
            for logprobs, expected_token_id in zip(answer_rows, answer):
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
    count_stats: dict[str, int] | None = None,
):
    suffixes: list[list[tuple[float, list[int]]]] = [[] for _ in prefix_tokens]
    stack: list[tuple[int, list[int], float, int]] = [(i, [], 0.0, max_new_tokens) for i in range(len(prefix_tokens))]
    started_at = time.time()
    while stack and time.time() - started_at < 540 and time.time() < end_time:
        batch = stack[:64]
        del stack[:64]
        _bump_count(count_stats, "dfs_frames_expanded", len(batch))
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


def _record_spec_branch_result(count_stats: dict[str, int] | None, extra_appends: int):
    _bump_count(count_stats, "spec_extra_appends_total", extra_appends)
    bucket_keys = {
        0: "spec_branches_zero_extra",
        1: "spec_branches_one_extra",
        2: "spec_branches_two_extra",
        3: "spec_branches_three_extra",
        4: "spec_branches_four_extra",
    }
    _bump_count(count_stats, bucket_keys.get(extra_appends, "spec_branches_five_plus_extra"))


def _dynamic_repeat_len_from_gap(
    logprobs: dict[int, float],
    repeat_token_id: int,
    branch_remaining: int,
    config: SglangConfig,
    count_stats: dict[str, int] | None,
) -> int:
    max_repeat_len = max(1, min(config.speculative_repeat_len, branch_remaining))
    if not config.dynamic_repeat_enabled or max_repeat_len <= 1:
        chosen = max_repeat_len
    else:
        repeat_logprob = logprobs.get(repeat_token_id, float("-inf"))
        best_other = max(
            (logprobs.get(token_id, float("-inf")) for token_id in ARC_TOKENS if token_id != repeat_token_id),
            default=float("-inf"),
        )
        gap = repeat_logprob - best_other
        if gap < config.dynamic_repeat_gap_low:
            chosen = 1
        elif gap < config.dynamic_repeat_gap_mid:
            chosen = min(5, max_repeat_len)
        elif gap < config.dynamic_repeat_gap_high:
            chosen = min(9, max_repeat_len)
        else:
            chosen = max_repeat_len
        _bump_count(count_stats, "spec_dynamic_gap_lt_low" if gap < config.dynamic_repeat_gap_low else "spec_dynamic_gap_ge_low")
        if gap < config.dynamic_repeat_gap_mid:
            _bump_count(count_stats, "spec_dynamic_gap_lt_mid")
        else:
            _bump_count(count_stats, "spec_dynamic_gap_ge_mid")
        if gap < config.dynamic_repeat_gap_high:
            _bump_count(count_stats, "spec_dynamic_gap_lt_high")
        else:
            _bump_count(count_stats, "spec_dynamic_gap_ge_high")

    if chosen <= 1:
        _bump_count(count_stats, "spec_dynamic_len_1")
    elif chosen <= 5:
        _bump_count(count_stats, "spec_dynamic_len_5")
    elif chosen <= 9:
        _bump_count(count_stats, "spec_dynamic_len_9")
    else:
        _bump_count(count_stats, "spec_dynamic_len_max")
    return chosen


def _advance_speculative_pool(
    backend: ArcSglangBackend,
    prefix_tokens: list[list[int]],
    active: list[tuple[int, list[int], float, int, int, int]],
    max_score: float,
    end_time: float,
    suffixes: list[list[tuple[float, list[int]]]],
    count_stats: dict[str, int] | None,
):
    returned_frames = []
    while active and time.time() < end_time:
        spec_batch = active[:64]
        del active[:64]
        verify_batch = []
        verify_queries = []
        verify_answers = []
        for cur_batch_id, cur_suffix, cur_score, cur_remaining, repeat_token_id, max_repeat_len in spec_batch:
            max_extra_appends = min(max_repeat_len - 1, cur_remaining - 1)
            if max_extra_appends <= 0:
                _record_spec_branch_result(count_stats, 0)
                returned_frames.append((cur_batch_id, cur_suffix, cur_score, cur_remaining))
                continue
            verify_batch.append((cur_batch_id, cur_suffix, cur_score, cur_remaining, repeat_token_id, max_extra_appends))
            verify_queries.append(prefix_tokens[cur_batch_id] + cur_suffix)
            verify_answers.append([repeat_token_id] * max_extra_appends)

        verified_rows = backend.draft_arc_logprobs(verify_queries, verify_answers) if verify_batch else []
        for (
            cur_batch_id,
            cur_suffix,
            cur_score,
            cur_remaining,
            repeat_token_id,
            max_extra_appends,
        ), draft_rows in zip(verify_batch, verified_rows):
            extra_appends = 0
            terminated = False
            for logprobs in draft_rows:
                _bump_count(count_stats, "dfs_frames_expanded")
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
                    _bump_count(count_stats, "spec_stop_eos", len(eos_beams))

                side_frames = []
                repeat_candidate = None
                for next_score, token_id in candidates:
                    if token_id == repeat_token_id:
                        repeat_candidate = next_score
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
                    _record_spec_branch_result(count_stats, extra_appends)
                    terminated = True
                    break

                cur_suffix = cur_suffix + [repeat_token_id]
                cur_score = repeat_candidate
                cur_remaining -= 1
                extra_appends += 1

                if cur_remaining <= 1:
                    _bump_count(count_stats, "spec_stop_remaining_exhausted")
                    _record_spec_branch_result(count_stats, extra_appends)
                    returned_frames.append((cur_batch_id, cur_suffix, cur_score, cur_remaining))
                    terminated = True
                    break

            if terminated:
                continue

            if extra_appends >= max_extra_appends:
                _bump_count(count_stats, "spec_stop_depth_limit")
            _record_spec_branch_result(count_stats, extra_appends)
            returned_frames.append((cur_batch_id, cur_suffix, cur_score, cur_remaining))

    if active:
        _bump_count(count_stats, "spec_stop_end_time", len(active))
        for cur_batch_id, cur_suffix, cur_score, cur_remaining, _repeat_token_id, _max_repeat_len in active:
            _record_spec_branch_result(count_stats, 0)
            returned_frames.append((cur_batch_id, cur_suffix, cur_score, cur_remaining))
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
        _bump_count(count_stats, "dfs_frames_expanded", len(batch))
        prompts = [prefix_tokens[batch_id] + suffix for batch_id, suffix, _score, _remaining in batch]
        logprob_rows = backend.next_arc_logprobs(prompts)
        new_frames = []
        spec_active = []
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
                _bump_count(count_stats, "spec_branches_started")
                max_repeat_len = _dynamic_repeat_len_from_gap(
                    logprobs=logprobs,
                    repeat_token_id=token_id,
                    branch_remaining=branch_remaining,
                    config=backend.config,
                    count_stats=count_stats,
                )
                if branch_remaining <= 1 or max_repeat_len <= 1:
                    _record_spec_branch_result(count_stats, 0)
                    new_frames.append((batch_id, branch_suffix, next_score, branch_remaining))
                    continue
                spec_active.append((batch_id, branch_suffix, next_score, branch_remaining, token_id, max_repeat_len))
        new_frames.extend(
            _advance_speculative_pool(
                backend=backend,
                prefix_tokens=prefix_tokens,
                active=spec_active,
                max_score=max_score,
                end_time=end_time,
                suffixes=suffixes,
                count_stats=count_stats,
            )
        )
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
