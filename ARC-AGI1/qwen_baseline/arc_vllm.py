import gc
import inspect
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_rescoring import BaseRescorer
from arc_search import ARC_TOKENS, EOS_ID


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(name, fn):
    _sync_cuda()
    started_at = time.perf_counter()
    result = fn()
    _sync_cuda()
    elapsed = time.perf_counter() - started_at
    print(f"[vllm_bench] {name}_s={elapsed:.3f}", flush=True)
    return result


def _gpu_mem(label):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[vllm_bench] {label}: alloc_gb={alloc:.2f} reserved_gb={reserved:.2f} peak_gb={peak:.2f}", flush=True)


def _import_vllm():
    def do_import():
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        return LLM, SamplingParams, LoRARequest

    return _timed("import_vllm", do_import)


def _make_prompt(token_ids):
    return {"prompt_token_ids": list(token_ids)}


def _logprob_value(entry):
    return entry.logprob if hasattr(entry, "logprob") else entry[0]


def _get_logprob(logprobs_by_token, token_id):
    if logprobs_by_token is None or token_id not in logprobs_by_token:
        raise KeyError(f"vLLM did not return logprob for required token id {token_id}")
    return float(_logprob_value(logprobs_by_token[token_id]))


@dataclass
class VllmConfig:
    model_path: str
    adapter_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    max_model_len: int = 8192


class ArcVllmBackend:
    def __init__(self, config: VllmConfig):
        self.config = config
        LLM, SamplingParams, LoRARequest = _import_vllm()
        self.SamplingParams = SamplingParams
        self.llm = _timed(
            "engine_init",
            lambda: LLM(
                model=config.model_path,
                enable_lora=True,
                enable_prefix_caching=config.enable_prefix_caching,
                trust_remote_code=True,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                skip_tokenizer_init=True,
            ),
        )
        _gpu_mem("after_engine_init")
        self.lora_request = LoRARequest(
            lora_name=os.path.basename(config.adapter_path.rstrip(os.sep)) or "arc_adapter",
            lora_int_id=1,
            lora_path=config.adapter_path,
        )
        sampling_params_fields = inspect.signature(SamplingParams).parameters
        self._uses_logprob_token_ids = "logprob_token_ids" in sampling_params_fields
        dfs_params = dict(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(ARC_TOKENS) if self._uses_logprob_token_ids else -1,
        )
        if self._uses_logprob_token_ids:
            dfs_params["logprob_token_ids"] = ARC_TOKENS
        self._dfs_params = SamplingParams(**dfs_params)
        self._score_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=-1,
        )
        print(
            "[vllm_bench] "
            f"prefix_caching_enabled={config.enable_prefix_caching} "
            f"tensor_parallel_size={config.tensor_parallel_size} "
            f"gpu_memory_utilization={config.gpu_memory_utilization:.2f} "
            f"max_model_len={config.max_model_len} "
            f"dfs_logprob_mode={'arc_tokens' if self._uses_logprob_token_ids else 'full_vocab'}",
            flush=True,
        )
        self._lora_warmed = False

    def close(self):
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def next_arc_logprobs(self, prefixes):
        if not prefixes:
            return []
        def run_generate():
            return self.llm.generate(
                [_make_prompt(prefix) for prefix in prefixes],
                sampling_params=self._dfs_params,
                lora_request=self.lora_request,
                use_tqdm=False,
            )

        if self._lora_warmed:
            outputs = run_generate()
        else:
            outputs = _timed("lora_first_use", run_generate)
            self._lora_warmed = True
        result = []
        for output in outputs:
            token_logprobs = output.outputs[0].logprobs[0]
            result.append({token_id: _get_logprob(token_logprobs, token_id) for token_id in ARC_TOKENS})
        return result

    def score_tokenized_answers(self, query_tokens_list, answer_tokens_list):
        prompts = []
        full_tokens_list = []
        for query_tokens, answer_tokens in zip(query_tokens_list, answer_tokens_list):
            full_tokens = list(query_tokens) + list(answer_tokens)
            prompts.append(_make_prompt(full_tokens))
            full_tokens_list.append(full_tokens)

        outputs = self.llm.generate(
            prompts,
            sampling_params=self._score_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )

        scores = []
        for output, query_tokens, answer_tokens, full_tokens in zip(outputs, query_tokens_list, answer_tokens_list, full_tokens_list):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("vLLM did not return prompt_logprobs for answer rescoring")
            total_logprob = 0.0
            start = len(query_tokens)
            for pos, token_id in enumerate(answer_tokens, start=start):
                if pos >= len(prompt_logprobs):
                    raise IndexError(
                        f"Missing prompt logprob at answer position {pos}; "
                        f"prompt_logprobs_len={len(prompt_logprobs)} full_tokens_len={len(full_tokens)}"
                    )
                total_logprob += _get_logprob(prompt_logprobs[pos], token_id)
            scores.append(-total_logprob)
        return scores


def inference_vllm_dfs(backend: ArcVllmBackend, prefix_tokens, max_new_tokens, max_score, end_time):
    start_time = time.time()
    prefixes = [list(tokens) for tokens in prefix_tokens]
    states = [(batch_id, prefix, 0.0, []) for batch_id, prefix in enumerate(prefixes)]
    suffixes = defaultdict(list)

    for _pos in range(max_new_tokens):
        if not states or time.time() - start_time >= 540 or time.time() >= end_time:
            break

        logprob_rows = backend.next_arc_logprobs([prefix for _batch_id, prefix, _score, _suffix in states])
        next_states = []
        for (batch_id, prefix, parent_score, suffix), token_logprobs in zip(states, logprob_rows):
            for token_id in ARC_TOKENS:
                child_score = parent_score - token_logprobs[token_id]
                if child_score >= max_score:
                    continue
                child_suffix = suffix + [token_id]
                if token_id == EOS_ID:
                    suffixes[batch_id].append((child_score, child_suffix))
                else:
                    next_states.append((batch_id, prefix + [token_id], child_score, child_suffix))
        states = sorted(next_states, key=lambda item: item[2])

    result = []
    for batch_id, beams in suffixes.items():
        result.append((batch_id, sorted(beams, key=lambda x: x[0])))
    return result


class VllmRescorer(BaseRescorer):
    def __init__(
        self,
        backend: ArcVllmBackend,
        tokenizer,
        formatter: QwenFormatter,
        puzzle_ds_multi: ArcDataset,
        base_key: str,
        max_seq_length: int,
        max_new_tokens: int,
        seed: int,
    ):
        super().__init__(
            model=None,
            tokenizer=tokenizer,
            formatter=formatter,
            puzzle_ds_multi=puzzle_ds_multi,
            base_key=base_key,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        self.backend = backend
        self.entries = self._build_entries()

    def _build_entries(self):
        entries = []
        for sample in self._build_template().as_list(self.formatter):
            entries.append((sample["key"], sample["input"], self.tokenizer.encode(sample["input"])))
        return entries

    def score_solution(self, solution):
        started_at = time.perf_counter()
        query_tokens_list = []
        answer_tokens_list = []
        solution_list = solution.tolist()
        for key, _query_text, query_tokens in self.entries:
            augmented_solution = ArcDataset.forward_mod(solution_list, key)
            answer_text = self.formatter.fmt_reply([augmented_solution])
            query_tokens_list.append(query_tokens)
            answer_tokens_list.append(self.tokenizer.encode(answer_text))
        scores = self.backend.score_tokenized_answers(query_tokens_list, answer_tokens_list)
        elapsed = time.perf_counter() - started_at
        self.stats["score_calls"] += 1
        self.stats["score_time_s"] += elapsed
        self.stats["answers_scored"] += len(answer_tokens_list)
        self.stats["query_tokens"] += sum(len(tokens) for tokens in query_tokens_list)
        self.stats["answer_tokens"] += sum(len(tokens) for tokens in answer_tokens_list)
        return scores
