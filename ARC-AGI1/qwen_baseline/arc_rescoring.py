import copy
from dataclasses import dataclass

import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_search import PAD_ID


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded_tokens.append(tokens + [PAD_ID] * (max_len - len(tokens)))
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length - 1 : query_length - 1 + len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


@dataclass
class PrefixCacheEntry:
    key: str
    query_tokens: list[int]
    query_text: str
    prefix_next_logprobs: torch.Tensor
    past_key_values: object


@dataclass
class FullPassEntry:
    key: str
    query_text: str


class BaseRescorer:
    def __init__(self, model, tokenizer, formatter: QwenFormatter, puzzle_ds_multi: ArcDataset, base_key: str, max_seq_length: int, max_new_tokens: int, seed: int):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.puzzle_ds_multi = puzzle_ds_multi
        self.base_key = base_key
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def _build_template(self):
        template = ArcDataset(
            keys=[self.base_key],
            queries={self.base_key: self.puzzle_ds_multi.queries.get(self.base_key)},
            replies={},
        )
        template = template.augment(seed=self.seed)
        template = template.cut_to_len(
            formatter=self.formatter,
            name="input",
            max_len=self.max_seq_length - self.max_new_tokens,
        )
        return template


class FullPassRescorer(BaseRescorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = self._build_entries()

    def _build_entries(self):
        entries = []
        for sample in self._build_template().as_list(self.formatter):
            entries.append(
                FullPassEntry(
                    key=sample["key"],
                    query_text=sample["input"],
                )
            )
        return entries

    @torch.no_grad()
    def score_solution(self, solution):
        queries = []
        answers = []
        solution_list = solution.tolist()
        for entry in self.entries:
            augmented_solution = ArcDataset.forward_mod(solution_list, entry.key)
            queries.append(entry.query_text)
            answers.append(self.formatter.fmt_reply([augmented_solution]))
        scores = []
        for offset in range(0, len(queries), 4):
            scores.extend(calc_scores(queries[offset : offset + 4], answers[offset : offset + 4], tokenizer=self.tokenizer, model=self.model))
        return scores


class PrefixCachedRescorer(BaseRescorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = self._build_entries()

    def _build_entries(self):
        entries = []
        for sample in self._build_template().as_list(self.formatter):
            query_tokens = self.tokenizer.encode(sample["input"])
            input_ids = torch.tensor([query_tokens], device=self.model.device, dtype=torch.long)
            outputs = self.model(input_ids=input_ids, return_dict=True, use_cache=True)
            entries.append(
                PrefixCacheEntry(
                    key=sample["key"],
                    query_tokens=query_tokens,
                    query_text=sample["input"],
                    prefix_next_logprobs=outputs.logits[0, -1].float().cpu().log_softmax(-1),
                    past_key_values=outputs.past_key_values,
                )
            )
        return entries

    @torch.no_grad()
    def score_solution(self, solution):
        scores = []
        solution_list = solution.tolist()
        for entry in self.entries:
            augmented_solution = ArcDataset.forward_mod(solution_list, entry.key)
            answer_text = self.formatter.fmt_reply([augmented_solution])
            answer_tokens = self.tokenizer.encode(answer_text)
            scores.append(self._score_answer_tokens(entry, answer_tokens))
        return scores

    def _score_answer_tokens(self, entry: PrefixCacheEntry, answer_tokens: list[int]):
        if not answer_tokens:
            return 0.0

        total_logprob = entry.prefix_next_logprobs[answer_tokens[0]].item()
        if len(answer_tokens) == 1:
            return -total_logprob

        suffix_input = torch.tensor([answer_tokens[:-1]], device=self.model.device, dtype=torch.long)
        prefix_len = len(entry.query_tokens)
        position_ids = torch.arange(prefix_len, prefix_len + suffix_input.size(1), device=self.model.device, dtype=torch.long).unsqueeze(0)
        # Some Transformers cache implementations are mutable and get extended
        # even when we only want to score a fixed suffix. Clone the cached
        # prefix state so different candidates do not contaminate each other.
        past_key_values = copy.deepcopy(entry.past_key_values)
        outputs = self.model(
            input_ids=suffix_input,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=False,
        )
        suffix_logprobs = outputs.logits[0].float().cpu().log_softmax(-1)
        total_logprob += suffix_logprobs[torch.arange(len(answer_tokens) - 1), answer_tokens[1:]].sum().item()
        return -total_logprob
