import time
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
        self.stats = {
            "score_calls": 0,
            "score_time_s": 0.0,
            "answers_scored": 0,
            "answer_tokens": 0,
            "query_tokens": 0,
        }

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

    def format_stats(self):
        stats = self.stats
        avg_answer_tokens = stats["answer_tokens"] / stats["answers_scored"] if stats["answers_scored"] else 0.0
        avg_query_tokens = stats["query_tokens"] / stats["answers_scored"] if stats["answers_scored"] else 0.0
        avg_score_ms = 1000.0 * stats["score_time_s"] / stats["score_calls"] if stats["score_calls"] else 0.0
        return (
            f"{self.__class__.__name__}(base_key={self.base_key}, "
            f"score_calls={stats['score_calls']}, "
            f"answers_scored={stats['answers_scored']}, "
            f"score_time_s={stats['score_time_s']:.3f}, "
            f"avg_score_ms={avg_score_ms:.1f}, "
            f"avg_query_tokens={avg_query_tokens:.1f}, "
            f"avg_answer_tokens={avg_answer_tokens:.1f})"
        )


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
        started_at = time.perf_counter()
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
        elapsed = time.perf_counter() - started_at
        self.stats["score_calls"] += 1
        self.stats["score_time_s"] += elapsed
        self.stats["answers_scored"] += len(answers)
        self.stats["query_tokens"] += sum(len(self.tokenizer.encode(query)) for query in queries)
        self.stats["answer_tokens"] += sum(len(self.tokenizer.encode(answer)) for answer in answers)
        return scores
