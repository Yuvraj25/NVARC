import time
import unittest
from types import SimpleNamespace

import torch

from arc_search import inference_turbo_dfs
from arc_search_multitoken import inference_turbo_dfs_multitoken


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, prompt_len, targets, ambiguous_positions=()):
        self.prompt_len = prompt_len
        self.targets = targets
        self.ambiguous_positions = set(ambiguous_positions)
        self.q_lens = []

    def _row(self, generated):
        row = torch.full((16,), -20.0)
        target = self.targets[min(generated, len(self.targets) - 1)]
        row[target] = 8.0
        if generated in self.ambiguous_positions:
            row[3 if target != 3 else 4] = 7.5
        return row

    def __call__(self, input_ids, past_key_values=None, **_kwargs):
        batch, q_len = input_ids.shape
        self.q_lens.append(q_len)
        previous = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        logits = torch.empty((batch, q_len, 16))
        for offset in range(q_len):
            generated = max(previous + offset + 1 - self.prompt_len, 0)
            logits[:, offset] = self._row(generated)
        total = previous + q_len
        cache_tensor = torch.zeros((batch, 1, total, 1))
        return SimpleNamespace(
            logits=logits,
            past_key_values=[(cache_tensor, cache_tensor.clone())],
        )


def token_results(result):
    return {
        lane: [tokens for _score, tokens in beams]
        for lane, beams in result
    }


class MultiTokenDfsTest(unittest.TestCase):
    def compare_paths(self, ambiguous_positions=()):
        prefix = [[11, 12], [11, 12]]
        targets = [4, 4, 4, 10, 2, 2, 15]
        baseline_model = FakeModel(len(prefix[0]), targets, ambiguous_positions)
        speculative_model = FakeModel(len(prefix[0]), targets, ambiguous_positions)
        deadline = time.time() + 30
        baseline = inference_turbo_dfs(
            baseline_model, prefix, max_new_tokens=12, max_score=2.302585, end_time=deadline
        )
        stats = {}
        speculative = inference_turbo_dfs_multitoken(
            speculative_model,
            prefix,
            max_new_tokens=12,
            max_score=2.302585,
            end_time=deadline,
            repeat_len=4,
            stats=stats,
        )
        self.assertEqual(token_results(speculative), token_results(baseline))
        self.assertGreater(stats["block_calls"], 0)
        return baseline_model, speculative_model, stats

    def test_unique_repeated_chains_match_baseline(self):
        baseline, speculative, stats = self.compare_paths()
        self.assertGreater(stats["accepted_extra_tokens"], 0)
        self.assertLess(len(speculative.q_lens), len(baseline.q_lens))

    def test_ambiguous_frame_falls_back_without_reordering_results(self):
        _baseline, _speculative, stats = self.compare_paths({2})
        self.assertGreater(stats["zero_extra_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
