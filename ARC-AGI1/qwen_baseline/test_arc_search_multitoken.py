import time
import unittest
from types import SimpleNamespace

import torch

from arc_search import inference_turbo_dfs
from arc_search_multitoken import (
    _GridState,
    _draft_len_for_token,
    _token_is_structurally_legal,
    inference_turbo_dfs_multitoken,
    turbo_dfs_multitoken_structured,
)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, prompt_len, targets, ambiguous_positions=()):
        self.prompt_len = prompt_len
        self.targets = targets
        self.ambiguous_positions = set(ambiguous_positions)
        self.q_lens = []
        self.batch_sizes = []

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
        self.batch_sizes.append(batch)
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

    def test_structured_path_preserves_rectangular_candidate(self):
        prefix = [[11, 12], [11, 12]]
        targets = [4, 4, 10, 2, 2, 15]
        baseline_model = FakeModel(len(prefix[0]), targets)
        structured_model = FakeModel(len(prefix[0]), targets)
        deadline = time.time() + 30
        baseline = inference_turbo_dfs(
            baseline_model,
            prefix,
            max_new_tokens=12,
            max_score=2.302585,
            end_time=deadline,
        )
        stats = {}
        structured = inference_turbo_dfs_multitoken(
            structured_model,
            prefix,
            max_new_tokens=12,
            max_score=2.302585,
            end_time=deadline,
            repeat_len=4,
            stats=stats,
            structured_rows=True,
        )
        self.assertEqual(token_results(structured), token_results(baseline))
        self.assertGreater(stats["length_bucket_calls"], 0)

    def test_grid_state_rejects_ragged_rows_and_caps_first_row(self):
        state = _GridState()
        for _ in range(30):
            self.assertTrue(_token_is_structurally_legal(state, 1))
            state = state.advance(1)
        self.assertFalse(_token_is_structurally_legal(state, 1))
        self.assertTrue(_token_is_structurally_legal(state, 10))

        state = state.advance(10)
        for _ in range(29):
            state = state.advance(2)
        self.assertFalse(_token_is_structurally_legal(state, 10))
        self.assertFalse(_token_is_structurally_legal(state, 15))
        self.assertTrue(_token_is_structurally_legal(state, 2))
        state = state.advance(2)
        self.assertTrue(_token_is_structurally_legal(state, 10))
        self.assertTrue(_token_is_structurally_legal(state, 15))

    def test_draft_length_stops_at_known_row_boundary(self):
        first_row = _GridState(column=27)
        self.assertEqual(_draft_len_for_token(first_row, 3, 9, 100), 3)
        known_width = _GridState(width=20, column=13, completed_rows=2)
        self.assertEqual(_draft_len_for_token(known_width, 3, 9, 100), 7)
        self.assertEqual(_draft_len_for_token(known_width, 10, 9, 100), 1)

    def test_different_safe_lengths_make_separate_calls(self):
        prompt_len = 2
        model = FakeModel(prompt_len, [1, 1, 1, 1, 1, 15])
        logits = torch.full((2, 16), -20.0)
        logits[:, 1] = 8.0
        cache_tensor = torch.zeros((2, 1, prompt_len, 1))
        stats = {}
        turbo_dfs_multitoken_structured(
            model=model,
            logits=logits,
            max_new_tokens=8,
            max_score=2.302585,
            scores=[0.0, 0.0],
            pos=prompt_len,
            cache=[(cache_tensor, cache_tensor.clone())],
            states=[_GridState(width=3), _GridState(width=5)],
            start_time=time.time(),
            end_time=time.time() + 30,
            repeat_len=4,
            stats=stats,
        )
        self.assertIn(4, model.q_lens)
        self.assertIn(3, model.q_lens)
        self.assertGreaterEqual(stats["q4_calls"], 1)
        self.assertGreaterEqual(stats["q3_calls"], 1)
        # Cached Unsloth allocates its paged-attention scratch buffers at the
        # prefill batch size. Logical length buckets therefore use duplicated
        # real lanes as disposable fillers instead of changing physical batch.
        self.assertEqual(set(model.batch_sizes), {2})
        self.assertGreater(stats["padded_model_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
