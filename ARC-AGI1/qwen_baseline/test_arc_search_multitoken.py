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
    resume_turbo_dfs_multitoken,
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
        self.input_batches = []
        self.cache_batch_sizes = []

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
        self.input_batches.append(input_ids[:, 0].tolist())
        if past_key_values is not None:
            self.cache_batch_sizes.append(past_key_values[0][0].shape[0])
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


class FrontierFakeModel(FakeModel):
    def _row(self, generated):
        row = torch.full((16,), -20.0)
        target = self.targets[min(generated, len(self.targets) - 1)]
        row[target] = 8.0
        if generated in self.ambiguous_positions:
            row[3 if target != 3 else 4] = 6.4
        return row


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

    def test_relaxed_threshold_resumes_only_captured_boundary_branches(self):
        prefix = [[11, 12], [11, 12]]
        targets = [4, 4, 10, 2, 2, 15]
        primary_max = -torch.log(torch.tensor(0.2)).item()
        relaxed_max = -torch.log(torch.tensor(0.1)).item()
        deadline = time.time() + 30

        primary_model = FrontierFakeModel(len(prefix[0]), targets, {0, 2})
        primary, frontier = inference_turbo_dfs_multitoken(
            primary_model,
            prefix,
            max_new_tokens=12,
            max_score=primary_max,
            end_time=deadline,
            repeat_len=4,
            frontier_max_score=relaxed_max,
            return_frontier=True,
        )
        self.assertTrue(frontier)
        # Position 2 is traversed inside a speculative block. Its rejected
        # sibling must still be captured, not only ordinary recursion frames.
        self.assertTrue(any(len(entry["tokens"]) == 3 for entry in frontier))

        resumed_model = FrontierFakeModel(len(prefix[0]), targets, {0, 2})
        resumed = resume_turbo_dfs_multitoken(
            resumed_model,
            prefix,
            frontier,
            max_new_tokens=12,
            max_score=relaxed_max,
            end_time=deadline,
            repeat_len=4,
        )
        full_model = FrontierFakeModel(len(prefix[0]), targets, {0, 2})
        full_relaxed = inference_turbo_dfs_multitoken(
            full_model,
            prefix,
            max_new_tokens=12,
            max_score=relaxed_max,
            end_time=deadline,
            repeat_len=4,
        )
        combined = {
            lane: {tuple(tokens) for _score, tokens in beams}
            for lane, beams in primary
        }
        for lane, beams in resumed:
            combined.setdefault(lane, set()).update(
                tuple(tokens) for _score, tokens in beams
            )
        expected = {
            lane: {tuple(tokens) for _score, tokens in beams}
            for lane, beams in full_relaxed
        }
        self.assertEqual(combined, expected)

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

    def test_unbucketed_pruning_rejects_ragged_branch(self):
        prefix = [[11, 12]]
        # First row has width two; the third digit in row two is therefore
        # structurally impossible and must end the branch without another
        # length-bucketed model path.
        targets = [1, 1, 10, 2, 2, 2, 15]
        model = FakeModel(len(prefix[0]), targets)
        stats = {}
        result = inference_turbo_dfs_multitoken(
            model,
            prefix,
            max_new_tokens=12,
            max_score=2.302585,
            end_time=time.time() + 30,
            repeat_len=4,
            stats=stats,
            prune_structural_invalid=True,
        )
        self.assertEqual(token_results(result), {})
        self.assertGreater(stats["structural_pruned_candidates"], 0)
        self.assertEqual(stats.get("structural_invalid_model_lanes", 0), 0)

    def test_unbucketed_pruning_preserves_rectangular_candidate(self):
        prefix = [[11, 12]]
        targets = [1, 1, 10, 2, 2, 15]
        control_model = FakeModel(len(prefix[0]), targets)
        pruned_model = FakeModel(len(prefix[0]), targets)
        control = inference_turbo_dfs_multitoken(
            control_model,
            prefix,
            max_new_tokens=12,
            max_score=2.302585,
            end_time=time.time() + 30,
            repeat_len=4,
        )
        pruned = inference_turbo_dfs_multitoken(
            pruned_model,
            prefix,
            max_new_tokens=12,
            max_score=2.302585,
            end_time=time.time() + 30,
            repeat_len=4,
            prune_structural_invalid=True,
        )
        self.assertEqual(token_results(pruned), token_results(control))

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
        self.assertEqual(set(model.cache_batch_sizes), {2})
        self.assertTrue(
            any(q_len == 4 and tokens == [0, 1]
                for q_len, tokens in zip(model.q_lens, model.input_batches))
        )
        self.assertTrue(
            any(q_len == 3 and tokens == [1, 0]
                for q_len, tokens in zip(model.q_lens, model.input_batches))
        )
        self.assertGreater(stats["padded_model_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
