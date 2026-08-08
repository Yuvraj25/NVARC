import unittest

import numpy as np

from analyze_candidate_recovery import analyze_grid, grid_tokens


class CandidateRecoveryTest(unittest.TestCase):
    def test_exact_grid(self):
        result = analyze_grid(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))
        self.assertEqual(result["category"], "exact")
        self.assertIsNone(result["first_divergence_token"])

    def test_single_cell_error_recovers_immediately(self):
        result = analyze_grid(np.array([[1, 9], [3, 4]]), np.array([[1, 2], [3, 4]]))
        self.assertEqual(result["category"], "same_shape_single_cell")
        self.assertEqual(result["wrong_cells"], 1)
        self.assertTrue(result["immediate_next_cell_correct"])
        self.assertEqual(result["correct_cell_suffix"], 2)

    def test_sustained_same_shape_error(self):
        result = analyze_grid(np.array([[1, 9], [9, 9]]), np.array([[1, 2], [3, 4]]))
        self.assertEqual(result["category"], "same_shape_multiple_cells")
        self.assertEqual(result["wrong_cells"], 3)
        self.assertFalse(result["immediate_next_cell_correct"])
        self.assertEqual(result["cell_error_spans"], 1)

    def test_early_newline_from_shorter_width(self):
        candidate = np.array([[1, 2], [3, 4]])
        gold = np.array([[1, 2, 3], [4, 5, 6]])
        result = analyze_grid(candidate, gold)
        self.assertEqual(result["category"], "wrong_shape_early_newline_short_row")
        self.assertEqual(result["first_divergence_token"], 2)
        self.assertEqual(result["first_candidate_token_type"], "newline")
        self.assertEqual(result["first_gold_token_type"], "digit")

    def test_early_eos_for_correct_prefix_rows(self):
        candidate = np.array([[1, 2]])
        gold = np.array([[1, 2], [3, 4]])
        result = analyze_grid(candidate, gold)
        self.assertEqual(result["category"], "wrong_shape_early_eos")
        self.assertEqual(grid_tokens(candidate), [1, 2, 15])
        self.assertEqual(result["first_divergence_token"], 2)

    def test_late_eos_for_extra_rows(self):
        candidate = np.array([[1, 2], [3, 4]])
        gold = np.array([[1, 2]])
        result = analyze_grid(candidate, gold)
        self.assertEqual(result["category"], "wrong_shape_late_eos_extra_output")
        self.assertEqual(result["first_divergence_token"], 2)


if __name__ == "__main__":
    unittest.main()
