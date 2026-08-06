import unittest

import numpy as np

from arc_decoder import score_kgmon, score_kgmon_median
import run_chunked_sglang_pipeline as pipeline


def _sample(solution, scores):
    return {
        "solution": np.asarray(solution),
        "beam_score": 0.0,
        "score_aug": scores,
    }


class KgmonAggregationTests(unittest.TestCase):
    def test_mean_remains_the_default_selection_algorithm(self):
        args = pipeline.build_parser().parse_args([])
        self.assertEqual(args.selection_algorithm, "score_kgmon")

    def test_median_is_exposed_as_a_selection_algorithm(self):
        args = pipeline.build_parser().parse_args(
            ["--selection-algorithm", "score_kgmon_median"]
        )
        self.assertEqual(args.selection_algorithm, "score_kgmon_median")

    def test_median_is_robust_to_an_outlier_rescore_view(self):
        robust_candidate = [[1]]
        steady_candidate = [[2]]
        guesses = {
            "robust-1": _sample(robust_candidate, [1.0, 1.0, 100.0]),
            "robust-2": _sample(robust_candidate, [1.0, 1.0, 100.0]),
            "steady": _sample(steady_candidate, [2.0, 2.0, 2.0]),
        }

        mean_ranked = score_kgmon(guesses)
        median_ranked = score_kgmon_median(guesses)

        self.assertTrue(np.array_equal(mean_ranked[0], steady_candidate))
        self.assertTrue(np.array_equal(median_ranked[0], robust_candidate))


if __name__ == "__main__":
    unittest.main()
