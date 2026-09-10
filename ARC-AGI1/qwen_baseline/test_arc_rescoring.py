import math
import unittest
from types import SimpleNamespace

import torch

from arc_rescoring import calc_scores


class _CharacterTokenizer:
    def encode(self, text):
        return [0] * len(text)


class _UniformModel:
    device = "cpu"

    def __call__(self, input_ids, **_kwargs):
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 16)))


class RescoringTests(unittest.TestCase):
    def test_mean_nll_removes_answer_length_effect(self):
        raw = calc_scores(
            ["qq", "q"],
            ["aaa", "a"],
            _CharacterTokenizer(),
            _UniformModel(),
        )
        normalized = calc_scores(
            ["qq", "q"],
            ["aaa", "a"],
            _CharacterTokenizer(),
            _UniformModel(),
            normalize_by_answer_tokens=True,
        )

        self.assertAlmostEqual(raw[0], 3 * math.log(16), places=5)
        self.assertAlmostEqual(raw[1], math.log(16), places=5)
        self.assertAlmostEqual(normalized[0], math.log(16), places=5)
        self.assertAlmostEqual(normalized[1], math.log(16), places=5)


if __name__ == "__main__":
    unittest.main()
