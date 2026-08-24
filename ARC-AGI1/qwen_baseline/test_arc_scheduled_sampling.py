import unittest

import torch

from arc_scheduled_sampling import mix_final_answer_with_restricted_argmax


class ScheduledSamplingMixTest(unittest.TestCase):
    def test_only_final_answer_digits_are_replaced(self):
        input_ids = torch.tensor([[11, 10, 12, 10, 1, 2, 15, 11, 10, 12, 10, 3, 4, 10, 5, 15]])
        labels = torch.tensor([[-100, -100, -100, -100, 1, 2, 15, -100, -100, -100, -100, 3, 4, 10, 5, 15]])
        logits = torch.full((1, input_ids.shape[1], 16), -20.0)
        logits[0, 10, 7] = 20.0
        logits[0, 11, 8] = 20.0
        logits[0, 13, 9] = 20.0

        mixed, diagnostic = mix_final_answer_with_restricted_argmax(
            input_ids, labels, logits, mix_probability=1.0, seed=123
        )

        self.assertEqual(mixed[0, 4:7].tolist(), [1, 2, 15])
        self.assertEqual(mixed[0, 11:16].tolist(), [7, 8, 10, 9, 15])
        self.assertEqual(diagnostic["eligible_digit_tokens"], 3)
        self.assertEqual(diagnostic["changed_digit_tokens"], 3)
        self.assertEqual(diagnostic["first_changed_offset"], 0)

    def test_restricted_argmax_ignores_illegal_vocabulary_logits(self):
        input_ids = torch.tensor([[12, 10, 2, 15]])
        labels = torch.tensor([[-100, -100, 2, 15]])
        logits = torch.full((1, 4, 32), -20.0)
        logits[0, 1, 29] = 100.0
        logits[0, 1, 6] = 10.0

        mixed, diagnostic = mix_final_answer_with_restricted_argmax(
            input_ids, labels, logits, mix_probability=1.0, seed=123
        )

        self.assertEqual(int(mixed[0, 2]), 6)
        self.assertEqual(diagnostic["predicted_wrong_digit_tokens"], 1)

    def test_zero_mix_leaves_input_unchanged_but_reports_accuracy(self):
        input_ids = torch.tensor([[12, 10, 2, 3, 15]])
        labels = torch.tensor([[-100, -100, 2, 3, 15]])
        logits = torch.full((1, 5, 16), -20.0)
        logits[0, 1, 2] = 20.0
        logits[0, 2, 9] = 20.0

        mixed, diagnostic = mix_final_answer_with_restricted_argmax(
            input_ids, labels, logits, mix_probability=0.0, seed=123
        )

        self.assertTrue(torch.equal(mixed, input_ids))
        self.assertEqual(diagnostic["eligible_digit_tokens"], 2)
        self.assertEqual(diagnostic["predicted_wrong_digit_tokens"], 1)
        self.assertEqual(diagnostic["changed_digit_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
