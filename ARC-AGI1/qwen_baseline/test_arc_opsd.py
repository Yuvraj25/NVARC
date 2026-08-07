import copy
import unittest

import numpy as np
import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_opsd import (
    build_opsd_examples,
    deterministic_reserved_pair_index,
    exact_reverse_kl,
    split_puzzle_for_opsd,
)


def _pair(value):
    return {
        "input": [[value, 0], [0, value]],
        "output": [[value, value], [0, value]],
    }


class ArcOpsdTest(unittest.TestCase):
    def setUp(self):
        self.key = "abcdef12"
        self.query = {
            "train": [_pair(1), _pair(2), _pair(3), _pair(4)],
            "test": [{"input": [[9]]}],
        }
        self.dataset = ArcDataset(
            queries={self.key: copy.deepcopy(self.query)},
            replies={},
            keys=[self.key],
            is_orig=True,
        )
        self.formatter = QwenFormatter(tokenizer=None)

    def test_pair_split_removes_reserved_pair_completely(self):
        split = split_puzzle_for_opsd(self.dataset, self.key, reserved_pair_index=2)
        self.assertEqual(split.reserved_pair, self.query["train"][2])
        self.assertEqual(split.sft_pair_indices, [0, 1, 3])
        self.assertEqual(split.reduced_dataset.queries[self.key]["train"], [self.query["train"][i] for i in [0, 1, 3]])
        self.assertEqual(self.dataset.queries[self.key], self.query)

    def test_pair_choice_is_reproducible_and_bounded(self):
        first = deterministic_reserved_pair_index(self.key, 4)
        self.assertEqual(first, deterministic_reserved_pair_index(self.key, 4))
        self.assertIn(first, range(4))

    def test_same_view_prompts_differ_only_by_privileged_demo_prefix(self):
        split = split_puzzle_for_opsd(self.dataset, self.key, reserved_pair_index=2)
        examples = build_opsd_examples(
            split,
            formatter=self.formatter,
            color_permutations=1,
            cross_view_probability=0.0,
            seed=7,
        )
        self.assertEqual(len(examples), 8)
        for example in examples:
            self.assertFalse(example.is_cross_view)
            self.assertEqual(example.g_key, example.h_key)
            self.assertTrue(example.teacher_prompt.endswith(example.student_prompt))
            self.assertNotEqual(example.teacher_prompt, example.student_prompt)
            self.assertTrue(example.gold_reply.endswith("<|im_end|>"))

    def test_cross_view_examples_reject_transformation_collisions(self):
        split = split_puzzle_for_opsd(self.dataset, self.key, reserved_pair_index=2)
        examples = build_opsd_examples(
            split,
            formatter=self.formatter,
            color_permutations=2,
            cross_view_probability=1.0,
            seed=11,
        )
        self.assertGreater(len(examples), 0)
        for example in examples:
            self.assertTrue(example.is_cross_view)
            self.assertNotEqual(example.g_key, example.h_key)
            h_input = np.asarray(example.transformed_input)
            h_output = np.asarray(example.transformed_output)
            g_input = np.asarray(example.privileged_input)
            g_output = np.asarray(example.privileged_output)
            # A different descriptor alone is insufficient; at least one actual
            # transformed grid must differ. build_opsd_examples enforces this.
            self.assertFalse(np.array_equal(h_input, g_input) and np.array_equal(h_output, g_output))

    def test_reverse_kl_is_exact_and_only_backpropagates_to_student(self):
        student = torch.tensor([[[1.0, 0.0], [0.2, 0.8]]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_(True)
        loss, per_position = exact_reverse_kl(student, teacher)
        self.assertEqual(tuple(per_position.shape), (1, 2))
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

        shifted_student = torch.tensor([[2.0, -1.0]], requires_grad=True)
        shifted_teacher = torch.tensor([[-1.0, 2.0]], requires_grad=True)
        shifted_loss, _ = exact_reverse_kl(shifted_student, shifted_teacher)
        self.assertGreater(float(shifted_loss.detach()), 0.0)
        shifted_loss.backward()
        self.assertIsNotNone(shifted_student.grad)
        self.assertIsNone(shifted_teacher.grad)


if __name__ == "__main__":
    unittest.main()
