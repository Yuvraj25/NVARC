import copy
import unittest

import numpy as np
import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_opsd import (
    _gold_metrics,
    _teacher_trajectory_diagnostics,
    build_opsd_examples,
    classify_rollout,
    clone_frozen_teacher_adapter,
    deployment_rollout_limit,
    deterministic_reserved_pair_index,
    exact_reverse_kl,
    split_puzzle_for_opsd,
    teacher_advantage_gate,
)
from arc_search import EOS_ID


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

    def test_teacher_clone_accepts_named_adapter_key_spelling(self):
        class FakeModel:
            def __init__(self):
                self.peft_config = {"default": {"r": 1}}
                self.student = torch.tensor([1.0, 2.0])
                self.teacher = None
                self.active_adapter = "default"

            def add_adapter(self, name, config):
                self.peft_config[name] = config
                self.teacher = torch.zeros_like(self.student)

            def set_adapter(self, name):
                self.active_adapter = name

            def named_parameters(self):
                return iter(
                    [
                        ("base.lora_A.default.weight", self.student),
                        ("base.lora_A.opsd_teacher.weight", self.teacher),
                    ]
                )

        model = FakeModel()

        def get_state(_model, adapter_name):
            if adapter_name == "default":
                return {"base.lora_A.weight": _model.student}
            return {f"base.lora_A.{adapter_name}.weight": _model.teacher}

        def set_state(_model, state, adapter_name):
            _model.teacher.copy_(state["base.lora_A.weight"])
            return type("LoadResult", (), {"unexpected_keys": []})()

        clone_frozen_teacher_adapter(model, get_state, set_state)
        self.assertTrue(torch.equal(model.student, model.teacher))
        self.assertEqual(model.active_adapter, "opsd_teacher")

    def test_rollout_limit_uses_deployment_cap_and_both_context_lengths(self):
        self.assertEqual(
            deployment_rollout_limit(
                formatter_max_new_tokens=902,
                max_seq_length=8192,
                student_prompt_tokens=7000,
                teacher_prompt_tokens=7600,
            ),
            592,
        )
        self.assertEqual(
            deployment_rollout_limit(
                formatter_max_new_tokens=902,
                max_seq_length=8192,
                student_prompt_tokens=100,
                teacher_prompt_tokens=200,
            ),
            902,
        )

    def test_teacher_gate_requires_exactness_and_nll_advantage_for_every_view(self):
        student = {"nll": 3.0, "restricted_greedy_exact": False}
        accepted, reason = teacher_advantage_gate(
            student,
            {"nll": 2.0, "restricted_greedy_exact": False},
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "teacher_not_restricted_greedy_exact")

        accepted, reason = teacher_advantage_gate(
            student,
            {"nll": 3.5, "restricted_greedy_exact": True},
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "teacher_no_nll_advantage")

        accepted, reason = teacher_advantage_gate(
            student,
            {"nll": 2.0, "restricted_greedy_exact": True},
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

    def test_gold_metrics_records_wrong_argmax_positions_and_confidence(self):
        logits = torch.full((3, 16), -10.0)
        logits[0, 1] = 8.0
        logits[1, 4] = 7.0
        logits[1, 2] = 6.0
        logits[2, EOS_ID] = 8.0
        metrics = _gold_metrics(logits, [1, 2, EOS_ID])
        self.assertFalse(metrics["restricted_greedy_exact"])
        self.assertEqual(metrics["wrong_token_count"], 1)
        detail = metrics["wrong_token_details"][0]
        self.assertEqual(detail["position"], 1)
        self.assertEqual(detail["gold_token_id"], 2)
        self.assertEqual(detail["argmax_token_id"], 4)
        self.assertGreater(detail["argmax_probability"], detail["gold_probability"])
        self.assertGreater(detail["argmax_margin"], 0.0)

    def test_missing_eos_is_never_passed_to_grid_converter(self):
        class TrackingFormatter:
            def __init__(self):
                self.calls = 0

            def convert_tokens_to_array(self, tokens):
                self.calls += 1
                return np.asarray([[1]])

        formatter = TrackingFormatter()
        grid, reason = classify_rollout(formatter, [1, 10, 2])
        self.assertIsNone(grid)
        self.assertEqual(reason, "missing_eos")
        self.assertEqual(formatter.calls, 0)

        grid, reason = classify_rollout(formatter, [1, EOS_ID])
        self.assertTrue(np.array_equal(grid, [[1]]))
        self.assertIsNone(reason)
        self.assertEqual(formatter.calls, 1)

    def test_corrupted_trajectory_teacher_diagnostics(self):
        rollout = [1, 2, 10, EOS_ID, 4]
        gold = [1, 3, 10, EOS_ID]
        logits = torch.full((len(rollout), 16), -4.0)
        logits[0, 1] = 4.0
        logits[1, 3] = 4.0
        logits[2, 2] = 4.0
        logits[3, EOS_ID] = 4.0
        logits[4, 4] = 4.0
        diagnostics = _teacher_trajectory_diagnostics(
            teacher_logits=logits,
            rollout_ids=rollout,
            gold_ids=gold,
            first_divergence=1,
        )
        self.assertEqual(diagnostics["divergence_student_token_id"], 2)
        self.assertEqual(diagnostics["divergence_gold_token_id"], 3)
        self.assertEqual(diagnostics["divergence_student_token_type"], "digit")
        self.assertEqual(diagnostics["divergence_gold_token_type"], "digit")
        self.assertTrue(diagnostics["teacher_gold_correct_at_divergence"])
        self.assertAlmostEqual(diagnostics["teacher_next_token_accuracy_after_divergence"], 0.5)
        self.assertEqual(diagnostics["trajectory_positions_beyond_gold"], 1)
        self.assertIsNone(diagnostics["teacher_aligned_gold_token_probabilities"][-1])


if __name__ == "__main__":
    unittest.main()
