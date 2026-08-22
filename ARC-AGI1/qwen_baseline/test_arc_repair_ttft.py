import copy
import unittest

from arc_loader import ArcDataset, QwenFormatter
from arc_repair_ttft import (
    build_pair_probe_views,
    build_stage_two_mixture,
    deterministic_rows,
    repair_prompt,
    split_for_known_pair,
)


def pair(value):
    return {
        "input": [[value, 0], [0, value]],
        "output": [[value, value], [0, value]],
    }


class ArcRepairTTFTTest(unittest.TestCase):
    def setUp(self):
        self.key = "abcdef12"
        self.query = {
            "train": [pair(1), pair(2), pair(3)],
            "test": [{"input": [[9]]}],
        }
        self.dataset = ArcDataset(
            queries={self.key: copy.deepcopy(self.query)},
            replies={},
            keys=[self.key],
            is_orig=True,
        )

    def test_known_pair_split_never_places_target_output_in_demonstrations(self):
        split = split_for_known_pair(self.dataset, self.key, 2)
        self.assertEqual(split.reserved_pair, self.query["train"][2])
        self.assertEqual(split.sft_pair_indices, [0, 1])
        self.assertEqual(split.reduced_dataset.queries[self.key]["train"], self.query["train"][:2])

    def test_probe_view_budget_is_exact(self):
        views = build_pair_probe_views(
            self.dataset,
            self.key,
            pair_index=2,
            formatter=QwenFormatter(tokenizer=None),
            view_count=16,
            seed=17,
        )
        self.assertEqual(len(views), 16)
        self.assertTrue(all(len(view.transformed_output) == 2 for view in views))

    def test_stage_budgets_are_exact_and_repair_is_oversampled(self):
        ordinary = [{"kind": "ordinary", "value": index} for index in range(128)]
        repairs = [{"kind": "repair", "value": index} for index in range(3)]
        mixture, stats = build_stage_two_mixture(
            ordinary_rows=ordinary,
            repair_rows=repairs,
            total_steps=64,
            repair_fraction=0.5,
            seed=3,
        )
        self.assertEqual(len(mixture), 64)
        self.assertEqual(stats, {"ordinary_steps": 32, "repair_steps": 32})
        self.assertEqual(sum(row["kind"] == "repair" for row in mixture), 32)

    def test_no_repair_rows_reallocates_entire_budget_to_ordinary_sft(self):
        mixture, stats = build_stage_two_mixture(
            ordinary_rows=[{"value": index} for index in range(8)],
            repair_rows=[],
            total_steps=96,
            repair_fraction=0.5,
            seed=4,
        )
        self.assertEqual(len(mixture), 96)
        self.assertEqual(stats, {"ordinary_steps": 96, "repair_steps": 0})

    def test_deterministic_rows_repeats_without_changing_budget(self):
        rows = [{"value": 0}, {"value": 1}]
        self.assertEqual(deterministic_rows(rows, 7, 9), deterministic_rows(rows, 7, 9))
        self.assertEqual(len(deterministic_rows(rows, 7, 9)), 7)

    def test_repair_prompt_contains_wrong_candidate_and_gold_shaped_mask(self):
        prompt = repair_prompt(
            "<|im_start|>user\n0<|im_end|><|im_start|>assistant\n",
            [[1, 0]],
            [[1, 2], [3, 4]],
        )
        self.assertIn("<REPAIR>\n", prompt)
        self.assertIn("01\n11<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()
