import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from repair_mining import (
    build_leave_one_out_probe,
    deterministic_sample_paths,
    discover_subset_root,
    error_mask_diagnostics,
    gold_shape_error_mask,
    stabilize_inference_state,
    transform_grid,
    validate_pairs,
)


def pair(value, rows=2, cols=3):
    grid = np.full((rows, cols), value, dtype=int).tolist()
    output = np.full((rows, cols), (value + 1) % 10, dtype=int).tolist()
    return {"input": grid, "output": output}


class RepairMiningTest(unittest.TestCase):
    def test_leave_one_out_is_reproducible_and_never_leaks_query(self):
        pairs = [pair(value) for value in range(6)]
        first = build_leave_one_out_probe(
            pairs,
            subset="nvarc_training",
            puzzle_id="puzzle_1",
            anchor_id="anchor",
            source_path="puzzle_1.json",
            seed=19,
        )
        second = build_leave_one_out_probe(
            pairs,
            subset="nvarc_training",
            puzzle_id="puzzle_1",
            anchor_id="anchor",
            source_path="puzzle_1.json",
            seed=19,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.demonstration_indices), 2)
        self.assertNotIn(first.query_index, first.demonstration_indices)
        self.assertEqual(len(set(first.demonstration_indices + (first.query_index,))), 3)

    def test_one_global_augmentation_is_applied_to_inputs_and_outputs(self):
        mapping = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        grid = [[0, 1, 2], [3, 4, 5]]
        transformed = transform_grid(grid, 1, mapping)
        expected = np.asarray(mapping)[np.rot90(np.asarray(grid), 1)].tolist()
        self.assertEqual(transformed, expected)

    def test_gold_shape_mask_handles_missing_and_extra_cells(self):
        gold = [[1, 2, 3], [4, 5, 6]]
        short = [[1, 9], [4, 5]]
        self.assertEqual(gold_shape_error_mask(short, gold), [[0, 1, 1], [0, 0, 1]])

        tall = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(gold_shape_error_mask(tall, gold), [[0, 0, 0], [0, 0, 0]])
        diagnostics = error_mask_diagnostics(tall, gold)
        self.assertEqual(diagnostics["wrong_or_missing_gold_cells"], 0)
        self.assertEqual(diagnostics["extra_prediction_cells"], 3)
        self.assertEqual(diagnostics["total_wrong_missing_or_extra_cells"], 3)
        self.assertFalse(diagnostics["shape_equal"])

    def test_deterministic_path_sampling_excludes_validation_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for anchor in ["keep_a", "keep_b", "excluded"]:
                (root / anchor).mkdir()
                for index in range(3):
                    (root / anchor / f"{anchor}_{index}.json").write_text(json.dumps([pair(index)] * 3))
            sampled = deterministic_sample_paths(
                root,
                count=10,
                seed=7,
                excluded_anchor_ids={"excluded"},
            )
            self.assertEqual(len(sampled), 6)
            self.assertTrue(all(path.parent.name != "excluded" for path in sampled))

    def test_discovers_kaggle_owner_scoped_dataset_mount_without_recursive_walk(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            subset_root = (
                input_root
                / "datasets"
                / "sorokin"
                / "nvarc-synthetic-puzzles"
                / "nvarc_training"
            )
            puzzle_dir = subset_root / "anchor"
            puzzle_dir.mkdir(parents=True)
            (puzzle_dir / "puzzle.json").write_text(json.dumps([pair(1), pair(2), pair(3)]))
            self.assertEqual(discover_subset_root(input_root, "nvarc_training"), subset_root)

    def test_pair_validation_rejects_bad_grids(self):
        self.assertTrue(validate_pairs([pair(1), pair(2), pair(3)]))
        bad = [pair(1), pair(2), {"input": [[11]], "output": [[0]]}]
        self.assertFalse(validate_pairs(bad))

    def test_stabilizes_patched_inference_state(self):
        class Layer:
            gradient_checkpointing = True

        class Model:
            training = True

            def __init__(self):
                self.layer = Layer()

            def eval(self):
                self.training = False

            def gradient_checkpointing_disable(self):
                self.layer.gradient_checkpointing = False

            def modules(self):
                return [self, self.layer]

        model = Model()
        stabilize_inference_state(model)
        self.assertFalse(model.training)
        self.assertFalse(model.layer.gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
