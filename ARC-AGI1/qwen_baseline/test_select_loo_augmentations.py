import bz2
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arc_loader import ArcDataset
from arc_selected_augmentations import apply_selected_augmentations
from select_loo_augmentations import select_successful_augmentations


class SelectLooAugmentationsTest(unittest.TestCase):
    def test_selects_view_when_correct_grid_appears_anywhere(self):
        labels = {
            "taskl1": {
                "task": "task",
                "output": [[7]],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_0.rot90.permute0123456789.ex10"
            candidates = [
                {"solution": np.asarray([[9]])},
                {"solution": np.asarray([[7]])},
            ]
            with bz2.BZ2File(path, "wb") as handle:
                pickle.dump(candidates, handle)
            selected, stats = select_successful_augmentations(directory, labels, ["task"])

        self.assertEqual(selected["task"], ["rot90.permute0123456789.ex10"])
        self.assertEqual(stats["task"]["valid_candidates"], 2)

    def test_replays_descriptor_on_every_test_output(self):
        puzzle = ArcDataset(
            queries={
                "task_0": {
                    "train": [
                        {"input": [[1, 2]], "output": [[3, 4]]},
                        {"input": [[5, 6]], "output": [[7, 8]]},
                    ],
                    "test": [{"input": [[1, 2], [3, 4]]}],
                },
                "task_1": {
                    "train": [
                        {"input": [[1, 2]], "output": [[3, 4]]},
                        {"input": [[5, 6]], "output": [[7, 8]]},
                    ],
                    "test": [{"input": [[5, 6], [7, 8]]}],
                },
            },
            keys=["task_0", "task_1"],
        )
        replay = apply_selected_augmentations(puzzle, ["rot90.ex10"])

        self.assertEqual(len(replay.keys), 2)
        self.assertTrue(all(key.endswith(".rot90.ex10") for key in replay.keys))
        self.assertTrue(
            np.array_equal(
                replay.queries[replay.keys[0]]["train"][0]["input"],
                [[6], [5]],
            )
        )


if __name__ == "__main__":
    unittest.main()
