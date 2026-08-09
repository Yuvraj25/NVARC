import unittest

from build_loo_calibration_tasks import build_loo_tasks


class BuildLooCalibrationTasksTest(unittest.TestCase):
    def setUp(self):
        self.challenges = {
            "abc12345": {
                "train": [
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                    {"input": [[5]], "output": [[6]]},
                ],
                "test": [{"input": [[7]]}],
            }
        }

    def test_last_pair_is_removed_from_training_data(self):
        tasks, labels = build_loo_tasks(self.challenges, ["abc12345"], holdout="last")

        self.assertEqual(list(tasks), ["abc12345l2"])
        self.assertEqual(len(tasks["abc12345l2"]["train"]), 2)
        self.assertEqual(tasks["abc12345l2"]["test"], [{"input": [[5]]}])
        self.assertNotIn("output", tasks["abc12345l2"]["test"][0])
        self.assertEqual(labels["abc12345l2"]["output"], [[6]])

    def test_all_builds_one_task_per_fold(self):
        tasks, labels = build_loo_tasks(self.challenges, ["abc12345"], holdout="all")

        self.assertEqual(set(tasks), {"abc12345l0", "abc12345l1", "abc12345l2"})
        self.assertEqual(set(tasks), set(labels))
        for task in tasks.values():
            self.assertEqual(len(task["train"]), 2)
            self.assertNotIn("output", task["test"][0])

    def test_rejects_tasks_with_one_training_pair(self):
        challenges = {
            "short": {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        }
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_loo_tasks(challenges, ["short"])


if __name__ == "__main__":
    unittest.main()
