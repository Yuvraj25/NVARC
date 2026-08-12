import json
import tempfile
import unittest
from pathlib import Path

from finalize_repair_dataset import finalize


def record(index, anchor, record_type, wrong_cells):
    return {
        "record_type": record_type,
        "global_index": index,
        "source_relpath": f"{anchor}/p{index}.json",
        "anchor_id": anchor,
        "puzzle_id": f"{anchor}_p{index}",
        "input": "prompt<REPAIR>\n0",
        "reply": "0",
        "prediction": [[0]],
        "shape_equal": True,
        "prediction_shape": [1, 1],
        "gold_shape": [1, 1],
        "wrong_or_missing_gold_cells": wrong_cells,
        "extra_prediction_cells": 0,
        "total_wrong_missing_or_extra_cells": wrong_cells,
        "decoder": {"batch_member_global_indices": [index]},
    }


class FinalizeRepairDatasetTest(unittest.TestCase):
    def test_filters_legacy_noop_and_preserves_anchor_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = root / "shards"
            shards.mkdir()
            validation = root / "validation.json"
            validation.write_text(json.dumps({"heldout": {}}))
            rows = [
                record(0, "a", "repair_failure", 3),
                record(1, "b", "repair_noop", 0),
                record(2, "c", "repair_failure", 7),
            ]
            (shards / "repair_failures.rank0.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (shards / "summary.rank0.json").write_text(
                json.dumps(
                    {
                        "counts": {
                            "assigned_probes": 3,
                            "sequence_too_long": 0,
                            "teacher_forced_exact": 0,
                            "teacher_forced_failures": 3,
                            "usable_repair_failures": 3,
                            "invalid_rollouts": 0,
                        }
                    }
                )
            )

            manifest = finalize(
                shards,
                validation,
                root / "final",
                seed=20260811,
                expected_probes=3,
            )

            self.assertEqual(manifest["raw_record_types"], {"repair_failure": 2, "repair_noop": 1})
            self.assertEqual(manifest["records"]["repair_failures"], 2)
            self.assertEqual(manifest["records"]["excluded_rollout_exact_noops"], 1)
            self.assertEqual(manifest["records"]["summary_only_rollout_exact_noops"], 0)
            self.assertEqual(manifest["records"]["legacy_materialized_rollout_exact_noops"], 1)
            combined = [
                json.loads(line)
                for line in (root / "final" / "repair_failures.all.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["global_index"] for row in combined}, {0, 2})
            self.assertTrue(all(row["record_type"] == "repair_failure" for row in combined))
            self.assertTrue(all(row["split"] in {"train", "dev", "test"} for row in combined))


if __name__ == "__main__":
    unittest.main()
