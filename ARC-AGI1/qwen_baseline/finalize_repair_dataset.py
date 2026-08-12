"""Validate, split, and manifest ARC repair-mining JSONL shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--expected-probes", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_for(anchor_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:split:{anchor_id}".encode()).hexdigest()[:16], 16)
    bucket = value / float(16**16)
    if bucket < 0.8:
        return "train"
    if bucket < 0.9:
        return "dev"
    return "test"


def distribution(values: Iterable[int]) -> dict[str, float] | None:
    ordered = sorted(values)
    if not ordered:
        return None

    def percentile(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))

    return {
        "min": float(ordered[0]),
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def finalize(
    shard_dir: Path,
    validation_path: Path,
    output_dir: Path,
    *,
    seed: int,
    expected_probes: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    validation_anchors = set(json.loads(validation_path.read_text()))
    shard_paths = sorted(shard_dir.glob("repair_failures.rank*.jsonl"))
    summary_paths = sorted(shard_dir.glob("summary.rank*.json"))
    if not shard_paths or not summary_paths:
        raise FileNotFoundError("Repair shards and worker summaries are required")

    summaries = [json.loads(path.read_text()) for path in summary_paths]
    aggregate = {
        key: sum(summary["counts"].get(key, 0) for summary in summaries)
        for key in sorted({key for summary in summaries for key in summary["counts"]})
    }
    if expected_probes is not None and aggregate["assigned_probes"] != expected_probes:
        raise AssertionError((aggregate["assigned_probes"], expected_probes))
    if (
        aggregate["sequence_too_long"]
        + aggregate["teacher_forced_exact"]
        + aggregate["teacher_forced_failures"]
        != aggregate["assigned_probes"]
    ):
        raise AssertionError(aggregate)

    raw_records = [record for path in shard_paths for record in load_jsonl(path)]
    type_counts = Counter(record["record_type"] for record in raw_records)
    unexpected_types = set(type_counts) - {"repair_failure", "repair_noop"}
    if unexpected_types:
        raise AssertionError(f"Unexpected record types: {sorted(unexpected_types)}")
    records = [record for record in raw_records if record["record_type"] == "repair_failure"]
    records.sort(key=lambda record: record["global_index"])

    if len({record["global_index"] for record in raw_records}) != len(raw_records):
        raise AssertionError("Duplicate global indices")
    if len({record["source_relpath"] for record in raw_records}) != len(raw_records):
        raise AssertionError("Duplicate source paths")
    if any(record["anchor_id"] in validation_anchors for record in raw_records):
        raise AssertionError("Validation anchor leaked into repair records")
    if any("<REPAIR>\n" not in record["input"] for record in raw_records):
        raise AssertionError("Repair control token missing")
    if any(record["total_wrong_missing_or_extra_cells"] <= 0 for record in records):
        raise AssertionError("Non-error record survived failure filtering")
    if any(
        record["global_index"] not in record["decoder"]["batch_member_global_indices"]
        for record in raw_records
    ):
        raise AssertionError("Decoder batch provenance is inconsistent")

    for record in records:
        record["split"] = split_for(record["anchor_id"], seed)
    split_anchors = {
        split: {record["anchor_id"] for record in records if record["split"] == split}
        for split in ("train", "dev", "test")
    }
    if any(
        split_anchors[left] & split_anchors[right]
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise AssertionError("Anchor leakage across splits")

    combined_path = output_dir / "repair_failures.all.jsonl"
    combined_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )
    split_counts = {}
    for split in ("train", "dev", "test"):
        split_records = [record for record in records if record["split"] == split]
        split_path = output_dir / f"repair_failures.{split}.jsonl"
        split_path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in split_records)
        )
        split_counts[split] = {
            "records": len(split_records),
            "anchors": len(split_anchors[split]),
            "sha256": sha256(split_path),
        }

    manifest = {
        "config": {
            "seed": seed,
            "expected_probes": expected_probes,
            "source_subset": "nvarc_training",
        },
        "aggregate_worker_counts": aggregate,
        "raw_record_types": dict(sorted(type_counts.items())),
        "records": {
            "repair_failures": len(records),
            "excluded_rollout_exact_noops": type_counts["repair_noop"],
            "unique_anchors": len({record["anchor_id"] for record in records}),
            "unique_puzzles": len({record["puzzle_id"] for record in records}),
            "shape_errors": sum(not record["shape_equal"] for record in records),
            "wrong_cells": distribution(
                record["total_wrong_missing_or_extra_cells"] for record in records
            ),
            "wrong_or_missing_gold_cells": distribution(
                record["wrong_or_missing_gold_cells"] for record in records
            ),
            "extra_prediction_cells": distribution(
                record["extra_prediction_cells"] for record in records
            ),
        },
        "splits": split_counts,
        "shard_sha256": {path.name: sha256(path) for path in shard_paths},
        "combined_sha256": sha256(combined_path),
        "worker_summaries": summaries,
    }
    manifest_path = output_dir / "repair_mining_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    args = parse_args()
    manifest = finalize(
        args.shard_dir,
        args.validation_path,
        args.output_dir,
        seed=args.seed,
        expected_probes=args.expected_probes,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "worker_summaries"}, indent=2))


if __name__ == "__main__":
    main()
