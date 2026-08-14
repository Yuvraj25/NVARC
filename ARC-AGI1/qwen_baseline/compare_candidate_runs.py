"""Compare two ARC candidate directories on selected top-2 and oracle accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from probe_candidate_selection import evaluate, hashable, load_candidates


def _valid_grid(grid) -> bool:
    array = np.asarray(grid)
    return (
        array.ndim == 2
        and 1 <= array.shape[0] <= 30
        and 1 <= array.shape[1] <= 30
        and np.issubdtype(array.dtype, np.integer)
        and bool(np.all((0 <= array) & (array <= 9)))
    )


def _run_summary(candidate_dir: Path, solutions: dict, task_keys: list[str]) -> dict:
    grouped = load_candidates(candidate_dir)
    totals, oracle_total, rows = evaluate(grouped, solutions, task_keys)
    by_basekey = {row["basekey"]: row for row in rows}
    sample_count = 0
    valid_samples = 0
    unique_valid = 0
    for groups in grouped.values():
        for grid, samples in groups.items():
            sample_count += len(samples)
            valid = _valid_grid(grid)
            valid_samples += len(samples) if valid else 0
            unique_valid += int(valid)
    return {
        "candidate_dir": str(candidate_dir),
        "selected_top2_score": totals["kgmon"],
        "oracle_score": oracle_total,
        "decoded_outputs": len(grouped),
        "expected_outputs": sum(len(solutions[key]) for key in task_keys),
        "candidate_samples": sample_count,
        "valid_candidate_samples": valid_samples,
        "invalid_candidate_samples": sample_count - valid_samples,
        "unique_valid_grids": unique_valid,
        "rows": by_basekey,
    }


def compare_runs(
    baseline_dir: Path,
    repaired_dir: Path,
    solutions: dict,
    task_keys: list[str],
) -> tuple[dict, list[dict]]:
    baseline = _run_summary(baseline_dir, solutions, task_keys)
    repaired = _run_summary(repaired_dir, solutions, task_keys)
    rows = []
    for task in task_keys:
        weight = 1.0 / len(solutions[task])
        for output_index in range(len(solutions[task])):
            basekey = f"{task}_{output_index}"
            before = baseline["rows"][basekey]
            after = repaired["rows"][basekey]
            rows.append(
                {
                    "task": task,
                    "basekey": basekey,
                    "weight": weight,
                    "baseline_selected": before["kgmon_hit"],
                    "repaired_selected": after["kgmon_hit"],
                    "selected_change": int(after["kgmon_hit"]) - int(before["kgmon_hit"]),
                    "baseline_oracle": before["oracle_hit"],
                    "repaired_oracle": after["oracle_hit"],
                    "oracle_change": int(after["oracle_hit"]) - int(before["oracle_hit"]),
                    "baseline_candidates": before["candidates"],
                    "repaired_candidates": after["candidates"],
                    "baseline_unique_grids": before["unique_grids"],
                    "repaired_unique_grids": after["unique_grids"],
                }
            )

    summary = {
        "task_keys": task_keys,
        "tasks": len(task_keys),
        "outputs": len(rows),
        "baseline": {key: value for key, value in baseline.items() if key != "rows"},
        "repaired": {key: value for key, value in repaired.items() if key != "rows"},
        "selected_gains": [row["basekey"] for row in rows if row["selected_change"] > 0],
        "selected_losses": [row["basekey"] for row in rows if row["selected_change"] < 0],
        "oracle_gains": [row["basekey"] for row in rows if row["oracle_change"] > 0],
        "oracle_losses": [row["basekey"] for row in rows if row["oracle_change"] < 0],
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--repaired-dir", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--task-keys", required=True, help="JSON list")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    solutions = json.loads(args.solutions.read_text())
    task_keys = json.loads(args.task_keys)
    summary, rows = compare_runs(
        args.baseline_dir,
        args.repaired_dir,
        solutions,
        task_keys,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
