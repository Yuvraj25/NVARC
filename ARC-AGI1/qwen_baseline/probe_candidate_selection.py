import argparse
import bz2
import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def hashable(grid):
    return tuple(map(tuple, np.asarray(grid).tolist()))


def load_task_keys(value, solutions):
    if value is None:
        return sorted(solutions)
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def load_candidates(candidate_dir):
    grouped = defaultdict(lambda: defaultdict(list))
    for name in sorted(os.listdir(candidate_dir)):
        path = Path(candidate_dir) / name
        if not path.is_file():
            continue
        basekey = name.split(".")[0]
        with bz2.BZ2File(path) as handle:
            samples = pickle.load(handle)
        for sample in samples:
            grouped[basekey][hashable(sample["solution"])].append(sample)
    return grouped


def candidate_stats(samples):
    beam_scores = np.asarray([float(sample["beam_score"]) for sample in samples])
    augmented_nll = np.asarray([sample["score_aug"] for sample in samples], dtype=float)
    return {
        "support": len(samples),
        "beam_nll_mean": float(beam_scores.mean()),
        "beam_nll_min": float(beam_scores.min()),
        "aug_nll_mean": float(augmented_nll.mean()),
        "aug_nll_sum_mean": float(augmented_nll.sum(axis=1).mean()),
    }


def kgmon_score(stats):
    return stats["support"] - stats["aug_nll_mean"]


def probmul_score(samples):
    generation = sum(3.0 - float(sample["beam_score"]) for sample in samples)
    rescoring = np.mean(
        [sum(3.0 - float(score) for score in sample["score_aug"]) for sample in samples]
    )
    return float(generation + rescoring)


def rank_candidates(groups, method):
    candidates = []
    for grid, samples in groups.items():
        stats = candidate_stats(samples)
        candidates.append({"grid": grid, "samples": samples, "stats": stats})

    if method == "kgmon":
        return sorted(candidates, key=lambda row: kgmon_score(row["stats"]), reverse=True)
    if method == "probmul":
        return sorted(candidates, key=lambda row: probmul_score(row["samples"]), reverse=True)
    if method == "kgmon_beam_second":
        consensus = sorted(candidates, key=lambda row: kgmon_score(row["stats"]), reverse=True)
        if len(consensus) < 2:
            return consensus
        first = consensus[0]
        remaining = sorted(
            consensus[1:],
            key=lambda row: (-row["stats"]["beam_nll_mean"], kgmon_score(row["stats"])),
            reverse=True,
        )
        return [first, remaining[0], *[row for row in consensus[1:] if row is not remaining[0]]]
    raise ValueError(f"Unknown method: {method}")


def evaluate(grouped, solutions, task_keys):
    methods = ["kgmon", "probmul", "kgmon_beam_second"]
    totals = {method: 0.0 for method in methods}
    oracle_total = 0.0
    rows = []

    for task in task_keys:
        task_solutions = solutions[task]
        output_weight = 1.0 / len(task_solutions)
        for output_index, target_grid in enumerate(task_solutions):
            basekey = f"{task}_{output_index}"
            target = hashable(target_grid)
            groups = grouped.get(basekey, {})
            oracle_hit = target in groups
            if oracle_hit:
                oracle_total += output_weight

            row = {
                "task": task,
                "basekey": basekey,
                "candidates": sum(len(samples) for samples in groups.values()),
                "unique_grids": len(groups),
                "oracle_hit": oracle_hit,
            }
            for method in methods:
                ranked = rank_candidates(groups, method)
                hit = any(candidate["grid"] == target for candidate in ranked[:2])
                row[f"{method}_hit"] = hit
                if hit:
                    totals[method] += output_weight
            rows.append(row)

    return totals, oracle_total, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument(
        "--task-keys",
        default=None,
        help="JSON list or path to a JSON list. Defaults to every task in the solutions file.",
    )
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    with open(args.solutions) as handle:
        solutions = json.load(handle)
    task_keys = load_task_keys(args.task_keys, solutions)
    grouped = load_candidates(args.candidate_dir)
    totals, oracle_total, rows = evaluate(grouped, solutions, task_keys)

    print(f"tasks={len(task_keys)} outputs={len(rows)} decoded_outputs={len(grouped)}")
    for method, score in totals.items():
        print(f"{method}: {score:.12g}")
    print(f"oracle: {oracle_total:.12g}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
