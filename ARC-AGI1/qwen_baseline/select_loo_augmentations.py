import argparse
import bz2
import json
import pickle
from pathlib import Path

import numpy as np


def hashable(grid):
    return tuple(map(tuple, np.asarray(grid).tolist()))


def select_successful_augmentations(candidate_dir, labels, task_keys):
    label_by_task = {row["task"]: row for row in labels.values()}
    selected = {task: [] for task in task_keys}
    files_seen = {task: 0 for task in task_keys}
    valid_candidates = {task: 0 for task in task_keys}

    for path in sorted(Path(candidate_dir).iterdir()):
        if not path.is_file() or "." not in path.name:
            continue
        basekey, descriptor = path.name.split(".", 1)
        task, output_index = basekey.rsplit("_", 1)
        if task not in selected or output_index != "0":
            continue
        files_seen[task] += 1
        with bz2.BZ2File(path, "rb") as handle:
            candidates = pickle.load(handle)
        target = hashable(label_by_task[task]["output"])
        valid_candidates[task] += len(candidates)
        if any(hashable(candidate["solution"]) == target for candidate in candidates):
            selected[task].append(descriptor)

    for task in task_keys:
        selected[task] = sorted(set(selected[task]))
    stats = {
        task: {
            "source_files": files_seen[task],
            "valid_candidates": valid_candidates[task],
            "successful_augmentations": len(selected[task]),
        }
        for task in task_keys
    }
    return selected, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--keys-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    task_keys = json.loads(args.keys_json)
    selected, stats = select_successful_augmentations(
        args.candidate_dir,
        labels,
        task_keys,
    )
    Path(args.output).write_text(json.dumps(selected, indent=2, sort_keys=True))
    for task in task_keys:
        print(
            f"LOO_AUGMENTATIONS task={task} "
            f"source_files={stats[task]['source_files']} "
            f"valid_candidates={stats[task]['valid_candidates']} "
            f"successful={stats[task]['successful_augmentations']}"
        )
        for descriptor in selected[task]:
            print(f"LOO_SUCCESS task={task} descriptor={descriptor}")
    print(f"LOO_SELECTED_OUTPUT {args.output}")


if __name__ == "__main__":
    main()
