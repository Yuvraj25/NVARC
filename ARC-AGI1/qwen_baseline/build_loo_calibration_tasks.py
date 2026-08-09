import argparse
import copy
import json
from pathlib import Path


DEFAULT_TASKS = [
    "142ca369",
    "20270e3b",
    "20a9e565",
    "28a6681f",
    "62593bfd",
]


def build_loo_tasks(challenges, task_keys, holdout="last"):
    loo_challenges = {}
    labels = {}

    for task_key in task_keys:
        if task_key not in challenges:
            raise KeyError(f"Unknown task: {task_key}")
        task = challenges[task_key]
        train = task.get("train", [])
        if len(train) < 2:
            raise ValueError(f"Task {task_key} needs at least two training pairs")

        if holdout == "last":
            holdout_indices = [len(train) - 1]
        elif holdout == "all":
            holdout_indices = list(range(len(train)))
        else:
            raise ValueError(f"Unsupported holdout mode: {holdout}")

        for heldout_index in holdout_indices:
            heldout = train[heldout_index]
            loo_key = f"{task_key}l{heldout_index}"
            if loo_key in loo_challenges:
                raise ValueError(f"Duplicate leave-one-out key: {loo_key}")

            loo_challenges[loo_key] = {
                "train": copy.deepcopy(
                    [pair for index, pair in enumerate(train) if index != heldout_index]
                ),
                "test": [{"input": copy.deepcopy(heldout["input"])}],
            }
            labels[loo_key] = {
                "task": task_key,
                "heldout_index": heldout_index,
                "num_original_train_pairs": len(train),
                "input": copy.deepcopy(heldout["input"]),
                "output": copy.deepcopy(heldout["output"]),
            }

    return loo_challenges, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--output-challenges", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--tasks-json", default=json.dumps(DEFAULT_TASKS))
    parser.add_argument("--holdout", choices=["last", "all"], default="last")
    args = parser.parse_args()

    challenges = json.loads(Path(args.challenges).read_text())
    task_keys = json.loads(args.tasks_json)
    if not isinstance(task_keys, list) or not all(isinstance(key, str) for key in task_keys):
        raise ValueError("--tasks-json must decode to a list of task keys")

    loo_challenges, labels = build_loo_tasks(
        challenges,
        task_keys,
        holdout=args.holdout,
    )
    Path(args.output_challenges).write_text(json.dumps(loo_challenges))
    Path(args.output_labels).write_text(json.dumps(labels, indent=2, sort_keys=True))

    print(f"source_tasks={len(task_keys)} loo_tasks={len(loo_challenges)} holdout={args.holdout}")
    for loo_key, label in labels.items():
        print(
            f"{loo_key}: task={label['task']} heldout={label['heldout_index']} "
            f"train_pairs={label['num_original_train_pairs'] - 1}"
        )


if __name__ == "__main__":
    main()
