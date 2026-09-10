import argparse
import bz2
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from arc_loader import ArcDataset


def grid_key(grid):
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def prompt_name(candidate_key):
    return candidate_key.rsplit(".out", 1)[0]


def geometry_name(prompt):
    return prompt.split(".permute", 1)[0]


def group_candidates(guesses, sources=None):
    grouped = defaultdict(list)
    for candidate in guesses.values():
        if sources is None or candidate["candidate_source"] in sources:
            grouped[grid_key(candidate["solution"])].append(candidate)
    return grouped


def rank_mean_nll(guesses, sources=None):
    ranked = []
    for key, occurrences in group_candidates(guesses, sources).items():
        scores = np.concatenate(
            [np.asarray(candidate["score_aug"], dtype=float) for candidate in occurrences]
        )
        ranked.append((float(scores.mean()), np.asarray(key)))
    return [grid for _, grid in sorted(ranked, key=lambda item: item[0])]


def rank_original_kgmon(guesses):
    ranked = []
    for key, occurrences in group_candidates(guesses, {"qwen"}).items():
        score = len(occurrences) - np.mean(
            [np.mean(candidate["qwen_original_score_aug"]) for candidate in occurrences]
        )
        ranked.append((float(score), np.asarray(key)))
    return [grid for _, grid in sorted(ranked, key=lambda item: item[0], reverse=True)]


def distinct(grids):
    result = []
    seen = set()
    for grid in grids:
        key = grid_key(grid)
        if key not in seen:
            seen.add(key)
            result.append(np.asarray(grid))
    return result


def select(policy, guesses):
    original = rank_original_kgmon(guesses)
    qwen_nll = rank_mean_nll(guesses, {"qwen"})
    union_nll = rank_mean_nll(guesses, {"qwen", "trm"})
    if policy == "qwen_original_kgmon":
        return original[:2]
    if policy == "qwen_mean_nll":
        return qwen_nll[:2]
    if policy == "qwen_trm_mean_nll":
        return union_nll[:2]
    if policy == "qwen_original_top1_then_union_mean_nll":
        return distinct(original[:1] + union_nll)[:2]
    raise ValueError(policy)


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "p975": float(np.quantile(values, 0.975)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--keys-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    args = parser.parse_args()

    keys = json.loads(args.keys_json)
    data = ArcDataset.from_file(args.challenges, keys=keys).load_replies(args.solutions)

    slot_by_prompt = {}
    prompts_by_geometry = defaultdict(list)
    for task in keys:
        puzzle = data.change_keys([task], keep_flags=True)
        expected = puzzle.split_multi_replies().augment(n=4, seed=2)
        for prompt in expected.keys:
            prompts_by_geometry[geometry_name(prompt)].append(prompt)
    for geometry, prompts in prompts_by_geometry.items():
        if len(prompts) != 4:
            raise RuntimeError(f"Expected four colour slots for {geometry}, got {len(prompts)}")
        for slot, prompt in enumerate(prompts):
            slot_by_prompt[prompt] = slot

    guesses_by_output = defaultdict(dict)
    for path in Path(args.candidate_dir).iterdir():
        with bz2.open(path, "rb") as handle:
            candidates = pickle.load(handle)
        output_key = path.name.split(".", 1)[0]
        for candidate_index, candidate in enumerate(candidates):
            candidate_key = f"{path.name}.out{candidate_index}"
            guesses_by_output[output_key][candidate_key] = candidate

    policies = [
        "qwen_original_kgmon",
        "qwen_mean_nll",
        "qwen_trm_mean_nll",
        "qwen_original_top1_then_union_mean_nll",
    ]
    outputs = []
    for task in keys:
        output_count = len(data.queries[task]["test"])
        for output_index in range(output_count):
            outputs.append({
                "task": task,
                "output_key": f"{task}_{output_index}",
                "weight": 1.0 / output_count,
                "gold": np.asarray(data.replies[task][output_index]),
            })

    def evaluate(omit_slots=None):
        scores = {policy: 0.0 for policy in policies}
        task_scores = {policy: defaultdict(float) for policy in policies}
        selections = {}
        for output in outputs:
            guesses = guesses_by_output[output["output_key"]]
            if omit_slots is not None:
                guesses = {
                    candidate_key: candidate
                    for candidate_key, candidate in guesses.items()
                    if candidate["candidate_source"] == "trm"
                    or slot_by_prompt[prompt_name(candidate_key)]
                    != omit_slots[geometry_name(prompt_name(candidate_key))]
                }
            for policy in policies:
                selected = select(policy, guesses)
                credit = output["weight"] * any(
                    np.array_equal(output["gold"], candidate) for candidate in selected
                )
                scores[policy] += credit
                task_scores[policy][output["task"]] += credit
                selections[(output["output_key"], policy)] = [grid_key(grid) for grid in selected]
        return scores, task_scores, selections

    full_scores, full_task_scores, full_selections = evaluate()
    changes = {}
    for policy in policies[1:]:
        gains = []
        losses = []
        for task in keys:
            delta = full_task_scores[policy][task] - full_task_scores["qwen_original_kgmon"][task]
            if delta > 0:
                gains.append({"task": task, "delta": delta})
            elif delta < 0:
                losses.append({"task": task, "delta": delta})
        changes[policy] = {"gains": gains, "losses": losses}

    trm_selection_counts = {}
    for policy in policies[1:]:
        count = 0
        for output in outputs:
            guesses = guesses_by_output[output["output_key"]]
            trm_grids = set(group_candidates(guesses, {"trm"}))
            count += sum(
                grid in trm_grids for grid in full_selections[(output["output_key"], policy)]
            )
        trm_selection_counts[policy] = count

    geometries = sorted(prompts_by_geometry)
    leave_one = []
    for omitted_slot in range(4):
        scores, _, _ = evaluate({geometry: omitted_slot for geometry in geometries})
        leave_one.append({"omitted_slot": omitted_slot, "scores": scores})

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = {policy: np.zeros(args.bootstrap_replicates) for policy in policies}
    for replicate in range(args.bootstrap_replicates):
        omit_slots = {geometry: int(rng.integers(4)) for geometry in geometries}
        scores, _, _ = evaluate(omit_slots)
        for policy, score in scores.items():
            bootstrap[policy][replicate] = score

    summary = {
        "inputs": {
            "validation_tasks": len(keys),
            "test_outputs": len(outputs),
            "candidate_files": sum(1 for _ in Path(args.candidate_dir).iterdir()),
        },
        "full_8x4": {
            "scores": full_scores,
            "changes_vs_original_qwen_kgmon": changes,
            "selected_trm_slots": trm_selection_counts,
        },
        "global_leave_one_slot_out_8x3": leave_one,
        "bootstrap_8x3": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "scores": {policy: summarize(values) for policy, values in bootstrap.items()},
            "delta_union_nll_vs_qwen_nll": summarize(
                bootstrap["qwen_trm_mean_nll"] - bootstrap["qwen_mean_nll"]
            ),
            "delta_union_nll_vs_original_qwen_kgmon": summarize(
                bootstrap["qwen_trm_mean_nll"] - bootstrap["qwen_original_kgmon"]
            ),
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
