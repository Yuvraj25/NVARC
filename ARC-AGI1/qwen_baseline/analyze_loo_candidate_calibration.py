import argparse
import bz2
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def hashable(grid):
    return tuple(map(tuple, np.asarray(grid).tolist()))


def source_view(filename):
    """Return the spatial/color view while ignoring demonstration order."""
    parts = filename.split(".")[1:]
    return ".".join(part for part in parts if not part.startswith("ex"))


def load_candidate_pool(candidate_dir, wanted_tasks):
    wanted = set(wanted_tasks)
    grouped = defaultdict(lambda: defaultdict(list))
    views = defaultdict(set)
    for path in sorted(Path(candidate_dir).iterdir()):
        if not path.is_file():
            continue
        basekey = path.name.split(".", 1)[0]
        task = basekey.rsplit("_", 1)[0]
        if task not in wanted:
            continue
        view = source_view(path.name)
        views[basekey].add(view)
        with bz2.BZ2File(path, "rb") as handle:
            samples = pickle.load(handle)
        for sample in samples:
            grid = hashable(sample["solution"])
            grouped[basekey][grid].append(
                {
                    "source_name": path.name,
                    "source_view": view,
                    "beam_score": float(sample["beam_score"]),
                    "score_aug": np.asarray(sample["score_aug"], dtype=float),
                }
            )
    return grouped, views


def standardized_weights(quality, beta):
    keys = sorted(quality)
    if not keys:
        return {}
    values = np.asarray([quality[key] for key in keys], dtype=float)
    std = float(values.std())
    if std < 1e-12 or beta == 0:
        weights = np.ones_like(values)
    else:
        zscore = (values - values.mean()) / std
        weights = np.exp(np.clip(beta * zscore, -3.0, 3.0))
        weights /= weights.mean()
    return {key: float(weight) for key, weight in zip(keys, weights)}


def rank_groups(groups, source_weights=None, rescore_weights=None):
    rows = []
    for grid, occurrences in groups.items():
        support = float(
            sum(
                1.0 if source_weights is None else source_weights[row["source_view"]]
                for row in occurrences
            )
        )
        per_occurrence_nll = []
        for occurrence in occurrences:
            scores = occurrence["score_aug"]
            if rescore_weights is None:
                per_occurrence_nll.append(float(scores.mean()))
            else:
                if len(scores) != len(rescore_weights):
                    raise ValueError(
                        f"Expected {len(rescore_weights)} rescore values, got {len(scores)}"
                    )
                per_occurrence_nll.append(
                    float(np.dot(scores, rescore_weights) / np.sum(rescore_weights))
                )
        aug_nll = float(np.mean(per_occurrence_nll))
        rows.append(
            {
                "grid": grid,
                "score": support - aug_nll,
                "support": support,
                "aug_nll": aug_nll,
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def exact_rank(ranked, target):
    return next((index for index, row in enumerate(ranked, 1) if row["grid"] == target), None)


def derive_calibration(groups, all_views, correct_grid, beta, variant):
    correct_occurrences = groups.get(correct_grid, [])
    source_quality = {
        view: float(any(row["source_view"] == view for row in correct_occurrences))
        for view in sorted(all_views)
    }
    source_weights = None
    if variant in {"source", "combined"}:
        source_weights = standardized_weights(source_quality, beta)

    rescore_weights = None
    if variant in {"rescore", "combined"}:
        if not correct_occurrences:
            return None
        lengths = {len(row["score_aug"]) for row in correct_occurrences}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent rescore lengths for correct grid: {lengths}")
        correct_nll = np.mean(
            np.stack([row["score_aug"] for row in correct_occurrences]),
            axis=0,
        )
        rescore_quality = {index: -float(value) for index, value in enumerate(correct_nll)}
        weight_map = standardized_weights(rescore_quality, beta)
        rescore_weights = np.asarray([weight_map[index] for index in range(len(correct_nll))])

    return {
        "variant": variant,
        "beta": beta,
        "source_weights": source_weights,
        "rescore_weights": rescore_weights,
    }


def choose_calibration(groups, all_views, correct_grid, betas=DEFAULT_BETAS):
    baseline_ranked = rank_groups(groups)
    baseline_rank = exact_rank(baseline_ranked, correct_grid)
    baseline = {
        "variant": "baseline",
        "beta": 0.0,
        "source_weights": None,
        "rescore_weights": None,
        "loo_rank": baseline_rank,
    }
    if baseline_rank is not None and baseline_rank <= 2:
        return baseline

    configurations = [baseline]
    for beta in betas:
        for variant in ("source", "rescore", "combined"):
            config = derive_calibration(groups, all_views, correct_grid, beta, variant)
            if config is None:
                continue
            ranked = rank_groups(
                groups,
                source_weights=config["source_weights"],
                rescore_weights=config["rescore_weights"],
            )
            config["loo_rank"] = exact_rank(ranked, correct_grid)
            configurations.append(config)

    variant_order = {"baseline": 0, "source": 1, "rescore": 2, "combined": 3}

    def preference(config):
        rank = config["loo_rank"]
        top2_failure = rank is None or rank > 2
        rank_value = math.inf if rank is None else rank
        # Top-two success is the objective. Within that bucket, prefer the
        # smallest perturbation instead of optimizing C's exact rank further.
        return (
            top2_failure,
            rank_value if top2_failure else 0,
            config["beta"],
            variant_order[config["variant"]],
        )

    return min(configurations, key=preference)


def evaluate(
    calibration_groups,
    calibration_views,
    target_groups,
    target_views,
    heldout_outputs,
    target_solutions,
    challenges,
    task_keys,
    betas=DEFAULT_BETAS,
):
    baseline_total = 0.0
    calibrated_total = 0.0
    oracle_total = 0.0
    heldout_oracle_outputs = 0
    rows = []

    for task in task_keys:
        num_outputs = len(challenges[task]["test"])
        weight = 1.0 / num_outputs
        heldout_grid = hashable(heldout_outputs[task])
        for output_index in range(num_outputs):
            basekey = f"{task}_{output_index}"
            loo_groups = calibration_groups.get(basekey, {})
            loo_views = calibration_views.get(basekey, set())
            target = hashable(target_solutions[task][output_index])
            groups = target_groups.get(basekey, {})

            target_source_views = {
                occurrence["source_view"]
                for occurrences in groups.values()
                for occurrence in occurrences
            }
            missing_views = sorted(target_source_views - set(loo_views))
            if missing_views:
                raise ValueError(
                    f"Calibration views do not cover target views for {basekey}: {missing_views}"
                )

            config = choose_calibration(loo_groups, loo_views, heldout_grid, betas=betas)
            baseline_ranked = rank_groups(groups)
            calibrated_ranked = rank_groups(
                groups,
                source_weights=config["source_weights"],
                rescore_weights=config["rescore_weights"],
            )
            baseline_rank = exact_rank(baseline_ranked, target)
            calibrated_rank = exact_rank(calibrated_ranked, target)
            baseline_hit = baseline_rank is not None and baseline_rank <= 2
            calibrated_hit = calibrated_rank is not None and calibrated_rank <= 2
            oracle_hit = target in groups
            loo_oracle_hit = heldout_grid in loo_groups

            baseline_total += weight if baseline_hit else 0.0
            calibrated_total += weight if calibrated_hit else 0.0
            oracle_total += weight if oracle_hit else 0.0
            heldout_oracle_outputs += int(loo_oracle_hit)
            rows.append(
                {
                    "task": task,
                    "basekey": basekey,
                    "weight": weight,
                    "loo_candidates": sum(len(value) for value in loo_groups.values()),
                    "loo_unique_grids": len(loo_groups),
                    "loo_oracle_hit": loo_oracle_hit,
                    "loo_baseline_rank": exact_rank(rank_groups(loo_groups), heldout_grid),
                    "loo_selected_rank": config["loo_rank"],
                    "selected_variant": config["variant"],
                    "selected_beta": config["beta"],
                    "source_view_coverage": (
                        len(target_source_views & set(loo_views)) / max(len(target_source_views), 1)
                    ),
                    "target_candidates": sum(len(value) for value in groups.values()),
                    "target_unique_grids": len(groups),
                    "target_baseline_rank": baseline_rank,
                    "target_calibrated_rank": calibrated_rank,
                    "baseline_top2_hit": baseline_hit,
                    "calibrated_top2_hit": calibrated_hit,
                    "gain": (not baseline_hit) and calibrated_hit,
                    "regression": baseline_hit and (not calibrated_hit),
                    "target_oracle_hit": oracle_hit,
                }
            )

    return {
        "tasks": len(task_keys),
        "outputs": len(rows),
        "baseline_top2": baseline_total,
        "calibrated_top2": calibrated_total,
        "target_oracle": oracle_total,
        "heldout_oracle_outputs": heldout_oracle_outputs,
        "gains": [row["basekey"] for row in rows if row["gain"]],
        "regressions": [row["basekey"] for row in rows if row["regression"]],
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-candidate-dir", required=True)
    parser.add_argument("--target-candidate-dir", required=True)
    parser.add_argument("--heldout-labels", required=True)
    parser.add_argument("--target-solutions", required=True)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--keys-json", required=True)
    parser.add_argument("--betas", default=",".join(map(str, DEFAULT_BETAS)))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    task_keys = json.loads(args.keys_json)
    labels = json.loads(Path(args.heldout_labels).read_text())
    heldout_outputs = {row["task"]: row["output"] for row in labels.values()}
    target_solutions = json.loads(Path(args.target_solutions).read_text())
    challenges = json.loads(Path(args.challenges).read_text())
    betas = tuple(float(value) for value in args.betas.split(","))

    calibration_groups, calibration_views = load_candidate_pool(
        args.calibration_candidate_dir, task_keys
    )
    target_groups, target_views = load_candidate_pool(args.target_candidate_dir, task_keys)
    result = evaluate(
        calibration_groups,
        calibration_views,
        target_groups,
        target_views,
        heldout_outputs,
        target_solutions,
        challenges,
        task_keys,
        betas=betas,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "loo_candidate_calibration_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    with open(output_dir / "loo_candidate_calibration_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])

    print(
        f"LOO_RESULT baseline_top2={result['baseline_top2']:.12g} "
        f"calibrated_top2={result['calibrated_top2']:.12g} "
        f"target_oracle={result['target_oracle']:.12g} "
        f"heldout_oracle_outputs={result['heldout_oracle_outputs']}/{result['outputs']}"
    )
    print("LOO_GAINS", result["gains"])
    print("LOO_REGRESSIONS", result["regressions"])
    for row in result["rows"]:
        print(
            "LOO_ROW",
            row["basekey"],
            f"C_oracle={int(row['loo_oracle_hit'])}",
            f"C_rank={row['loo_baseline_rank']}->{row['loo_selected_rank']}",
            f"selector={row['selected_variant']}@{row['selected_beta']}",
            f"test_rank={row['target_baseline_rank']}->{row['target_calibrated_rank']}",
            f"test_hit={int(row['baseline_top2_hit'])}->{int(row['calibrated_top2_hit'])}",
            f"oracle={int(row['target_oracle_hit'])}",
        )


if __name__ == "__main__":
    main()
