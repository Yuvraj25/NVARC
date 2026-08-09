#!/usr/bin/env python3
"""Audit whether valid ARC candidates recover after their first gold divergence."""

import argparse
import bz2
import csv
import difflib
import json
import pickle
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np


NEWLINE = 10
EOS = 15
MISSING = -1


def grid_tokens(grid):
    grid = np.asarray(grid)
    tokens = []
    for row_index, row in enumerate(grid):
        tokens.extend(int(value) for value in row)
        if row_index + 1 < len(grid):
            tokens.append(NEWLINE)
    tokens.append(EOS)
    return tokens


def token_type(token):
    if token == MISSING:
        return "missing"
    if token == NEWLINE:
        return "newline"
    if token == EOS:
        return "eos"
    if 0 <= token <= 9:
        return "digit"
    return "illegal"


def first_divergence(candidate, gold):
    for index, (left, right) in enumerate(zip_longest(candidate, gold, fillvalue=MISSING)):
        if left != right:
            return index
    return None


def boolean_spans(flags):
    spans = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    return spans


def common_suffix_length(left, right):
    count = 0
    for left_token, right_token in zip(reversed(left), reversed(right)):
        if left_token != right_token:
            break
        count += 1
    return count


def classify_structural_divergence(candidate_token, gold_token):
    candidate_type = token_type(candidate_token)
    gold_type = token_type(gold_token)
    if candidate_type == "eos" and gold_type != "eos":
        return "early_eos"
    if gold_type == "eos" and candidate_type != "eos":
        return "late_eos_extra_output"
    if candidate_type == "newline" and gold_type == "digit":
        return "early_newline_short_row"
    if candidate_type == "digit" and gold_type == "newline":
        return "late_newline_long_row"
    if candidate_type == "digit" and gold_type == "digit":
        return "digit_substitution_before_shape_change"
    return f"{candidate_type}_instead_of_{gold_type}"


def analyze_grid(candidate, gold):
    candidate = np.asarray(candidate)
    gold = np.asarray(gold)
    candidate_tokens = grid_tokens(candidate)
    gold_tokens = grid_tokens(gold)
    divergence = first_divergence(candidate_tokens, gold_tokens)
    shape_equal = candidate.shape == gold.shape
    exact = shape_equal and np.array_equal(candidate, gold)

    result = {
        "candidate_rows": int(candidate.shape[0]),
        "candidate_cols": int(candidate.shape[1]),
        "gold_rows": int(gold.shape[0]),
        "gold_cols": int(gold.shape[1]),
        "shape_equal": shape_equal,
        "exact": exact,
        "candidate_tokens": len(candidate_tokens),
        "gold_tokens": len(gold_tokens),
        "first_divergence_token": divergence,
        "first_candidate_token_type": None,
        "first_gold_token_type": None,
        "raw_post_divergence_match_fraction": None,
        "raw_error_spans": 0,
        "raw_first_error_span_tokens": 0,
        "raw_equal_length_suffix_tokens": 0,
        "end_aligned_suffix_content_tokens": 0,
        "end_aligned_suffix_content_fraction": 0.0,
        "post_divergence_sequence_match_ratio": None,
        "post_divergence_longest_match_tokens": None,
        "first_wrong_row": None,
        "first_wrong_col": None,
        "wrong_cells": None,
        "cell_error_spans": None,
        "first_cell_error_span": None,
        "immediate_next_cell_correct": None,
        "post_first_wrong_cell_match_fraction": None,
        "correct_cell_suffix": None,
        "next_wrong_cell_row_distance": None,
        "next_wrong_cell_col_distance": None,
        "next_wrong_cell_manhattan_distance": None,
        "next_wrong_cell_chebyshev_distance": None,
        "next_wrong_cell_same_row": None,
        "next_wrong_cell_same_col": None,
    }
    if exact:
        result["category"] = "exact"
        return result

    assert divergence is not None
    candidate_at_divergence = candidate_tokens[divergence] if divergence < len(candidate_tokens) else MISSING
    gold_at_divergence = gold_tokens[divergence] if divergence < len(gold_tokens) else MISSING
    result["first_candidate_token_type"] = token_type(candidate_at_divergence)
    result["first_gold_token_type"] = token_type(gold_at_divergence)

    raw_flags = [
        left != right
        for left, right in zip_longest(candidate_tokens, gold_tokens, fillvalue=MISSING)
    ]
    raw_spans = boolean_spans(raw_flags)
    post_flags = raw_flags[divergence + 1 :]
    result["raw_post_divergence_match_fraction"] = (
        float(1.0 - np.mean(post_flags)) if post_flags else None
    )
    result["raw_error_spans"] = len(raw_spans)
    result["raw_first_error_span_tokens"] = raw_spans[0][1] - raw_spans[0][0]
    if len(candidate_tokens) == len(gold_tokens):
        result["raw_equal_length_suffix_tokens"] = common_suffix_length(candidate_tokens, gold_tokens)

    candidate_content = candidate_tokens[:-1]
    gold_content = gold_tokens[:-1]
    end_suffix = common_suffix_length(candidate_content, gold_content)
    result["end_aligned_suffix_content_tokens"] = end_suffix
    result["end_aligned_suffix_content_fraction"] = (
        end_suffix / min(len(candidate_content), len(gold_content))
        if candidate_content and gold_content
        else 0.0
    )

    candidate_post = candidate_tokens[divergence + 1 : -1]
    gold_post = gold_tokens[divergence + 1 : -1]
    matcher = difflib.SequenceMatcher(None, candidate_post, gold_post, autojunk=False)
    blocks = matcher.get_matching_blocks()
    result["post_divergence_sequence_match_ratio"] = matcher.ratio()
    result["post_divergence_longest_match_tokens"] = max((block.size for block in blocks), default=0)

    if shape_equal:
        candidate_cells = candidate.reshape(-1)
        gold_cells = gold.reshape(-1)
        wrong = candidate_cells != gold_cells
        wrong_indices = np.flatnonzero(wrong)
        first_wrong = int(wrong_indices[0])
        cell_spans = boolean_spans(wrong.tolist())
        post_cell_matches = ~wrong[first_wrong + 1 :]
        suffix = common_suffix_length(candidate_cells.tolist(), gold_cells.tolist())
        result.update(
            {
                "category": "same_shape_single_cell" if len(wrong_indices) == 1 else "same_shape_multiple_cells",
                "first_wrong_row": first_wrong // gold.shape[1],
                "first_wrong_col": first_wrong % gold.shape[1],
                "wrong_cells": int(len(wrong_indices)),
                "cell_error_spans": len(cell_spans),
                "first_cell_error_span": cell_spans[0][1] - cell_spans[0][0],
                "immediate_next_cell_correct": bool(post_cell_matches[0]) if len(post_cell_matches) else None,
                "post_first_wrong_cell_match_fraction": (
                    float(np.mean(post_cell_matches)) if len(post_cell_matches) else None
                ),
                "correct_cell_suffix": suffix,
            }
        )
        if len(wrong_indices) > 1:
            first_coordinates = np.asarray([first_wrong // gold.shape[1], first_wrong % gold.shape[1]])
            next_wrong = int(wrong_indices[1])
            next_coordinates = np.asarray([next_wrong // gold.shape[1], next_wrong % gold.shape[1]])
            row_distance, col_distance = map(int, np.abs(next_coordinates - first_coordinates))
            result.update(
                {
                    "next_wrong_cell_row_distance": row_distance,
                    "next_wrong_cell_col_distance": col_distance,
                    "next_wrong_cell_manhattan_distance": row_distance + col_distance,
                    "next_wrong_cell_chebyshev_distance": max(row_distance, col_distance),
                    "next_wrong_cell_same_row": row_distance == 0,
                    "next_wrong_cell_same_col": col_distance == 0,
                }
            )
    else:
        result["category"] = "wrong_shape_" + classify_structural_divergence(
            candidate_at_divergence, gold_at_divergence
        )
    return result


def solution_key(solution):
    array = np.asarray(solution)
    return tuple(map(tuple, array.tolist()))


def candidate_score(samples):
    augmentation_nll = np.mean([np.mean(sample["score_aug"]) for sample in samples])
    return len(samples) - augmentation_nll


def load_candidates(candidate_dir):
    grouped = defaultdict(lambda: defaultdict(list))
    source_files = Counter()
    for path in sorted(Path(candidate_dir).iterdir()):
        if not path.is_file():
            continue
        output_key = path.name.split(".", 1)[0]
        try:
            with bz2.BZ2File(path, "rb") as source:
                samples = pickle.load(source)
        except (OSError, EOFError, pickle.UnpicklingError) as error:
            raise RuntimeError(f"Could not decode candidate file {path}") from error
        source_files[output_key] += 1
        for sample in samples:
            grid = np.asarray(sample["solution"])
            if grid.ndim != 2 or not all(0 < size <= 30 for size in grid.shape):
                continue
            if not np.all((0 <= grid) & (grid <= 9)):
                continue
            grouped[output_key][solution_key(grid)].append(sample)
    return grouped, source_files


def weighted_fraction(rows, predicate, weight_field):
    denominator = sum(row[weight_field] for row in rows)
    if not denominator:
        return None
    return sum(row[weight_field] for row in rows if predicate(row)) / denominator


def weighted_quantiles(rows, value_field, weight_field):
    expanded = [
        float(row[value_field])
        for row in rows
        for _ in range(int(row[weight_field]))
        if row[value_field] is not None
    ]
    if not expanded:
        return None
    return {
        name: float(np.quantile(expanded, quantile))
        for name, quantile in [("p10", 0.1), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9)]
    }


def summarize(rows, all_solution_keys, covered_output_keys, source_files):
    wrong = [row for row in rows if not row["exact"]]
    same_shape = [row for row in wrong if row["shape_equal"]]
    wrong_shape = [row for row in wrong if not row["shape_equal"]]
    selected = [row for row in rows if row["selected_top2"]]

    def population(values, weight_field):
        erroneous = [row for row in values if not row["exact"]]
        same = [row for row in erroneous if row["shape_equal"]]
        shaped = [row for row in erroneous if not row["shape_equal"]]
        same_with_post = [
            row for row in same if row["post_first_wrong_cell_match_fraction"] is not None
        ]
        immediate_then_later = [
            row
            for row in same_with_post
            if row["immediate_next_cell_correct"] and row["wrong_cells"] > 1
        ]
        return {
            "candidates": sum(row[weight_field] for row in values),
            "exact_fraction": weighted_fraction(values, lambda row: row["exact"], weight_field),
            "same_shape_wrong_fraction_of_errors": weighted_fraction(
                erroneous, lambda row: row["shape_equal"], weight_field
            ),
            "wrong_shape_fraction_of_errors": weighted_fraction(
                erroneous, lambda row: not row["shape_equal"], weight_field
            ),
            "same_shape_single_cell_fraction": weighted_fraction(
                same, lambda row: row["wrong_cells"] == 1, weight_field
            ),
            "same_shape_immediate_next_cell_recovery_fraction": weighted_fraction(
                [row for row in same if row["immediate_next_cell_correct"] is not None],
                lambda row: row["immediate_next_cell_correct"],
                weight_field,
            ),
            "same_shape_at_least_90pct_post_error_match_fraction": weighted_fraction(
                [row for row in same if row["post_first_wrong_cell_match_fraction"] is not None],
                lambda row: row["post_first_wrong_cell_match_fraction"] >= 0.9,
                weight_field,
            ),
            "same_shape_below_50pct_post_error_match_fraction": weighted_fraction(
                [row for row in same if row["post_first_wrong_cell_match_fraction"] is not None],
                lambda row: row["post_first_wrong_cell_match_fraction"] < 0.5,
                weight_field,
            ),
            "wrong_shape_at_least_90pct_sequence_match_fraction": weighted_fraction(
                shaped,
                lambda row: row["post_divergence_sequence_match_ratio"] >= 0.9,
                weight_field,
            ),
            "same_shape_recovery": {
                "errors": sum(row[weight_field] for row in same),
                "errors_with_a_following_cell": sum(row[weight_field] for row in same_with_post),
                "single_cell_error_fraction": weighted_fraction(
                    same, lambda row: row["wrong_cells"] == 1, weight_field
                ),
                "first_error_span_is_one_cell_fraction": weighted_fraction(
                    same, lambda row: row["first_cell_error_span"] == 1, weight_field
                ),
                "immediate_next_cell_recovery_fraction": weighted_fraction(
                    same_with_post, lambda row: row["immediate_next_cell_correct"], weight_field
                ),
                "one_error_span_then_permanent_recovery_fraction": weighted_fraction(
                    same,
                    lambda row: row["cell_error_spans"] == 1 and row["correct_cell_suffix"] > 0,
                    weight_field,
                ),
                "multiple_error_spans_fraction": weighted_fraction(
                    same, lambda row: row["cell_error_spans"] > 1, weight_field
                ),
                "post_first_error_match_fraction_quantiles": weighted_quantiles(
                    same_with_post, "post_first_wrong_cell_match_fraction", weight_field
                ),
                "wrong_cell_count_quantiles": weighted_quantiles(same, "wrong_cells", weight_field),
                "error_span_count_quantiles": weighted_quantiles(same, "cell_error_spans", weight_field),
                "immediate_recovery_then_later_error": {
                    "candidates": sum(row[weight_field] for row in immediate_then_later),
                    "next_error_same_row_fraction": weighted_fraction(
                        immediate_then_later, lambda row: row["next_wrong_cell_same_row"], weight_field
                    ),
                    "next_error_same_col_fraction": weighted_fraction(
                        immediate_then_later, lambda row: row["next_wrong_cell_same_col"], weight_field
                    ),
                    "next_error_same_row_or_col_fraction": weighted_fraction(
                        immediate_then_later,
                        lambda row: row["next_wrong_cell_same_row"] or row["next_wrong_cell_same_col"],
                        weight_field,
                    ),
                    "next_error_within_chebyshev_2_fraction": weighted_fraction(
                        immediate_then_later,
                        lambda row: row["next_wrong_cell_chebyshev_distance"] <= 2,
                        weight_field,
                    ),
                    "next_error_manhattan_distance_quantiles": weighted_quantiles(
                        immediate_then_later, "next_wrong_cell_manhattan_distance", weight_field
                    ),
                },
            },
            "wrong_shape_first_divergence_counts": dict(
                sorted(
                    Counter(
                        {
                            category: sum(
                                row[weight_field] for row in shaped if row["category"] == category
                            )
                            for category in {row["category"] for row in shaped}
                        }
                    ).items()
                )
            ),
        }

    category_counts = Counter(row["category"] for row in rows)
    structural_counts = Counter(row["category"] for row in wrong_shape)
    missing_outputs = sorted(all_solution_keys - covered_output_keys)
    return {
        "coverage": {
            "solution_outputs": len(all_solution_keys),
            "covered_outputs": len(covered_output_keys),
            "missing_outputs": missing_outputs,
            "covered_tasks": len({key.rsplit("_", 1)[0] for key in covered_output_keys}),
            "candidate_source_files": sum(source_files.values()),
            "unique_valid_grids": len(rows),
            "valid_grid_occurrences": sum(row["occurrences"] for row in rows),
        },
        "category_counts_unique": dict(sorted(category_counts.items())),
        "wrong_shape_first_divergence_counts_unique": dict(sorted(structural_counts.items())),
        "unique_grid_population": population(rows, "unique_weight"),
        "occurrence_weighted_population": population(rows, "occurrences"),
        "selected_top2_population": population(selected, "unique_weight"),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(candidate_dir, solutions_path, output_dir):
    solutions = json.loads(Path(solutions_path).read_text())
    labels = {
        f"{task}_{index}": np.asarray(grid)
        for task, outputs in solutions.items()
        for index, grid in enumerate(outputs)
    }
    grouped, source_files = load_candidates(candidate_dir)
    unexpected = sorted(set(grouped) - set(labels))
    if unexpected:
        raise KeyError(f"Candidate outputs missing from solution labels: {unexpected[:10]}")

    rows = []
    for output_key in sorted(grouped):
        scored = []
        for key, samples in grouped[output_key].items():
            score = float(candidate_score(samples))
            scored.append((score, key, samples))
        scored.sort(key=lambda item: item[0], reverse=True)
        for rank, (score, key, samples) in enumerate(scored, 1):
            metrics = analyze_grid(np.asarray(key), labels[output_key])
            rows.append(
                {
                    "output_key": output_key,
                    "rank_kgmon": rank,
                    "selected_top2": rank <= 2,
                    "kgmon_score": score,
                    "occurrences": len(samples),
                    "unique_weight": 1,
                    **metrics,
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_recovery_rows.csv", rows)
    summary = summarize(rows, set(labels), set(grouped), source_files)
    (output_dir / "candidate_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("rows_csv =", output_dir / "candidate_recovery_rows.csv")
    print("summary_json =", output_dir / "candidate_recovery_summary.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.candidate_dir, args.solutions, args.output_dir)
