"""Selection helpers for deliberately shared multi-output augmentation views."""

from __future__ import annotations

import bz2
from collections import defaultdict
from itertools import product
import math
from pathlib import Path
import pickle

import numpy as np

from arc_decoder import ArcDecoder, hashable, score_kgmon


def _load_view_records(candidate_dir):
    records = defaultdict(lambda: defaultdict(dict))
    for path in sorted(Path(candidate_dir).iterdir()):
        if not path.is_file() or "." not in path.name:
            continue
        base_key, view = path.name.split(".", 1)
        puzzle_key, output_text = base_key.rsplit("_", 1)
        with bz2.BZ2File(path, "rb") as source:
            samples = pickle.load(source)
        if samples:
            records[puzzle_key][int(output_text)][view] = samples
    return records


def _unique_view_candidates(samples):
    unique = {}
    for sample in samples:
        grid = hashable(sample["solution"])
        if grid not in unique or sample["beam_score"] < unique[grid]["beam_score"]:
            unique[grid] = sample
    return unique


def _equal_tuple_support(puzzle_records, output_count):
    if not output_count:
        return {}
    complete_views = sorted(
        set.intersection(
            *[
                set(puzzle_records.get(output_index, {}))
                for output_index in range(output_count)
            ]
        )
    )
    support = defaultdict(float)
    for view in complete_views:
        choices = [
            list(_unique_view_candidates(puzzle_records[output_index][view]))
            for output_index in range(output_count)
        ]
        mass = 1.0 / math.prod(len(output_choices) for output_choices in choices)
        for grids in product(*choices):
            support[grids] += mass
    return support


def _local_scores(decoder):
    scores = {}
    arrays = {}
    unique_counts = {}
    for base_key, samples in decoder.decoded_results.items():
        grouped = defaultdict(list)
        for sample in samples.values():
            grid = hashable(sample["solution"])
            grouped[grid].append(sample)
            arrays[(base_key, grid)] = sample["solution"]
        unique_counts[base_key] = len(grouped)
        for grid, occurrences in grouped.items():
            mean_aug_nll = np.mean(
                [np.mean(sample["score_aug"]) for sample in occurrences]
            )
            scores[(base_key, grid)] = len(occurrences) - float(mean_aug_nll)
    return scores, arrays, unique_counts


def select_bounded_shared_support(
    data,
    split_data,
    candidate_dir,
    *,
    support_weight=2.0,
    n_guesses=2,
):
    """Rank primary candidates by KGMon plus a bounded shared-tuple bonus."""
    decoder = ArcDecoder(split_data, n_guesses=n_guesses)
    decoder.load_decoded_results(candidate_dir)
    independent = decoder.run_selection_algo(score_kgmon)
    local_scores, arrays, unique_counts = _local_scores(decoder)
    view_records = _load_view_records(candidate_dir)
    selected = {}

    for puzzle_key in data.keys:
        output_count = len(data.queries[puzzle_key]["test"])
        if output_count == 1:
            base_key = f"{puzzle_key}_0"
            selected[base_key] = list(independent.get(base_key, []))[:n_guesses]
            unique_counts.setdefault(base_key, 0)
            continue
        tuple_support = _equal_tuple_support(
            view_records.get(puzzle_key, {}), output_count
        )
        marginal_support = [defaultdict(float) for _ in range(output_count)]
        for grids, mass in tuple_support.items():
            for output_index, grid in enumerate(grids):
                marginal_support[output_index][grid] = max(
                    marginal_support[output_index][grid], mass
                )

        for output_index in range(output_count):
            base_key = f"{puzzle_key}_{output_index}"
            ranked = sorted(
                (
                    (grid, score)
                    for (candidate_key, grid), score in local_scores.items()
                    if candidate_key == base_key
                ),
                key=lambda item: (
                    item[1]
                    + support_weight
                    * math.log1p(marginal_support[output_index].get(item[0], 0.0)),
                    item[1],
                ),
                reverse=True,
            )
            guesses = [
                arrays[(base_key, grid)] for grid, _score in ranked[:n_guesses]
            ]
            for candidate in independent.get(base_key, []):
                if not any(np.array_equal(candidate, prior) for prior in guesses):
                    guesses.append(candidate)
                if len(guesses) == n_guesses:
                    break
            selected[base_key] = guesses
            unique_counts.setdefault(base_key, 0)

    return selected, unique_counts, decoder


def fill_missing_attempts(primary, combined, unique_counts, *, n_guesses=2):
    """Use adaptive candidates only when the primary pool lacks an attempt."""
    selected = {}
    for base_key in unique_counts:
        guesses = list(primary.get(base_key, []))[: min(unique_counts[base_key], n_guesses)]
        for candidate in combined.get(base_key, []):
            if not any(np.array_equal(candidate, prior) for prior in guesses):
                guesses.append(candidate)
            if len(guesses) == n_guesses:
                break
        selected[base_key] = guesses
    return selected
