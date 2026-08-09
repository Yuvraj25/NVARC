# V17 valid-candidate recovery audit

## Question

For a retained valid v17 candidate, once its serialized output first diverges
from the evaluation label, does it return to the gold trajectory or does the
remaining output collapse?

This is a descriptive audit of retained DFS candidates. It does not claim that
the candidates are unrestricted-greedy rollouts.

## Inputs and coverage

- Candidate directory:
  `/Users/banna/kaggle/temp/kaggle_versions/v17_files/inference_outputs_validation`
- Evaluation labels:
  `/Users/banna/kaggle/temp/arc_prize_2026_data/arc-agi_evaluation_solutions.json`
- 2,265 candidate files
- 4,056 valid-grid occurrences
- 2,268 unique grids within output
- 165 of 172 evaluation outputs, spanning 118 of 120 tasks

Two tasks are entirely absent because every output has zero retained candidates:

```text
271d71e2
4e34c42c
```

Four other tasks are partially covered. The seven outputs absent from the
retained corpus are:

```text
271d71e2_0
4e34c42c_0
4e34c42c_1
5dbc8537_0
6e4f6532_0
cbebaa4b_0
f560132c_0
```

The audit's `score_kgmon` reconstruction reproduces the existing v17 audit:
61 oracle-hit outputs and 49 top-two exact outputs.

## Definitions

Valid grids are serialized as digit tokens, newline tokens between rows, and a
terminal EOS token.

For same-shape candidates, recovery is measured directly over row-major cells:

- immediate recovery: the cell immediately after the first wrong cell is gold;
- post-error match fraction: fraction of all subsequent cells equal to gold;
- error span: one contiguous run of wrong cells;
- permanent recovery: one error span followed by a non-empty exact suffix.

For wrong-shape candidates, raw token alignment identifies the first structural
divergence. An additional `SequenceMatcher` ratio and end-aligned suffix are
reported to distinguish a local insertion/deletion from a broadly different
suffix. The edit-aligned number is diagnostic rather than a unique causal
alignment for repetitive grids.

## Main result

Among 1,966 unique, erroneous, correct-shape grids:

- 48.7% return to gold on the immediately following cell;
- 49.4% match at least 90% of all cells after the first error;
- only 7.7% match fewer than 50% of cells after the first error;
- the median post-error match fraction is 89.8%;
- the median candidate has 24 wrong cells in 12 separate error spans;
- 48.7% have a one-cell first error span;
- only 1.5% are true single-cell errors;
- only 2.5% have one error span followed by permanent recovery;
- 96.7% have multiple separate error spans.

The occurrence-weighted and selected-top-two populations are similar:

| Population | Immediate next cell | At least 90% post-error match | Below 50% post-error match | Multiple error spans |
|---|---:|---:|---:|---:|
| Unique grids | 48.7% | 49.4% | 7.7% | 96.7% |
| Occurrence weighted | 49.6% | 52.3% | 7.9% | 95.4% |
| Selected top two | 48.5% | 49.8% | 7.5% | 95.0% |

Across 151 outputs containing at least one such candidate, output-macro averages
are 46.5% immediate recovery, 47.8% at least-90% post-error matching, and 7.8%
below-50% post-error matching. The conclusion is therefore not explained only
by outputs with unusually many candidates.

The common failure pattern is neither a single isolated error nor total
autoregressive collapse. The candidate usually returns to matching gold, but
then makes additional localized errors later.

Of the 958 unique candidates that recover on the immediately following cell,
only 30 have no later error. The remaining 928 return immediately and then
diverge again. For those 928:

- 61.3% have the next error in the same row or column as the first;
- 60.6% have it within Chebyshev distance two;
- the median Manhattan distance to the next error is two cells;
- the 75th-percentile Manhattan distance is five cells.

The later error is therefore often, but not always, in the spatial vicinity of
the first. Across every multi-error candidate, including candidates whose first
error run is contiguous, 81.5% have the next wrong cell in the same row or
column and the median Manhattan distance is one.

## Wrong-shape candidates

There are 241 unique wrong-shape grids. Their first divergence is:

| First divergence | Unique grids |
|---|---:|
| Digit substitution before the later shape difference | 165 |
| Early newline / short row | 43 |
| Late newline / long row | 24 |
| Early EOS | 8 |
| Late EOS / extra output | 1 |

Only 5.8% of unique wrong-shape candidates have at least 90% post-divergence
sequence-match similarity after allowing alignment. Most wrong shapes are not
merely an otherwise-correct output with one misplaced structural token.

## Hypothesis to test for OPSD

These candidates come from SFT-trained models; this audit contains no OPSD
training and cannot establish how OPSD loss is distributed. It does argue
against assuming that every first error causes a catastrophic trajectory
departure: most later cells remain correct while errors recur in multiple
islands. That motivates measuring OPSD KL by position. Only if KL is also
concentrated at those error islands would uniform averaging be shown to dilute
their signal. Before increasing a global learning rate, measure the same-example
pre/post probability change at each divergence and the per-position KL profile.

## Reproduction

```bash
python analyze_candidate_recovery.py \
  --candidate-dir /Users/banna/kaggle/temp/kaggle_versions/v17_files/inference_outputs_validation \
  --solutions /Users/banna/kaggle/temp/arc_prize_2026_data/arc-agi_evaluation_solutions.json \
  --output-dir /Users/banna/kaggle/temp/kaggle_versions/v17_candidate_recovery
```

Generated artifacts:

- `candidate_recovery_rows.csv`: one row per unique grid within output;
- `candidate_recovery_summary.json`: coverage and population summaries.
