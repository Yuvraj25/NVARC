# OPSD fixed-pool experiment

This is an unvalidated experiment. It does not replace the default ARC solver.
`full_sft` remains the default TTFT method, and the production selector remains
the mean-NLL `score_kgmon` selector.

## Execution design

- Unsloth continues to load, patch, and train the Qwen/PEFT model.
- The custom OPSD stage is a small PyTorch loop over the Unsloth-patched model.
- The base model remains frozen.
- Initial SFT physically removes one deterministic solved pair `C`, including
  every augmentation of its input and output.
- A frozen named teacher LoRA and a trainable student LoRA both start from the
  end of reduced SFT.
- The student samples legal ARC tokens. The teacher scores the same sampled
  prefix with an additional solved view of `C` in its prompt.
- The primary loss is exact reverse KL over the complete 16-token output head.
- Phase 1 only rescores the preserved v17 candidate pool. It never runs DFS.

Puzzles with fewer than three training pairs fall back to unchanged full SFT.

## Cohort

The 44 completed-result tasks contain:

- 32 eligible puzzles with at least three solved training pairs;
- 12 excluded two-pair puzzles.

Only 12 eligible puzzles contain an exact output in the v17 fixed candidate
pool. These are informative for Phase 1 because their exact candidate can move
into or out of the selected top two:

```text
142ca369  1818057f  1ae2feb7  20270e3b
269e22fb  28a6681f  2ba387bc  2d0172a1
36a08778  3dc255db  4a21e3da  58f5dbd5
```

They were split before observing any OPSD result:

```text
development:
142ca369  1818057f  269e22fb  2d0172a1  36a08778  3dc255db

confirmation:
1ae2feb7  20270e3b  28a6681f  2ba387bc  4a21e3da  58f5dbd5
```

The remaining 20 eligible puzzles have zero fixed-pool oracle and therefore
cannot produce a Phase-1 exact-score gain. They remain useful only for broader
distribution-shift diagnostics after the informative cohort shows promise.

## Methods

The launcher exposes:

```text
full_sft
reduced_sft
reduced_plus_sft_c
reduced_plus_opsd
```

`reduced_plus_sft_c` is the ordinary augmented-SFT correction control. The
initial implementation is step/dataset matched, not yet automatically
wall-clock matched. A claimed OPSD improvement still requires an equal-time
control after the one-puzzle implementation smoke succeeds.

## Kaggle assets

```text
notebook:
yuvraj/arc26-opsd-fixed-pool-validation

code patch dataset:
yuvraj/arc2026-opsd-code-patch

fixed candidate dataset:
yuvraj/arc26-v17-opsd-fixed-candidates-32
```

Notebook version 1 is deliberately inert and GPU-disabled. To run the first
smoke:

1. Edit the notebook.
2. Select the pinned 2025 environment used by the vanilla submission.
3. Enable the L4 x4 accelerator.
4. Set `RUN_EXPERIMENT = True`.
5. Leave `SELECTED_KEYS = ['142ca369']`.
6. Save a version and inspect the OPSD log before expanding to `DEV_KEYS`.

The notebook installs no packages and starts no SGLang process.
