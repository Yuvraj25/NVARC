# Unsloth cached multi-token probe

This is an experimental gate for extending the winner-patched Qwen3 cached
inference path from `q_len == 1` to `q_len >= 1`. It is not connected to DFS
or rescoring.

The patch must be applied before the Python process imports `unsloth`. The
probe does this itself, so run it as a fresh subprocess after the existing
Unsloth and FlashAttention setup cells:

```bash
python probe_unsloth_qwen3_multitoken.py \
  --unsloth-package-dir /usr/local/lib/python3.11/dist-packages/unsloth \
  --model-path /kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1 \
  --adapter-path /kaggle/input/<retained-adapter-dataset>/<adapter-directory> \
  --test-path /kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json \
  --puzzle-key 136b0064 \
  --draft-len 9 \
  --repeats 10
```

If the installed package is under a different Python directory, locate it
without importing Unsloth:

```bash
python -c 'import importlib.util; print(importlib.util.find_spec("unsloth").submodule_search_locations[0])'
```

The experiment passes only when:

- the maximum log-probability difference is at most `0.02` for ARC tokens
  that could still fit under the DFS probability budget;
- the next ARC-token argmax agrees;
- viable/pruned token membership agrees;
- the returned cache grows by exactly the draft length; and
- one cached `q_len=9` call is at least 1.05x faster than nine cached
  `q_len=1` calls.

The output line beginning `MULTITOKEN_PROBE` also reports full-vocabulary
log-probability difference, median latency, and peak allocated VRAM. Do not
integrate this path into DFS unless both `parity_pass` and `speed_pass` are
true on the pinned Kaggle environment.

## 2026-08-10 base-model result

Kaggle notebook version 4 passed on four augmented prompts from `136b0064`:

- `q_len=9` search-relevant maximum log-probability difference: `0.0020483`
- viable/pruned membership disagreements: `0`
- ARC argmax disagreements: `0`
- cache length: `1129`, expected `1129`
- sequential median: `421.34 ms`
- block median: `51.98 ms`
- speedup: `8.11x`
- sequential peak allocated VRAM: `9,916,151,296` bytes
- block peak allocated VRAM: `10,081,059,840` bytes

Low-probability tail tokens differed by as much as `1.0`, but all were below
`log(0.2)` and could not survive the DFS budget even from a zero-cost parent.
Both cached paths also showed BF16 tail differences against the full-forward
reference.

## 2026-08-10 retained-LoRA result

Kaggle notebook version 5 passed with the retained `136b0064` adapter and four
real augmented prompts:

- `parity_pass=true`; `speed_pass=true`
- `q_len=2` search-relevant maximum difference: `3.5763e-7`
- `q_len=5` search-relevant maximum difference: `3.5763e-7`
- `q_len=9` search-relevant maximum difference: `0.0043401`
- viable/pruned membership disagreements: `0`
- ARC argmax disagreements: `0`
- cache length: `1129`, expected `1129`
- sequential median: `605.46 ms`
- block median: `81.97 ms`
- speedup: `7.39x`
- sequential peak allocated VRAM: `13,079,281,152` bytes
- block peak allocated VRAM: `13,247,554,560` bytes

This establishes the cached multi-token primitive on the production model and
LoRA path. It does not establish end-to-end speculative DFS equivalence: the
zero-parent probability gate does not directly test every cumulative-NLL
retain/prune decision made by DFS.

## 2026-08-10 threshold-0.1 shadow-DFS result

Kaggle notebook version 6 ran the sequential `q_len=1` DFS as the authority at
`max_score=-log(0.1)=2.3025851`. It recorded real frames, selected the candidate
branches initially closest to the pruning boundary, and replayed their
repeated-token steps through both sequential and cached multi-token paths.

- retained adapter: `136b0064`
- real augmented prompts: `4`
- sequential DFS frames recorded: `128`
- candidate branches found: `131`
- riskiest branches compared: `32`
- repeated-token decisions compared: `65`
- retain/prune set divergences: `0`
- closest tested sequential decision to the NLL boundary: `0.5581424`
- maximum raw ARC-token log-probability difference: `0.5`
- shadow decision gate: passed
- probe wall time after model loading: `168.00 s`

This is the first test that applies the real cumulative score rule,
`next_score = parent_score - log_probability`, rather than classifying tokens
only by an isolated probability threshold. It is still bounded evidence from
one puzzle and early DFS frames. It does not prove full candidate equality,
preservation of sibling traversal order, or equivalent timeout behavior. The
next gate is a guarded inference-only integration that runs sequential DFS as
the output authority while comparing multi-token decisions and candidate sets.

## 2026-08-10 complete DFS candidate gates

The conservative integration accepts an extra repeated token only while every
active lane has exactly one viable non-EOS continuation and that continuation
is the repeated token. Any ambiguity returns to the ordinary recursive DFS
frame. The implementation is in `arc_search_multitoken.py` and remains disabled
unless `starter.py` receives `--use-unsloth-multitoken-dfs`.

Kaggle notebook version 7, one augmented prompt:

- complete candidates: `1` vanilla, `1` multi-token
- candidate token sequences equal: `true`
- vanilla DFS: `16.5485 s`
- multi-token DFS: `8.7244 s`
- speedup: `1.8968x`
- accepted extra repeated tokens: `127`
- zero-extra blocks: `45 / 89`
- maximum matching candidate NLL difference: `0.04081`

Kaggle notebook version 8, production-shaped batch of four augmented prompts:

- complete candidates: `4` vanilla, `4` multi-token
- candidate token sequences equal: `true`
- vanilla DFS: `46.4069 s`
- multi-token DFS: `28.2084 s`
- speedup: `1.6451x`
- accepted extra repeated tokens: `497`
- zero-extra blocks: `150 / 278`
- maximum matching candidate NLL difference: `0.28520`

Both paths completed under identical threshold-`0.1`, `max_new_tokens=256`,
and 240-second per-path caps. Candidate-token parity passed, but beam NLL is
not numerically identical. The current `score_kgmon` selector does not use
`beam_score`; other selection algorithms must not assume score parity. Before
competition use, run the flagged vanilla path on multiple validation puzzles
and compare selected top-2, any-candidate oracle, completion counts, runtime,
and the new multi-token profiling counters.
