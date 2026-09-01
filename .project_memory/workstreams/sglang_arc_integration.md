# Workstream
SGLang integration, speculative DFS evaluation, persistent LoRA hot-swap inference, and Kaggle submission orchestration for `ARC-AGI1/qwen_baseline`.

OPSD, train-pair calibration, and repair-training results are now maintained canonically in `opsd_and_repair_training.md`; older OPSD entries below remain historical runtime context.

# Current Objective
- Treat Vanilla V2 q9 with 24 views and `score_kgmon` (`31.94` public LB) as the production control.
- Stop investing in Canon-CPT: its production submission scored `30.14`, and its completed validation48 run damaged oracle coverage without improving selected score.
- Measure the opt-in row-structured, length-bucketed q9 DFS against the unchanged Vanilla control. Require valid-candidate preservation plus a measured wall/model-token gain before considering submission use.
- Continue candidate-discovery and selection work using existing archives where possible. Extra color views mainly improved discovery, while the present selector captured little of the additional oracle opportunity.
- Preserve exact notebook/version, view count, DFS threshold, model, TTFT recipe, and leaderboard provenance. Do not compare a smoke, partial run, or differently sized candidate pool as if it were controlled.

# Decisions
- SGLang replaced the earlier vLLM direction for this thread.
- Baseline SGLang parity is treated as solved for:
  - `0934a4d8`
  - `135a2760`
- Dead HF-only `PrefixCachedRescorer` logic was removed from the vanilla path.
- Saved-adapter reuse mode was added so SGLang can score the exact same LoRA weights without retraining.
- Teacher-forced speculative verification replaced the earlier per-token speculative verification.
  - one backend call verifies a whole repeated-token draft
  - side branches are still spawned from intermediate positions to preserve exactness
- Dynamic repeat length was tested and is not considered a meaningful win versus the added complexity.
- Persistent SGLang inference workers with runtime `load_lora_adapter` / `unload_lora_adapter` are implemented; they reduced some startup overhead but have not improved the public LB.
- Both vanilla and the current SGLang adapter phase train four different puzzles concurrently on four GPUs. This is independent per-puzzle training, not DDP for one puzzle.
- The bounded producer-consumer pipeline with two trainer processes on GPUs 0-1 and two persistent inference processes on GPUs 2-3 was implemented in `cb60ac1` and configured in `80aae1f`; it is no longer a proposed next step.
- Persistent trainers load the base model once and reset/save LoRA weights between puzzles; this removed the old adapter-only worker's repeated per-puzzle base-model load.
- Speculative DFS is not assumed to be faster. `next_arc_logprobs` and `draft_arc_logprobs` are distinct batched target-model calls, and the saved-adapter standard/speculative comparison did not establish a competition-quality gain.
- The simple global regularized linear pairwise candidate ranker is deprioritized: task-grouped five-fold evaluation on the full threshold-`0.1` candidate dump scored below `score_kgmon`.
- Kaggle vanilla HF / Unsloth with no SGLang path injection does work with `--nprocs 4`; the earlier claim that Kaggle `spawn` plus Unsloth is inherently broken was wrong.
- Mixed Python environments / path contamination was a real packaging risk when SGLang was injected into the Unsloth runtime; the mounted read-only stack isolates that dependency path, but isolation did not improve score.
- Top-level Unsloth import should not be part of infer-only Python paths; lazy-loading it only in training code paths is the right direction.
- `starter.py` now gives each worker its own `UNSLOTH_COMPILE_LOCATION` under `/tmp`; this is the fix for the intermittent Kaggle `Unsloth*Trainer` import race.
- `run_chunked_sglang_pipeline.py` should not stop the whole run just because one chunk is partial; continuing is the correct default for full validation / submit runs.
- When `KEEP_ADAPTERS = False`, adapters are deleted only after that chunk has been scored. They are not deleted before the matching infer/scoring phase.
- Validation scoring must score only the completed-key subset of decoded outputs. Feeding all decoded result files into a subset dataset causes `KeyError` such as `KeyError: '142ca369'`.
- Validation-only scoring logic must not alter the competition test submission route.
- Durable memory for this thread stays in this workstream file; do not create a second session file for the same topic by default.
- Kaggle notebook-output downloads for historical versions must use the version label form `v12`, `v13`, etc. Numeric `12` / `13` gives `404` in direct SDK calls.
- Do not use `scoring ...` log lines as the manifest for candidate files. Those lines cover rescored candidates only, not every generated candidate file.
- For notebook versions v12/v13, the full generated candidate files are under `inference_outputs_validation/<candidate>`, not `validation_scoring_subset/<candidate>`.
- For oracle/selection analysis, score the actual submitted top 2 only. `ArcDecoder.run_selection_algo(...)` returns the full ranked unique-grid list; checking whether an exact grid appears anywhere in that list overstates submitted score.
- The 2-trainer/2-inference streaming implementation is technically functional, but its public LB result (`26.11`) is worse than vanilla (`30.14`) and the sequential SGLang submissions (`29.03` and `26.94`); deprioritize it as a competition route.
- Chunk size `16` is not established as faster than chunk size `8`; the completed non-speculative threshold-`0.2` run covered only `42/120` tasks in about `2h37m`.
- The chunk-16 run used `KEEP_ADAPTERS = False`; its per-chunk adapters were deleted after scoring and are not available in Kaggle outputs.
- Manual notebook saves must default to `MODE = "submit_predictions"`. Competition reruns may promote through `KAGGLE_IS_COMPETITION_RERUN`; manually saving with `MODE = "submit_competition"` can silently consume hours on the 240-key dummy test route.
- Retained-adapter claims currently apply only to the eight v19 keys: `0934a4d8`, `135a2760`, `136b0064`, `13e47133`, `142ca369`, `16b78196`, `16de56c4`, and `1818057f`.
- `reduced_plus_opsd` means initial SFT excludes one deterministic reserved training pair C, then OPSD uses augmented C; puzzles with fewer than three training pairs fall back to full SFT.
- The OPSD fixed candidate directory contains retained v17 output grids, not augmented inputs. It freezes candidate generation so Phase 1 tests ranking only.
- Do not read `accepted_updates` as solved examples. An accepted update means the teacher gate passed and an optimizer step completed.
- The completed OPSD pilot did not log same-example post-update metrics and did not save the final adapter. It cannot answer whether KL, gold NLL, or divergence-token probability improved after learning.
- Cross-view and same-view are distinct: same-view still uses augmentation, but the privileged C demonstration and student query share that augmentation. Cross-view asks the teacher to transfer C from augmentation `g` to different augmentation `h`.
- Do not infer OPSD loss dilution from the v17 recovery audit. Those candidates come from SFT-trained models; their error topology only motivates measuring OPSD per-position KL.

# Constraints and Assumptions
- Main runtime for real debugging remains the RunPod / Kaggle GPU environment, not local macOS execution.
- Competition budget is four Kaggle L4 GPUs with an `11.5h` submission window; weekly Kaggle allocation is about `30` L4-hours and optional rented-GPU spend is capped near `$20/week`.
- Verified working stack on pod-side work:
  - Unsloth `2025.9.7`
  - SGLang `0.5.1.post3`
  - Torch `2.8.0+cu128`
- SGLang `0.5.1.post3` supports runtime LoRA load/unload:
  - `Engine.load_lora_adapter(...)`
  - `Engine.unload_lora_adapter(...)`
- Dynamic LoRA load/unload in this SGLang version requires `dp_size == 1`.
- The implemented persistent-infer configuration is `tp_size=1`, one base-model engine per inference GPU.
- One puzzle key still maps to one worker job; augmentations are batched within a worker, not split across workers.
- Adapter size is roughly `~1 GB` per puzzle for the current LoRA config.
- Persistent artifacts must live on persistent workspace/output disk, not pod root.
- Kaggle notebook cleanup can break `pip install` if the notebook deletes its current working directory first; the fix is to switch to a safe `cwd` such as `/kaggle/working` before cleanup/install.

# Current State
- Durable code paths:
  - `ARC-AGI1/qwen_baseline/starter.py`
  - `ARC-AGI1/qwen_baseline/arc_solver.py`
  - `ARC-AGI1/qwen_baseline/arc_sglang.py`
  - `ARC-AGI1/qwen_baseline/run_chunked_sglang_pipeline.py`
  - `ARC-AGI1/arc26-2025-winning-solution-submit-launcher.ipynb`

- Latest checkpoint as of `2026-08-06`:
  - public LB control: `Submit Vanilla - Version 2` = `30.14`
  - sequential SGLang submissions: main-fork Version 14 = `29.03`; Version 15 = `26.94`; Version 11 = `8.33`
  - streaming 2+2 submission: `[ARC26] Mounted SGLang 2x2 - Version 1` = `26.11`
  - conclusion: backend speed engineering has not yielded an end-product gain; the next bounded direction is offline candidate selection, not another orchestration rewrite
  - `origin/main` / local `main`: `80aae1f`; streaming runtime commit: `cb60ac1`
  - Kaggle patch dataset `yuvraj/arc2026-mounted-stack-patch` was refreshed from `cb60ac1`; local streaming orchestration tests passed `3/3`

- OPSD fixed-pool checkpoint as of `2026-08-09`:
  - notebook: `yuvraj/arc26-opsd-fixed-pool-validation`
  - code source is the Git-backed `yuvraj/arc2026` dataset; the temporary OPSD patch dataset was detached
  - `142ca369` completed end to end after fixing PEFT named-adapter verification in `42a8e1c`
  - reserved C index: `2`; attempted OPSD examples: `16`; accepted updates: `13`; correction wall: `523.37s`
  - every same-view example passed (`13/13`); every cross-view example failed teacher restricted-greedy exactness (`0/3`)
  - same-view teacher gold-token geometric-mean probability was nearly one (`median 0.999953`); student median was `0.966450`
  - the exact candidate for `142ca369_0` moved from rank `5` to rank `8`; mean augmentation NLL worsened from `18.036084` to `24.368544`; `142ca369_1` had no exact candidate in the fixed pool
  - this is negative end-to-end ranking evidence for full-SFT versus reduced-SFT+OPSD on one oracle-eligible output, but it does not isolate whether reduced SFT or OPSD caused the movement
  - the final OPSD adapter was not saved, and logs contain pre-update rather than same-example post-update measurements
  - `7036d1f` adds rejected-example gold-token error counts/details for future runs; it cannot recover discarded logits from the completed run
  - before another run, add per-update pre/post KL, divergence-token probabilities, gold NLL/exactness, final evaluation over all C views, and final adapter persistence

- Full v17 valid-candidate recovery audit as of `2026-08-09`:
  - code/report commits: `d15b1bb`, refined spatial metrics and corrected interpretation in `1107487`
  - coverage: `2,265` files, `4,056` valid-grid occurrences, `2,268` unique grids, `165/172` outputs, `118/120` tasks
  - entirely absent tasks: `271d71e2`, `4e34c42c`; partial missing outputs also occur for `5dbc8537`, `6e4f6532`, `cbebaa4b`, `f560132c`
  - among `1,966` unique erroneous correct-shape grids, immediate next-cell recovery is `48.7%`, median post-error match is `89.8%`, and only `7.7%` match below half of the remaining cells
  - only `1.5%` are true single-cell errors; `96.7%` contain multiple error spans; medians are `24` wrong cells and `12` spans
  - of `958` immediate recoveries, `928` err again later; for those, `61.3%` place the next error in the same row or column, with median Manhattan distance `2`
  - common SFT candidate pattern: return to gold, then recur in localized error islands; not usually one isolated error and not usually total suffix collapse
  - selected-top-two and occurrence-weighted populations show similar recovery rates

- Latest 2+2 evidence from `/Users/banna/Downloads/download (8).txt`:
  - loaded `source_commit=cb60ac1`
  - two trainer workers used GPUs `0-1`; two persistent inference workers used GPUs `2-3`
  - one-key smoke produced `16/16` decoded outputs and a schema-valid 240-key submission
  - one-key pipeline wall was about `312s` versus roughly `409s` for the earlier sequential one-key route, but concurrent training and SGLang initialization both slowed
  - this startup-heavy smoke did not establish sustained throughput or quality; the public LB result was `26.11`

- Chunk-16 evidence from `/Users/banna/Downloads/download (7).txt`:
  - sequential training/inference, standard DFS, threshold `0.2`, `CHUNK_SIZE=16`
  - completed `42/120` validation tasks in about `2h37m`; completed-key-only score was `11.3333`
  - logs show adapter cleanup after chunks, so this run did not retain reusable adapters
- Main branch history already includes:
  - `3dd8a1f` fix SGLang rsLoRA scaling and add RunPod bootstrap
  - `cccaa09` remove dead vanilla prefix-cached rescoring path
  - `5aacb9c` add SGLang speculative repeated-token DFS
- Additional later fixes already merged to `main`:
  - decouple top-level Unsloth use away from infer-only paths
  - add train-only and infer-only orchestration around saved adapters
  - add per-worker Unsloth compile-cache isolation in `starter.py`
  - make chunked pipeline continue after partial chunks instead of stopping
  - replace the earlier launcher notebook with the downloaded Kaggle notebook and patch it for:
    - `validation`
    - `submit_predictions`
    - `submit_competition`

- Persistent hot-swap flow exists in repo history:
  - branch: `codex/sglang-persistent-hot-swap`
  - commit: `61e049f`
  - PR: `https://github.com/Yuvraj25/NVARC/pull/4`
- CLI / orchestration additions that matter for the current flow:
  - `--sglang-adapter-manifest`
  - `--sglang-infer-from-manifest`
  - `--sglang-infer-workers`
  - `--sglang-train-adapters-only` writes/updates a manifest
- Persistent SGLang behavior:
  - infer-only mode reads ready adapters from a manifest
  - persistent workers keep one base-model engine alive
  - each worker loads one puzzle LoRA, runs DFS/rescoring, unloads it, then moves to the next job

- Verified parallelism and load behavior as of `2026-08-04`:
  - `starter.py --nprocs 4` spawns four processes and sets `CUDA_VISIBLE_DEVICES` to the worker rank
  - both vanilla and SGLang training therefore process up to four puzzle adapters concurrently
  - vanilla loads one base model per worker outside its puzzle loop and resets LoRA weights per puzzle
  - the SGLang adapter-only path currently loads the base model inside its puzzle loop, adding avoidable repeated checkpoint-loading overhead
  - summed per-puzzle work is GPU-worker time, not notebook wall time

- Full threshold-`0.1` 120-task validation timing:
  - sequential chunked pipeline wall: `6.263h`
  - completed tasks: `114/120`
  - summed adapter-pass work: `29962.5s = 8.323 GPU-hours`
  - summed inference work: `26752.4s = 7.431 GPU-hours`
  - projected sequential runtime for 240 competition tasks: about `12.66h`, above the `11.5h` limit
  - ideal 120-task 2-trainer/2-inference lower bound: `max(8.323/2, 7.431/2) = 4.16h`
  - ideal 240-task lower bound: about `8.32h`; real runtime needs measurement and will be higher
  - one trainer plus three inference workers is training-bound; three trainers plus one inference worker is inference-bound; this made `2+2` the planning choice, but the later smoke and `26.11` LB result did not validate it as the production route

- Full threshold-`0.1` candidate-selection evidence:
  - selected top-2 score: `36.333333333333336 / 120 = 30.28%`
  - oracle any-candidate score: `45.66666666666667 / 120 = 38.06%`
  - selection gap: `9.333333333333334` task-points, or `7.78` percentage points
  - candidate directory: `/Users/banna/kaggle/temp/kaggle_versions/v17_files/inference_outputs_validation`
  - outputs evaluated: `172`
  - unique candidate grids: `2268`
  - outputs with the exact answer present: `61`
  - global linear pairwise ranker, task-grouped five-fold results:
    - `score_kgmon` baseline: `36.333333333333336`
    - best tested linear ranker: `34.833333333333336`
    - other regularization settings: `33.833333333333336` to `34.333333333333336`
  - conclusion: do not spend GPU on this global linear-ranker direction without a materially different task-conditioned hypothesis

- Threshold-`0.2` speculative-call accounting over the 48-task validation run:
  - `next_arc_logprobs` calls: `24135`
  - `draft_arc_logprobs` calls: `24094`
  - total SGLang generation calls: `48229`
  - speculative branches: `99529`
  - branches accepting zero extra repeated tokens: `38772` (`~39%`)
  - accepted extra repeated tokens: `263608`
  - verified draft positions: `795871`
  - accepted-position fraction: `33.1%`
  - these counters alone do not show whether speculation wins because standard DFS would replace some skipped speculative levels with additional next-token calls
  - local instrumentation now records SGLang `cached_tokens` separately for next-token DFS, draft verification, and rescoring; Kaggle must run the updated `arc_sglang.py` and `arc_solver.py` for those counters to appear

- Immediate GPU experiment:
  - use threshold `0.2` with standard SGLang DFS: `USE_SPECULATIVE_DFS=False`
  - eight keys:
    - `0934a4d8`
    - `135a2760`
    - `136b0064`
    - `13e47133`
    - `142ca369`
    - `16b78196`
    - `16de56c4`
    - `1818057f`
  - retain adapters with `KEEP_ADAPTERS=True`
  - compare DFS wall, backend-call counts, candidate count, selected score, oracle, and cache fractions against the existing threshold-`0.2` speculative run
  - if the result is close, rerun speculative inference from the same retained adapters rather than retraining

- Kaggle vanilla control run on `2026-07-01` with clean Unsloth env and no SGLang path injection:
  - command shape:
    - `python starter.py --test-path /kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json --model-path /kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1 --output-dir /kaggle/working/inference_outputs_smoke --keys-json '["135a2760","981571dc","0934a4d8"]' --nprocs 4 --profile-timings`
  - env:
    - `UNSLOTH_DISABLE_STATISTICS=1`
    - `HF_HUB_OFFLINE=1`
    - `TRANSFORMERS_OFFLINE=1`
    - `HF_HUB_ENABLE_HF_TRANSFER=0`
    - `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
    - `OMP_NUM_THREADS=12`
  - result:
    - `135a2760`: `subkeys_written=16`, `beam_candidates_valid=18`, `training_s=128.686`, `dfs_s=425.140`, `rescoring_s=20.154`, `total_wall_s=575.155`
    - `0934a4d8`: `subkeys_written=15`, `beam_candidates_valid=18`, `training_s=397.950`, `dfs_s=41.282`, `rescoring_s=130.090`, `total_wall_s=571.208`
    - `981571dc`: `subkeys_written=16`, `beam_candidates_valid=16`, `training_s=855.049`, `dfs_s=377.172`, `rescoring_s=14.308`, `total_wall_s=1249.685`
  - implication:
    - one worker printing `[Rank 3] done!` early in a 3-key / 4-worker run is normal because that worker only consumes a sentinel
    - this strongly suggests the earlier Kaggle crash came from the SGLang-injected environment, not from vanilla Kaggle multiprocessing by itself

- Known same-adapter timing / correctness results:
  - HF / Unsloth vanilla total for `135a2760`:
    - `482.1s`
  - SGLang baseline reuse for `0934a4d8`:
    - `dfs_s=38.868`
    - `rescoring_s=16.959`
    - `total_wall_s=92.916`
    - `subkeys_written=16`
    - score `1.0`
  - SGLang speculative reuse for `0934a4d8` (`repeat_len=4`, old path):
    - `dfs_s=56.468`
    - `rescoring_s=16.896`
    - `total_wall_s=110.314`
    - `subkeys_written=16`
    - score `1.0`
    - interpretation: slower than baseline

- `135a2760` speculative results after teacher-forced verification:
  - static `repeat_len=5`:
    - `engine_init=32.17s`
    - `dfs_s=299.633s`
    - `rescoring_s=21.570s`
    - `total_wall_s=355.910s`
    - `subkeys_written=14`
    - score `1.0`
  - static `repeat_len=9`:
    - `engine_init=31.93s`
    - `dfs_s=288.133s`
    - `rescoring_s=21.773s`
    - `total_wall_s=344.029s`
    - `subkeys_written=15`
    - score `1.0`
  - static `repeat_len=17`:
    - `engine_init=32.04s`
    - `dfs_s=276.985s`
    - `rescoring_s=21.827s`
    - `total_wall_s=332.949s`
    - `subkeys_written=15`
    - score `1.0`
  - dynamic `17`, loose thresholds:
    - `total_wall_s=336.030s`
    - score `1.0`
    - not useful
  - dynamic `17`, stricter thresholds:
    - `engine_init=31.88s`
    - `dfs_s=274.804s`
    - `rescoring_s=21.542s`
    - `total_wall_s=330.437s`
    - `subkeys_written=14`
    - score `1.0`
    - best measured SGLang total on this puzzle so far, but still only about `1.46x` vs HF total

- Current verified failing-key difference between vanilla and SGLang:
  - vanilla non-SGLang run for:
    - `136b0064`
    - `13e47133`
    - `16b78196`
    - `16de56c4`
  - produced:
    - `136b0064`: `beam_candidates_seen=1`, `subkeys_written=1`
    - `13e47133`: `beam_candidates_seen=3`, `subkeys_written=3`
    - `16b78196`: `beam_candidates_seen=10`, `subkeys_written=7`
    - `16de56c4`: `beam_candidates_seen=7`, `subkeys_written=6`
  - SGLang runs on saved adapters for overlapping failing keys produced much less:
    - `136b0064`: `beam_candidates_seen=0`, `subkeys_written=0`
    - `13e47133`: `beam_candidates_seen=0`, `subkeys_written=0`
    - `16de56c4`: `beam_candidates_seen=1`, `subkeys_written=1`
  - implication:
    - the SGLang path is not only losing candidates during rescoring
    - on these failures it often never produces completed beam candidates at all
    - this is why validation coverage stalls even when training and adapter hot-loading succeed

- Recent 3-key persistent-infer SGLang run on saved adapters showed:
  - `136b0064`
    - `engine_init_s=0.000` after the persistent engine was warm
    - `dfs_frames_expanded=650`
    - `beam_candidates_seen=0`
    - `subkeys_written=0`
  - `16de56c4`
    - `dfs_frames_expanded=1588`
    - `beam_candidates_seen=1`
    - `subkeys_written=1`
  - `13e47133`
    - `dfs_frames_expanded=7860`
    - `beam_candidates_seen=0`
    - `subkeys_written=0`
  - this matches the broader diagnosis that SGLang DFS is diverging before candidate completion on some puzzles

- Kaggle validation launcher state from the latest logged validation run:
  - requested validation keys:
    - `120`
  - completed validation keys:
    - `9`
  - completed keys:
    - `0934a4d8`
    - `135a2760`
    - `1818057f`
    - `20270e3b`
    - `221dfab4`
    - `28a6681f`
    - `2ba387bc`
    - `2d0172a1`
    - `31f7f899`
  - selected outputs:
    - `12`
  - completed-only local validation score:
    - `4.5`
  - interpretation:
    - this is the ARC validation accuracy sum on the completed subset, not "4.5 completed puzzles"
    - on that completed subset the accuracy is `4.5 / 9 = 50%`
  - observed decoded file count:
    - `113`
    - this is decoded subkey-file count, not completed puzzle count
  - exact attempted-key count for a run must come from:
    - `/kaggle/working/inference_outputs_validation_chunk_state.json`
    - do not infer it from completed-key count or decoded-file count

- Current launcher notebook behavior:
  - modes:
    - `validation`
    - `submit_predictions`
    - `submit_competition`
  - validation mode stages only completed-key decoded outputs into a filtered scoring directory before calling `ArcDecoder`
  - the test submission route is intended to remain unchanged by that validation-only filter
  - after any `main` fix, the Kaggle runtime copy must be refreshed; stale copied code can preserve the old partial-chunk-stop behavior

- Current interpretation:
  - teacher-forced speculative verification was the real improvement
  - dynamic repeat length is not worth the complexity
  - removing per-puzzle `engine_init` was still the right performance move
  - the remaining blocker is not simply engine startup
  - the remaining blocker is that SGLang DFS is not logically matching vanilla DFS on some keys and is often failing before candidate completion

- Kaggle competition submission evidence:
  - corrected vanilla notebook public LB score:
    - `30.14`
  - SGLang/speculative competition notebook public LB score:
    - `8.33`
  - Do not infer final hidden-test output contents from Kaggle UI; Kaggle code competition submissions are black-box after rerun.
  - The `8.33` result is still unexplained by local validation evidence alone and remains an open issue.

- Kaggle notebook version mapping used for the recent validation analysis:
  - version `v12` = SGLang/speculative threshold `0.05`
  - version `v13` = SGLang/speculative threshold `0.2`
  - notebook slug:
    - `yuvraj/arc26-2025-winning-solution-v1-455107`

- Historical notebook-output download findings:
  - Stock `kaggle kernels output` only handles one page and prints a next-page token; it is insufficient for full output download by itself.
  - The installed Kaggle client output listing is polluted by copied dependency/code files before generated outputs:
    - candidate files first appeared around page `71`
  - Correct direct SDK request fields:
    - `ApiListKernelSessionOutputRequest.user_name = "yuvraj"`
    - `kernel_slug = "arc26-2025-winning-solution-v1-455107"`
    - `version_label = "v12"` or `"v13"`
    - use `page_token` pagination
  - Do not force `page_size=1000`; this caused `400 Bad Request`. Default page size returned `500` files/page and worked.

- Full downloaded output artifacts on local macOS:
  - v12 / threshold `0.05`:
    - full output dir: `/Users/banna/kaggle/temp/kaggle_versions/v12_full/inference_outputs_validation`
    - files downloaded: `998/998`
    - failures: `0`
  - v13 / threshold `0.2`:
    - full output dir: `/Users/banna/kaggle/temp/kaggle_versions/v13_full/inference_outputs_validation`
    - files downloaded: `781/781`
    - failures: `0`
  - Earlier incomplete folders from the wrong manifest should not be used for conclusions:
    - `/Users/banna/kaggle/temp/kaggle_versions/v12_files/validation_scoring_subset`
    - `/Users/banna/kaggle/temp/kaggle_versions/v13_files/validation_scoring_subset`

- Corrected 48-key validation comparison from full downloaded outputs:
  - denominator: explicit 48-key validation list used in the parity / threshold experiments
  - evaluator inputs:
    - solutions: `/Users/banna/kaggle/NVARC/external/TinyRecursiveModels/kaggle/combined/arc-agi_evaluation2_solutions.json`
    - candidate dirs:
      - `/Users/banna/kaggle/temp/kaggle_versions/v12_full/inference_outputs_validation`
      - `/Users/banna/kaggle/temp/kaggle_versions/v13_full/inference_outputs_validation`
  - v12 / threshold `0.05`:
    - decoded basekeys: `68`
    - selected top-2 score with `score_kgmon`: `11.833333333333334`
    - selected top-2 score with `score_full_probmul_3`: `11.833333333333334`
    - oracle any-candidate score: `18.333333333333336`
  - v13 / threshold `0.2`:
    - decoded basekeys: `63`
    - selected top-2 score with `score_kgmon`: `12.333333333333334`
    - selected top-2 score with `score_full_probmul_3`: `12.333333333333334`
    - oracle any-candidate score: `15.333333333333334`
  - conclusion:
    - threshold `0.05` generated materially more exact candidates than `0.2`
    - current ranking fails to select many of those candidates into top 2
    - selection/ranking work is justified; candidate generation at `0.05` is not just garbage
  - audit CSV:
    - `/Users/banna/kaggle/temp/kaggle_versions/oracle_compare_v12_v13_full_48_top2.csv`

- Corrected ranking misses from full-output oracle analysis:
  - v12 / threshold `0.05` exact candidates present but missed by top-2 selection:
    - `142ca369_0`
    - `20270e3b_0`
    - `269e22fb_1`
    - `28a6681f_0`
    - `31f7f899_0`
    - `3dc255db_0`
    - `4c7dc4dd_0`
    - `58490d8a_0`
    - `62593bfd_1`
  - v13 / threshold `0.2` exact candidates present but missed by top-2 selection:
    - `142ca369_0`
    - `20270e3b_0`
    - `20a9e565_0`
    - `28a6681f_0`
    - `62593bfd_1`

- Output-level differences that explain `0.05` vs `0.2` selected-score behavior:
  - `31f7f899_0`:
    - v12/0.05 has exact candidates but first exact rank is `5`, so top 2 misses it
    - v13/0.2 has an exact candidate at rank `2`, so top 2 submits it
  - `36a08778_1`:
    - v12/0.05 solves it at rank `1`
    - v13/0.2 has no exact candidate; best is `12` pixel errors
  - net selected-score effect on the 48-key denominator:
    - v13/0.2 beats v12/0.05 by `0.5`
    - this is a ranking/top-2 effect, not evidence that `0.05` generated worse candidates overall

# Open Questions
- Does an OPSD optimizer step reduce KL and increase gold probability on the exact same sampled trajectory immediately after the update?
- Is OPSD KL concentrated at divergence/error-island positions, or substantial even where sampled and gold tokens match?
- On cross views, how many of 420 gold-prefix positions have the wrong teacher argmax and with what margins? Future logs now record this; the completed pilot cannot recover it.
- Should cross-view OPSD remain fully gated, be disabled for the basic experiment, or use token-level confidence/gold-consistency masking? Do not train unfiltered cross-view distributions based only on aggregate NLL.
- Can a task-conditioned selector recover the v12/0.05 or full threshold-`0.1` oracle gap without overfitting?
  - Immediate target keys for ranker debugging:
    - `142ca369_0`
    - `20270e3b_0`
    - `269e22fb_1`
    - `28a6681f_0`
    - `31f7f899_0`
    - `3dc255db_0`
    - `4c7dc4dd_0`
    - `58490d8a_0`
    - `62593bfd_1`
  - First concrete example:
    - `142ca369_0` exact candidate was rank `7` under `score_kgmon`, so top 2 missed it.
- The tested global linear combination of candidate support, `beam_score`, `score_aug` distribution, grid size, and augmentation aggregates did not generalize. A future ranking proposal must introduce new information or task-conditioned calibration, not merely another linear reweighting of the same scalars.
- Can any offline selector recover at least two task-points under task-grouped validation before further GPU work is justified?
- After quota resets, which single production control should be retained: exact vanilla or the closest sequential SGLang configuration? Do not answer from partial validation timing alone.
- Hidden-test completion coverage is not observable in a Kaggle code competition; infer it only from controlled local/validation coverage plus the final public score, and label the inference as uncertain.

# References
- Workstream memory:
  - `NVARC/.project_memory/workstreams/sglang_arc_integration.md`
- Main code:
  - `NVARC/ARC-AGI1/qwen_baseline/starter.py`
  - `NVARC/ARC-AGI1/qwen_baseline/arc_solver.py`
  - `NVARC/ARC-AGI1/qwen_baseline/arc_sglang.py`
  - `NVARC/ARC-AGI1/qwen_baseline/run_chunked_sglang_pipeline.py`
  - `NVARC/ARC-AGI1/qwen_baseline/arc_opsd.py`
  - `NVARC/ARC-AGI1/qwen_baseline/analyze_candidate_recovery.py`
- OPSD and recovery reports:
  - `NVARC/ARC-AGI1/qwen_baseline/OPSD_FIXED_POOL_EXPERIMENT.md`
  - `NVARC/ARC-AGI1/qwen_baseline/V17_CANDIDATE_RECOVERY_AUDIT.md`
- Kaggle launcher notebook:
  - `NVARC/ARC-AGI1/arc26-2025-winning-solution-submit-launcher.ipynb`
- Kaggle notebook inventory (created or directly maintained during this work):
  - `https://www.kaggle.com/code/yuvraj/arc26-mounted-sglang-2x2` - 2-trainer/2-inference streaming launcher; smoke passed, but competition Version 1 scored `26.11`.
  - `https://www.kaggle.com/code/yuvraj/arc26-mounted-sglang-chunk16` - mounted-stack sequential chunk-16 launcher; used for standard-DFS threshold-`0.2` validation and the later threshold-`0.1` submission configuration; chunk-16 speedup was not established.
  - `https://www.kaggle.com/code/yuvraj/arc-mounted-sglang-stack-smoke` - minimal compatibility smoke proving the read-only notebook-output SGLang stack could be mounted without copying roughly 8 GB into `/kaggle/working`.
  - `https://www.kaggle.com/code/yuvraj/sglang` - utility notebook that builds/packages the offline SGLang dependency stack; infrastructure only, not a solver submission.
  - `https://www.kaggle.com/code/yuvraj/arc26-train-pair-calibration-poc` - eight-retained-adapter leave-one-training-pair-out ranking POC; uploaded inert by default and produced no accepted competition result.
  - `https://www.kaggle.com/code/yuvraj/arc26-train-pair-calibration-2025-environment` - pinned-2025-environment compatibility variant of the calibration POC; environment diagnostic, not a scoring result.
  - `https://www.kaggle.com/code/yuvraj/check-parity` - backend parity/compatibility probe reused during calibration environment debugging; diagnostic only.
  - `https://www.kaggle.com/code/yuvraj/arc26-2025-winning-solution-submit-launcher` - early mode-aware validation/prediction/competition launcher; superseded by the mounted launchers.
  - `https://www.kaggle.com/code/yuvraj/arc26-2025-winning-solution-v1-455107` - existing primary fork and artifact source, not newly created here; contains the sequential SGLang versions and the retained v19 eight-key adapters.
  - `https://www.kaggle.com/code/yuvraj/submit-vanilla` - historical 16-view vanilla control; public LB `30.14`; superseded by the 24-view Vanilla V2 Version 4 result at `31.94`.
- Downloaded full candidate outputs:
  - `/Users/banna/kaggle/temp/kaggle_versions/v12_full/inference_outputs_validation`
  - `/Users/banna/kaggle/temp/kaggle_versions/v13_full/inference_outputs_validation`
- Oracle comparison artifact:
  - `/Users/banna/kaggle/temp/kaggle_versions/oracle_compare_v12_v13_full_48_top2.csv`
- Copied final submissions used to verify local regeneration:
  - `/Users/banna/kaggle/temp/sglang_05/results.json`
  - `/Users/banna/kaggle/temp/sglang_20/results.json`
- Downloaded logs:
  - `/Users/banna/Downloads/download (1).txt`
  - `/Users/banna/Downloads/download (2).txt`
  - `/Users/banna/Downloads/download (3).txt`
  - `/Users/banna/Downloads/download (4).txt`
  - `/Users/banna/Downloads/download (7).txt` - chunk-16 standard-DFS threshold-`0.2` validation
  - `/Users/banna/Downloads/download (8).txt` - 2+2 streaming smoke/submission run
- PR for persistent hot-swap flow:
  - `https://github.com/Yuvraj25/NVARC/pull/4`
- Exact SGLang wheel previously inspected for runtime LoRA support:
  - `/Users/banna/runpod_artifacts/notebookc4ca2ea220_output/offline_pkgs/sglang-0.5.1.post3-py3-none-any.whl`
- Example new flow:
  - train/save adapters via `--sglang-train-adapters-only`
  - infer from manifest via `--sglang-infer-from-manifest ... --sglang-infer-workers N`
- Validation chunk-state file produced on Kaggle runs:
  - `/kaggle/working/inference_outputs_validation_chunk_state.json`

# Update - 2026-08-23: legacy q9, 24-view competition route

## Intended production configuration
- Control lineage: `Submit Vanilla V2`, public LB `30.14`.
- Legacy Sorokin stack: Unsloth `2025.9.7`, Unsloth Zoo `2025.9.9`, Transformers `4.55.4`, Torch `2.8.0+cu128`.
- Four L4 workers; each worker independently trains one puzzle-specific rank-256 LoRA. This is not DDP.
- LoRA targets are the seven projection modules plus `embed_tokens` and `lm_head`.
- HF/Unsloth inference, no SGLang; old-stack speculative DFS with `q_len=9`, threshold `0.2`, selector `score_kgmon`.
- Evaluation views increased from `16` to `24`: eight geometric transforms times three color permutations.
- Runtime controls:
  - CLI `--eval-color-permutations`
  - environment `ARC_EVAL_COLOR_PERMUTATIONS`

## First hidden submission failure and diagnosis
- Initial implementation: `c3081dd`; notebook-schema cleanup: `8d7bdfb`.
- Competition notebook: `yuvraj/arc26-vanilla-v2-q9-24-prompt-submit`.
- Competition Version 2 threw an opaque notebook exception. Kaggle competition reruns do not provide downloadable logs; do not ask for them or claim a hidden-stage cause from the generic status.
- A production-faithful ordinary GPU diagnostic was therefore created: `yuvraj/arc26-q9-24-validation8-diagnostic`.
- Version 1 reproduced the failure after the first puzzle completed 128 TTFT steps. The first q9 DFS batch crashed at `arc_search_multitoken.py:191` while constructing `torch.tensor(prefix_tokens, ...)`:
  - `ValueError: expected sequence of length 1120 at dim 1 (got 1116)`.
- Root cause: the legacy 16-view path paired prompts with compatible token lengths, but the newly added generic non-16 branch grouped four consecutive subkeys. With 24 views, consecutive prompts can tokenize to different lengths, and the q9 search code does not pad them.

## Fix and real validation
- Fix commit: `18da557 Batch 24-view DFS prompts by token length`.
- `_build_eval_batches(eval_ds, tokenizer=None, formatter=None)` now:
  - preserves the exact legacy batching for 16 views;
  - for non-16 view counts, groups subkeys by the tokenized length of `data["input"]`;
  - chunks each equal-length group into batches of at most four;
  - receives tokenizer/formatter in both HF and SGLang call sites.
- Added `test_eval_batches.py`. A local fake-tokenizer check on `13e47133` covered all 48 views exactly once in 12 equal-length batches.
- After refreshing `yuvraj/arc2026`, the mounted `arc_solver.py` SHA256 was verified as `1e21b8d22c3cef91f84f2cfb3ecb26f5f912aa539628c9f0bca8bd45f44c79b6`.
- Diagnostic Version 2 completed the same eight puzzles end to end in `1602.5s` (`26.7m`) with no traceback:
  - expected test outputs: `11`;
  - outputs with decoded candidates: `10`;
  - normal no-candidate output: `13e47133_0`;
  - `score_full_probmul_3 = 2.0/8`;
  - `score_kgmon = 2.0/8`;
  - `score_kgmon_median = 2.0/8`.
- The selected score matches the prior retained-eight control. This validates operability, not a quality gain.
- Sum of measured task times projected to roughly `9.7h` for 240 tasks on four workers. That is an estimate, not a runtime guarantee.

## Current competition handoff
- Budget commit: `5c30f8c Extend q9 competition inference to 11h50m`.
- Notebook source: `ARC-AGI1/arc26-vanilla-v2-q9-24-submit-competition.ipynb`.
- Kaggle competition notebook Version 4 is the only version intended for submission:
  - `https://www.kaggle.com/code/yuvraj/arc26-vanilla-v2-q9-24-prompt-submit`
  - `SUBMIT_COMPETITION_END_TIME_HOURS = 11 + 50 / 60`.
- Version 4 ordinary save/preflight completed and produced a 240-task placeholder JSON; full inference activates only under `KAGGLE_IS_COMPETITION_RERUN`.
- Do not submit Versions 2 or 3.

## Durable gate learned from this failure
- Static schema, view-count, and placeholder-submission checks are insufficient for a candidate-count/view-count change.
- Before a hidden competition rerun, run the exact Kaggle image, utility, model, four-GPU topology, TTFT, and at least the first real inference decode in an ordinary notebook.
- Inspect candidate coverage, output artifacts, errors, and measured timing. A save-version preflight is not a scored or production-faithful smoke.
- Old-stack q9 remains approximate. Do not confuse it with the later modern-stack result where only q5 passed strict speculative-parity gates and q9 did not.

## Work-log metadata
- Delegation/subagents: none.
- Verification performed for this update: inspected the authoritative repository history, the two existing project-memory workstreams, the saved q9 diagnostic artifacts, and the Version 4 handoff facts; Markdown was checked by re-reading the appended section.
- Git state: `.project_memory` is currently untracked in the nested `NVARC` repository, so these memory edits are local and were not committed or pushed.
- Next action: submit only competition notebook Version 4 if the user still wants this approximate old-stack q9/24 route; do not treat its ordinary save-version preflight as evidence that the hidden 240-task run completed.

# Update - 2026-08-31: Vanilla 31.94, color-view evidence, Canon rejection, and structured q9

## Production and leaderboard state
- Current best verified submission:
  - notebook: `yuvraj/arc26-vanilla-v2-q9-24-prompt-submit`
  - Version 4
  - public LB: `31.94`
  - model/TTFT: published `qwen3_4b_grids15_sft139` plus per-task Vanilla V2 rank-256 LoRA, including `embed_tokens` and `lm_head`
  - inference: HF/Unsloth q9 repeated-token DFS, threshold `0.2`, 24 views (`8` geometries x `3` color/order variants), `score_kgmon`
- Later leaderboard results did not beat this control:
  - uniform 32-view Vanilla: `30.94`
  - 24-view threshold `0.1`: `30.97`
  - one-pass scheduled sampling, 96 TTFT steps: `26.94`
  - Canon-CPT q9 24-view submission: `30.14`
- Do not resubmit the unchanged 24-view control as an experiment. New submissions need a concrete generation, selection, or runtime change.

## Extra-color and color-agreement findings
- Completed clean Vanilla 32-view validation48 archive:
  - local archive: `/Users/banna/kaggle/temp/kaggle_vanilla_v2_q9_32_validation48_v2_output/q9_32_validation48_candidates.zip`
  - tasks/outputs: `48 / 73`
  - valid prompt files: `1,508`
  - candidate occurrences: `1,840`
  - unique canonical candidates: `645`
  - outputs with no candidate: `4`
  - selected score: `11.833333333333334 / 48`
  - oracle: `14.833333333333334 / 48`, `23` correct outputs present
  - correct-rank distribution: rank `1`: `12`; rank `2`: `6`; rank `3`: `1`; rank `5`: `1`; rank `6`: `2`; rank `10`: `1`
- Within one geometry/output family, agreement means multiple color/order variants generate the same canonical grid. Abstention means a color variant generated no valid grid and must not be conflated with disagreement.
- Among geometry/output groups that generated the exact answer under the four colors, correct support was:
  - `1/4`: `5`
  - `2/4`: `13`
  - `3/4`: `7`
  - `4/4`: `74`
- Color agreement is strongly associated with correctness, but the observed 16-to-24-view gain was mainly additional candidate discovery rather than reranking an already present correct grid. Simple support boosts (`1.25` or `1.5` for repeated family support) did not recover the five missed exact candidates in the examined archive.
- Removing fully blank/background-only candidates did not materially fix the missed ranks.

## Candidate-selection and multi-output work
- The published Fayche tuple selector matches literal full augmentation tags and uses the first DFS candidate. In the downloaded implementation, different outputs were independently augmented, so literal shared view tags did not occur and the tuple mechanism effectively fell back/no-op.
- The useful idea is to generate all outputs of a multi-output puzzle from deliberately shared augmentation descriptors, then use shared-view evidence without assuming candidate-index equivalence.
- Code commit `8674d45` added:
  - `ArcDataset.split_multi_replies_shared_views()`;
  - cheap-first task ordering;
  - a primary 24-view threshold-`0.2` pass;
  - a separate adaptive 24-view threshold-`0.1` pool only for outputs with fewer than two primary unique candidates.
- Notebook: `yuvraj/arc26-shared-views-adaptive-validation49`; denominator is all `49` public multi-output tasks (`101` outputs). Do not state a score conclusion until its saved outputs are fetched and evaluated.
- Lux's cheap-first cost estimate was validated offline against completed Vanilla timing:
  - Pearson correlation with wall time: `0.8337`
  - Spearman: `0.7944`
  - cheapest-quartile overlap: `8/12`
  - this is useful for deadline coverage, not evidence of higher per-task accuracy.

## Scheduled-sampling conclusion
- One-pass scheduled sampling was implemented as a no-gradient probe pass followed by a mixed-prefix supervised pass.
- The tested configuration produced extremely few argmax replacements after TTFT and then scored `26.94` on the public leaderboard.
- Treat this route as negative. Do not keep proposing scheduled sampling unless the corruption policy/objective changes materially and a small controlled run establishes nontrivial exposure.

## Canon-CPT implementation and conclusion
- Canon-AC was implemented as residual causal horizontal convolutions over hidden states, initially trained by continued pretraining from the untouched published ARC checkpoint, with a global rank-256 LoRA introduced after the Canon-only phase.
- The completed continued-pretraining run saw only about `35k` records in `11.5h` on four L4s; this was roughly `1%` of the original `3.255M` corpus and was compute-limited.
- A premerge notebook merged the global LoRA into the base model and retained `canon_ac.pt` as a sidecar. Cached/full and sibling-backtracking parity had to be repaired before q9 DFS could run.
- Production result:
  - notebook: `ARC26 Canon CPT q9 24 submit - Version 3`
  - public LB: `30.14`
  - Vanilla control: `31.94`
- Completed Canon validation48 authoritative artifacts:
  - candidate archive: `/Users/banna/kaggle/temp/kaggle_canon_cpt_q9_24_validation48_results/canon_q9_24_validation48_candidates.zip`
  - submission: `/Users/banna/kaggle/temp/kaggle_canon_cpt_q9_24_validation48_results/canon_cpt_q9_24_validation48_submission.json`
  - tasks/outputs: `48 / 73`
  - valid prompt files: `1,089`
  - candidate occurrences: `1,365`
  - unique candidates: `556`
  - outputs with candidates: `67/73`; six missing outputs
  - selected: `17` correct outputs, `11.833333333333334 / 48`
  - oracle: `19` correct outputs, `12.833333333333334 / 48`
  - correct ranks: rank `1`: `12`; rank `2`: `5`; rank `5`: `1`; rank `8`: `1`
- Versus the 32-view Vanilla validation48 archive, Canon had the same selected score but lower oracle (`12.8333` versus `14.8333`). This comparison is view-count-confounded; the leaderboard comparison independently points in the same negative direction.
- Canon selected gain versus Vanilla32: `31f7f899_0`. Selected losses: `36a08778_1`, `4a21e3da_1`. Their task weights cancel, explaining equal selected score.
- Canon-only oracle outputs: `20270e3b_0`, `31f7f899_0`. Vanilla-only oracle outputs: `221dfab4_1`, `28a6681f_0`, `36a08778_1`, `4a21e3da_1`, `4c7dc4dd_0`, `4c7dc4dd_1`.
- Durable conclusion: stop spending competition compute on Canon-CPT. It did not improve leaderboard score and reduced candidate coverage/oracle in the available validation evidence.

## Row-structured, length-bucketed q9 experiment
- ARC outputs are bounded at `30x30`. The new structural decoder tracks branch-local:
  - established width or `None` during the first row;
  - current column;
  - completed row count.
- Structural rules:
  - first row cannot exceed `30` cells;
  - its first newline establishes width;
  - later rows must contain exactly that width;
  - EOS is legal only after a complete non-empty row;
  - no row beyond row `30` is explored;
  - original full-vocabulary NLL and DFS threshold are retained; there is no constrained renormalization.
- q9 calls are partitioned by each active lane's safe draft length. Newline/EOS lanes use q1; digit lanes use `min(9, remaining cells in row)`. Different safe lengths receive separate calls, keeping caches rectangular without a ragged attention mask.
- This path is opt-in and Vanilla-only for the first experiment:
  - flag: `--use-unsloth-structured-rows`
  - requires `--use-unsloth-multitoken-dfs`
  - existing production q9 behavior remains unchanged by default
  - Canon is explicitly rejected for this initial path
- Implementation commit: `02d9b96 Add row-structured length-bucketed q9 DFS`.
- Main code/tests:
  - `ARC-AGI1/qwen_baseline/arc_search_multitoken.py`
  - `ARC-AGI1/qwen_baseline/arc_solver.py`
  - `ARC-AGI1/qwen_baseline/starter.py`
  - `ARC-AGI1/qwen_baseline/test_arc_search_multitoken.py`
- Six CPU tests pass: unchanged legacy path, rectangular candidate preservation, first-row cap, ragged-row rejection, boundary q length, and separate calls for different safe lengths.
- Kaggle validation notebook:
  - repo: `ARC-AGI1/arc26-vanilla-v2-row-structured-q9-24-validation8.ipynb`
  - URL: `https://www.kaggle.com/code/yuvraj/arc26-vanilla-row-structured-q9-validation8`
  - Version 1 was running when this memory update was written.
- Acceptance gate:
  - complete without Unsloth cache/batch errors;
  - compare selected score and oracle with the retained-eight Vanilla control;
  - inspect valid candidate count and exact-grid preservation;
  - compare `model_calls`, `model_tokens`, q-length call histogram, DFS wall, and total wall;
  - do not promote based only on fewer malformed branches.

## Current next actions
1. Fetch Version 1 of `arc26-vanilla-row-structured-q9-validation8` after completion and evaluate the full saved candidate pool plus timing counters.
2. Fetch and evaluate `arc26-shared-views-adaptive-validation49` if completed; separately score primary-only, safe adaptive fill, shared-view weighting, and combined variants.
3. Keep Vanilla V2 31.94 as the competition fallback until one controlled experiment beats its validation behavior without damaging oracle or coverage.

## Update - 2026-09-01: structured-q9 arithmetic and pending DFS work

- Retained-eight aggregate task counters, counting each puzzle once:
  - control: task-wall sum `4652.633s`, DFS `2501.970s`, model `2298.685s`, rescoring `457.818s`, block calls `22449`, draft tokens `808080`, valid candidates `180`;
  - structured: task-wall sum `4662.287s`, DFS `2557.288s`, model `2370.762s`, rescoring `401.350s`, block calls `19635`, draft tokens `608844`, valid candidates `170`.
- Therefore the structured run reduced block calls by `12.53%` and draft tokens by `24.66%`, but increased aggregate model time by `3.14%`, DFS time by `2.21%`, and task-wall work by `0.21%`. Its shorter notebook makespan was load-balancing/scheduling, not a genuine throughput gain.
- Measured average model time per block call rose from `102.4ms` to `120.7ms` (`+17.9%`). Average draft tokens per block call fell from `36.0` to `31.0`. The current logs do not contain a q-length/effective-batch-size histogram, so attributing the regression specifically to batch fragmentation remains a hypothesis, not a measured conclusion.
- Pending hard-grammar experiment: after the first row establishes width, force newline at each later row boundary and disallow newline before that boundary; allow EOS only after a complete non-empty row and within the `30x30` limit. Compare against rejection-only grammar and unchanged q9. This changes search probabilities/branch availability and must pass candidate-preservation and score gates before submission use.
- Pending global adaptive scheduler: finish the primary threshold-`0.2` pass first, collect starved outputs globally, then spend the remaining wall-clock budget on a prioritized queue. The current fresh-adaptive implementation runs immediately per starved puzzle and is conditional allocation, not global leftover-time allocation.
- Pending efficient threshold relaxation: preserve threshold-`0.1`-eligible branches pruned by the primary `0.2` search and resume only those branches later. The first frontier implementation was not validated: its smoke produced zero decoded adaptive outputs, and it must not be restored until deterministic equivalence plus a successful starved-output smoke passes.
