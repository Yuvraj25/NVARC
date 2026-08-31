import json
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "arc26-vanilla-v2-q9-24-submit-competition.ipynb"
TARGET = ROOT / "arc26-shared-views-frontier-resume49-validation.ipynb"
SUBMISSION_TARGET = ROOT / "arc26-vanilla-v2-shared-frontier-bounded-submit.ipynb"
CHALLENGES = (
    Path(__file__).resolve().parents[2]
    / "external/TinyRecursiveModels/kaggle/combined/arc-agi_evaluation2_challenges.json"
)

challenges = json.loads(CHALLENGES.read_text())
multi_output_keys = sorted(
    key for key, task in challenges.items() if len(task["test"]) > 1
)
assert len(multi_output_keys) == 49

notebook = json.loads(SOURCE.read_text())
notebook["cells"][0]["source"] = (
    "# ARC26 shared-view + adaptive recovery validation (49 multi-output puzzles)\n\n"
    "Vanilla V2 production TTFT with q9 and 24 primary views. Every test output "
    "within a puzzle receives the identical geometry, color permutation, and "
    "demonstration ordering. Outputs with fewer than two distinct primary "
    "candidates resume only the primary DFS boundary branches newly admitted "
    "when the cumulative threshold relaxes from 0.2 to 0.1. The same per-task "
    "LoRA and primary augmentation views remain live; primary and resumed pools "
    "are preserved separately.\n"
)

notebook["cells"][2]["source"] = f'''MODE = "validation"

CODE_DATASET_ROOT = "/kaggle/input/datasets/yuvraj/arc2026"
MODEL_PATH = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
COMP_ROOT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"

VALIDATION_KEYS = {multi_output_keys!r}
NPROCS = 4
DFS_PROB_THRESHOLD = 0.2
UNSLOTH_MULTITOKEN_REPEAT_LEN = 9
EVAL_COLOR_PERMUTATIONS = 3
PROFILE_TIMINGS = True

ADAPTIVE_DFS_PROB_THRESHOLD = 0.1
ADAPTIVE_COLOR_PERMUTATIONS = 3
ADAPTIVE_MIN_UNIQUE_CANDIDATES = 2

VALIDATION_END_TIME_HOURS = 4.0
SUBMIT_COMPETITION_END_TIME_HOURS = 11 + 50 / 60
RESET_RUN_ARTIFACTS = True

WORK_NOTEBOOK_ROOT = "/kaggle/working/arc2026_shared_adaptive49"
WORK_CODE_DIR = WORK_NOTEBOOK_ROOT + "/ARC-AGI1/qwen_baseline"
WRITABLE_UNSLOTH_PARENT = "/kaggle/working/shared_adaptive49_stack"
'''

command_cell = notebook["cells"][7]
source = command_cell["source"]
needle = '        "--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),\n'
replacement = needle + '''        "--shared-eval-augmentations",
        "--adaptive-output-dir", ADAPTIVE_OUTPUT_DIR,
        "--adaptive-resume-frontier",
        "--adaptive-dfs-prob-threshold", str(ADAPTIVE_DFS_PROB_THRESHOLD),
        "--adaptive-color-permutations", str(ADAPTIVE_COLOR_PERMUTATIONS),
        "--adaptive-min-unique-candidates", str(ADAPTIVE_MIN_UNIQUE_CANDIDATES),
        "--cheap-first",
'''
assert needle in source
command_cell["source"] = source.replace(needle, replacement)

# Replace the reporting cell. Generation is the expensive part; all selection
# arms below operate on the retained candidate records only.
notebook["cells"][8]["source"] = r'''import json
import shutil
import sys
from pathlib import Path

import numpy as np

if WORK_CODE_DIR not in sys.path:
    sys.path.insert(0, WORK_CODE_DIR)

from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, score_kgmon
from arc_shared_selection import select_bounded_shared_support, fill_missing_attempts

data = ArcDataset.from_file(TEST_PATH, keys=SELECTED_KEYS).load_replies(SOLUTION_PATH)
split_data = data.split_multi_replies()
Path(ADAPTIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

primary_ranked, primary_counts, primary_decoder = select_bounded_shared_support(
    data, split_data, OUTPUT_DIR, support_weight=2.0, n_guesses=2
)
primary_kgmon = primary_decoder.run_selection_algo(score_kgmon)

combined_decoder = ArcDecoder(split_data, n_guesses=2)
if Path(OUTPUT_DIR).exists():
    combined_decoder.load_decoded_results(OUTPUT_DIR)
if Path(ADAPTIVE_OUTPUT_DIR).exists():
    combined_decoder.load_decoded_results(ADAPTIVE_OUTPUT_DIR, run_name=".adaptive")
combined_ranked = combined_decoder.run_selection_algo(score_kgmon)

safe_adaptive = fill_missing_attempts(
    primary_ranked, combined_ranked, primary_counts, n_guesses=2
)

def score(selected):
    submission = data.get_submission(selected)
    return data.validate_submission(submission)

def oracle(decoder):
    total = 0.0
    for base_key, records in decoder.decoded_results.items():
        puzzle_key, _ = base_key.rsplit("_", 1)
        gold = split_data.replies[base_key][0]
        if any(np.array_equal(record["solution"], gold) for record in records.values()):
            total += 1 / len(data.queries[puzzle_key]["test"])
    return total

summary = {
    "requested_tasks": len(SELECTED_KEYS),
    "requested_outputs": len(split_data.keys),
    "primary_decoded_outputs": len(primary_decoder.decoded_results),
    "adaptive_decoded_outputs": len(
        {
            path.name.split(".")[0]
            for path in Path(ADAPTIVE_OUTPUT_DIR).glob("*")
            if path.is_file()
        }
    ) if Path(ADAPTIVE_OUTPUT_DIR).exists() else 0,
    "primary_zero_candidate_outputs": sum(value == 0 for value in primary_counts.values()),
    "primary_one_candidate_outputs": sum(value == 1 for value in primary_counts.values()),
    "primary_two_plus_candidate_outputs": sum(value >= 2 for value in primary_counts.values()),
    "primary_kgmon_score": score(primary_kgmon),
    "primary_bounded_support_score": score(primary_ranked),
    "safe_adaptive_score": score(safe_adaptive),
    "primary_oracle": oracle(primary_decoder),
    "combined_oracle": oracle(combined_decoder),
    "primary_unique_counts": primary_counts,
}

SUMMARY_PATH = Path("/kaggle/working/shared_adaptive49_summary.json")
SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
shutil.make_archive("/kaggle/working/shared_primary_candidates", "zip", OUTPUT_DIR)
shutil.make_archive("/kaggle/working/shared_adaptive_candidates", "zip", ADAPTIVE_OUTPUT_DIR)
print(json.dumps(summary, indent=2, sort_keys=True))
'''

# The base setup cell defines OUTPUT_DIR. Add the separate adaptive location
# immediately afterward so it is available to both launcher and reporting.
setup_source = notebook["cells"][3]["source"]
anchor = 'print("mode_requested =", MODE)\n'
assert anchor in setup_source
notebook["cells"][3]["source"] = setup_source.replace(
    anchor,
    'ADAPTIVE_OUTPUT_DIR = "/kaggle/working/shared_adaptive_candidates"\n\n' + anchor,
)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

TARGET.write_text(json.dumps(notebook, indent=1) + "\n")

submission_notebook = copy.deepcopy(notebook)
submission_notebook["cells"][0]["source"] = (
    "# ARC26 Vanilla V2 shared-frontier bounded submission\n\n"
    "The 31.94 Vanilla V2 q9/24-view control with shared multi-output views, "
    "cheap-first ordering, threshold-0.2 primary DFS, starved-output frontier "
    "resume to threshold 0.1, bounded shared-tuple support weight 2, and safe "
    "adaptive filling that never displaces two primary attempts.\n"
)
submission_notebook["cells"][2]["source"] = submission_notebook["cells"][2][
    "source"
].replace('MODE = "validation"', 'MODE = "submit_competition"', 1)
submission_notebook["cells"][8]["source"] = r'''import json
import sys
from pathlib import Path

if WORK_CODE_DIR not in sys.path:
    sys.path.insert(0, WORK_CODE_DIR)

from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, score_kgmon
from arc_shared_selection import select_bounded_shared_support, fill_missing_attempts

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(ADAPTIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
data = ArcDataset.from_file(TEST_PATH, keys=SELECTED_KEYS)
split_data = data.split_multi_replies()

primary_ranked, primary_counts, primary_decoder = select_bounded_shared_support(
    data, split_data, OUTPUT_DIR, support_weight=2.0, n_guesses=2
)
combined_decoder = ArcDecoder(split_data, n_guesses=2)
combined_decoder.load_decoded_results(OUTPUT_DIR)
combined_decoder.load_decoded_results(ADAPTIVE_OUTPUT_DIR, run_name=".adaptive")
combined_ranked = combined_decoder.run_selection_algo(score_kgmon)
selected = fill_missing_attempts(
    primary_ranked, combined_ranked, primary_counts, n_guesses=2
)

submission = data.get_submission(selected)
Path(SUBMISSION_PATH).write_text(json.dumps(submission))
print("submission tasks =", len(submission))
print("primary zero/one/two+ =", {
    "zero": sum(value == 0 for value in primary_counts.values()),
    "one": sum(value == 1 for value in primary_counts.values()),
    "two_plus": sum(value >= 2 for value in primary_counts.values()),
})
print("submission_path =", SUBMISSION_PATH)
'''
for cell in submission_notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
SUBMISSION_TARGET.write_text(json.dumps(submission_notebook, indent=1) + "\n")

print(TARGET)
print(SUBMISSION_TARGET)
print("multi-output tasks:", len(multi_output_keys))
print("test outputs:", sum(len(challenges[key]["test"]) for key in multi_output_keys))
