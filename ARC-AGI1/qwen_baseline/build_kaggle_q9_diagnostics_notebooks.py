import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads(
    (ROOT / "arc26-vanilla-v2-q9-24-submit-competition.ipynb").read_text()
)
SHARED = json.loads(
    (ROOT / "arc26-shared-frontier-bounded-smoke4.ipynb").read_text()
)


def clean(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


# Diagnostic 1: train 13e47133 once, then run ordinary and structured DFS on
# the same live TTFT model.
structured = copy.deepcopy(BASE)
structured["cells"][0]["source"] = (
    "# ARC26 q9 fragmentation A/B on 13e47133\n\n"
    "One TTFT model, followed by ordinary q9 and row-structured q9 over the "
    "same 24 validation views. Records structural waste in the ordinary pass "
    "and q-length/lane/time counters in the structured pass.\n"
)
config = structured["cells"][2]["source"]
config = config.replace('MODE = "submit_competition"', 'MODE = "validation"')
config = config.replace("VALIDATION_KEYS = None", 'VALIDATION_KEYS = ["13e47133"]')
config = config.replace("NPROCS = 4", "NPROCS = 1")
config = config.replace("VALIDATION_END_TIME_HOURS = 2.5", "VALIDATION_END_TIME_HOURS = 4.0")
config = config.replace(
    'WORK_NOTEBOOK_ROOT = "/kaggle/working/arc2026_run_vanilla_v2_q9_24"',
    'WORK_NOTEBOOK_ROOT = "/kaggle/working/arc2026_q9_fragmentation_ab"',
)
config = config.replace(
    'WORK_CODE_DIR = "/kaggle/working/arc2026_run_vanilla_v2_q9_24/ARC-AGI1/qwen_baseline"',
    'WORK_CODE_DIR = "/kaggle/working/arc2026_q9_fragmentation_ab/ARC-AGI1/qwen_baseline"',
)
structured["cells"][2]["source"] = config
structured["cells"][3]["source"] += (
    '\nCOMPARE_STRUCTURED_OUTPUT_DIR = "/kaggle/working/q9_structured_compare"\n'
)
structured["cells"][7]["source"] = structured["cells"][7]["source"].replace(
    '"--end-time", str(time.time() + END_TIME_HOURS * 3600),',
    '"--compare-structured-output-dir", COMPARE_STRUCTURED_OUTPUT_DIR,\n'
    '        "--end-time", str(time.time() + END_TIME_HOURS * 3600),',
)
structured["cells"][8]["source"] = r'''import json
import shutil
import sys
from pathlib import Path

if WORK_CODE_DIR not in sys.path:
    sys.path.insert(0, WORK_CODE_DIR)

from arc_decoder import ArcDecoder, hashable, score_kgmon
from arc_loader import ArcDataset

data = ArcDataset.from_file(TEST_PATH, keys=SELECTED_KEYS).load_replies(SOLUTION_PATH)
split_data = data.split_multi_replies()

def load(directory):
    decoder = ArcDecoder(split_data, n_guesses=2)
    decoder.load_decoded_results(directory)
    return decoder

def canonical_sets(decoder):
    return {
        key: sorted({hashable(record["solution"]) for record in records.values()})
        for key, records in decoder.decoded_results.items()
    }

control = load(OUTPUT_DIR)
structured = load(COMPARE_STRUCTURED_OUTPUT_DIR)
diagnostics = json.loads(
    Path(COMPARE_STRUCTURED_OUTPUT_DIR + "_diagnostics/13e47133.json").read_text()
)
control_sets = canonical_sets(control)
structured_sets = canonical_sets(structured)
summary = {
    "candidate_sets_equal": control_sets == structured_sets,
    "control_unique_candidates": {key: len(value) for key, value in control_sets.items()},
    "structured_unique_candidates": {key: len(value) for key, value in structured_sets.items()},
    "control_score": data.validate_submission(
        data.get_submission(control.run_selection_algo(score_kgmon))
    ),
    "structured_score": data.validate_submission(
        data.get_submission(structured.run_selection_algo(score_kgmon))
    ),
    **diagnostics,
}
Path("/kaggle/working/q9_fragmentation_ab_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
shutil.make_archive("/kaggle/working/q9_control_candidates", "zip", OUTPUT_DIR)
shutil.make_archive(
    "/kaggle/working/q9_structured_candidates", "zip", COMPARE_STRUCTURED_OUTPUT_DIR
)
print(json.dumps(summary, indent=2, sort_keys=True))
'''
clean(structured)
(ROOT / "arc26-q9-fragmentation-ab-13e47133.ipynb").write_text(
    json.dumps(structured, indent=1) + "\n"
)


# Diagnostic 2: on one known starved puzzle, compare continuation from the
# threshold-0.2 frontier against a complete threshold-0.1 search.
frontier = copy.deepcopy(SHARED)
frontier["cells"][0]["source"] = (
    "# ARC26 frontier continuation A/B on 5dbc8537\n\n"
    "Runs primary threshold 0.2 once, continues saved 0.1-eligible branches, "
    "and independently runs the complete threshold-0.1 search on identical views.\n"
)
config = frontier["cells"][2]["source"]
start = config.index("VALIDATION_KEYS =")
end = config.index("\n", start)
config = config[:start] + 'VALIDATION_KEYS = ["5dbc8537"]' + config[end:]
config = config.replace("NPROCS = 4", "NPROCS = 1")
config = config.replace("VALIDATION_END_TIME_HOURS = 1.0", "VALIDATION_END_TIME_HOURS = 4.0")
frontier["cells"][2]["source"] = config
frontier["cells"][3]["source"] += (
    '\nFRESH_COMPARE_OUTPUT_DIR = "/kaggle/working/frontier_full_01_compare"\n'
)
launch = frontier["cells"][7]["source"]
launch = launch.replace(
    '"--adaptive-output-dir", ADAPTIVE_OUTPUT_DIR,',
    '"--adaptive-output-dir", ADAPTIVE_OUTPUT_DIR,\n'
    '        "--adaptive-resume-frontier",\n'
    '        "--compare-fresh-adaptive-output-dir", FRESH_COMPARE_OUTPUT_DIR,',
)
frontier["cells"][7]["source"] = launch
frontier["cells"][8]["source"] = r'''import json
import shutil
import sys
from pathlib import Path

if WORK_CODE_DIR not in sys.path:
    sys.path.insert(0, WORK_CODE_DIR)

from arc_decoder import ArcDecoder, hashable
from arc_loader import ArcDataset

data = ArcDataset.from_file(TEST_PATH, keys=SELECTED_KEYS).load_replies(SOLUTION_PATH)
split_data = data.split_multi_replies()

def load(directory):
    decoder = ArcDecoder(split_data, n_guesses=2)
    if Path(directory).exists():
        decoder.load_decoded_results(directory)
    return decoder

def canonical_sets(decoder):
    return {
        key: sorted({hashable(record["solution"]) for record in records.values()})
        for key, records in decoder.decoded_results.items()
    }

primary = load(OUTPUT_DIR)
resumed = load(ADAPTIVE_OUTPUT_DIR)
fresh = load(FRESH_COMPARE_OUTPUT_DIR)
primary_sets = canonical_sets(primary)
resumed_sets = canonical_sets(resumed)
fresh_sets = canonical_sets(fresh)
diagnostics_path = Path(FRESH_COMPARE_OUTPUT_DIR + "_diagnostics/5dbc8537.json")
diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
summary = {
    "primary_unique_candidates": {key: len(value) for key, value in primary_sets.items()},
    "resumed_unique_candidates": {key: len(value) for key, value in resumed_sets.items()},
    "fresh_01_unique_candidates": {key: len(value) for key, value in fresh_sets.items()},
    "resume_equals_full_01": resumed_sets == fresh_sets,
    "resume_missing_from_full": {
        key: len(set(fresh_sets.get(key, [])) - set(resumed_sets.get(key, [])))
        for key in set(fresh_sets) | set(resumed_sets)
    },
    "fresh_compare_diagnostics": diagnostics,
}
Path("/kaggle/working/frontier_resume_ab_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
shutil.make_archive("/kaggle/working/frontier_primary_candidates", "zip", OUTPUT_DIR)
shutil.make_archive("/kaggle/working/frontier_resumed_candidates", "zip", ADAPTIVE_OUTPUT_DIR)
shutil.make_archive("/kaggle/working/frontier_full_01_candidates", "zip", FRESH_COMPARE_OUTPUT_DIR)
print(json.dumps(summary, indent=2, sort_keys=True))
'''
clean(frontier)
(ROOT / "arc26-frontier-resume-ab-5dbc8537.ipynb").write_text(
    json.dumps(frontier, indent=1) + "\n"
)

print(ROOT / "arc26-q9-fragmentation-ab-13e47133.ipynb")
print(ROOT / "arc26-frontier-resume-ab-5dbc8537.ipynb")
