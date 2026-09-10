import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_NOTEBOOK = Path(
    "/Users/banna/kaggle/temp/kaggle_vanilla_v2_q9_32_validation48/"
    "arc26-vanilla-v2-q9-32-validation48.ipynb"
)
OUTPUT_NOTEBOOK = REPO_ROOT / "ARC-AGI1/arc26-qwen-trm-mean-nll-validation48.ipynb"
METADATA_TEMPLATE = Path(
    "/Users/banna/kaggle/temp/kaggle_vanilla_v2_q9_24_submit/kernel-metadata.json"
)
OUTPUT_METADATA = Path(
    "/Users/banna/kaggle/temp/kaggle_qwen_trm_mean_nll_validation48/kernel-metadata.json"
)
VALIDATION_KEYS = [
    "0934a4d8", "135a2760", "136b0064", "13e47133", "142ca369",
    "16b78196", "16de56c4", "1818057f", "195c6913", "1ae2feb7",
    "20270e3b", "20a9e565", "21897d95", "221dfab4", "247ef758",
    "269e22fb", "271d71e2", "28a6681f", "291dc1e1", "2b83f449",
    "2ba387bc", "2c181942", "2d0172a1", "31f7f899", "332f06d7",
    "35ab12c3", "36a08778", "38007db0", "3a25b0d8", "3dc255db",
    "3e6067c3", "409aa875", "446ef5d2", "45a5af55", "4a21e3da",
    "4c3d4a41", "4c416de3", "4c7dc4dd", "4e34c42c", "53fb4810",
    "5545f144", "581f7754", "58490d8a", "58f5dbd5", "5961cc34",
    "5dbc8537", "62593bfd", "64efde09",
]


def main():
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    notebook["cells"][0]["source"] = (
        "# ARC26 Qwen/TRM mean-NLL validation48\n\n"
        "Train the unchanged per-puzzle Qwen LoRA, skip DFS, and rescore the retained "
        "Qwen 8x4 candidate pool plus saved TRM attempts using mean target-token NLL.\n"
    )
    notebook["cells"][2]["source"] = f'''MODE = "validation"

CODE_DATASET_ROOT = "/kaggle/input/datasets/yuvraj/arc2026"
MODEL_PATH = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
COMP_ROOT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"

VALIDATION_KEYS = {json.dumps(VALIDATION_KEYS, indent=4)}
NPROCS = 4
PROFILE_TIMINGS = True
VALIDATION_END_TIME_HOURS = 4.0

WORK_NOTEBOOK_ROOT = "/kaggle/working/arc26_qwen_trm_mean_nll_validation48"
WORK_CODE_DIR = WORK_NOTEBOOK_ROOT + "/ARC-AGI1/qwen_baseline"
WRITABLE_UNSLOTH_PARENT = "/kaggle/working/qwen_trm_mean_nll_stack"
FIXED_CANDIDATE_DIR = "/kaggle/working/qwen_trm_fixed_candidates"
OUTPUT_DIR = "/kaggle/working/qwen_trm_mean_nll_scored"
SUMMARY_PATH = "/kaggle/working/qwen_trm_mean_nll_validation48_summary.json"
TEST_PATH = f"{{COMP_ROOT}}/arc-agi_evaluation_challenges.json"
SOLUTION_PATH = f"{{COMP_ROOT}}/arc-agi_evaluation_solutions.json"
END_TIME_HOURS = VALIDATION_END_TIME_HOURS
RUN_INFERENCE = True
'''
    notebook["cells"][3]["source"] = '''import os

IS_KAGGLE_RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").lower() in {
    "1", "true", "yes", "y", "on"
}
assert not IS_KAGGLE_RERUN, "This is a validation-only scoring notebook"
EFFECTIVE_MODE = "validation"
RESET_RUN_ARTIFACTS = True
SUBMISSION_PATH = SUMMARY_PATH
SELECTED_KEYS = VALIDATION_KEYS
print("test_path =", TEST_PATH)
print("output_dir =", OUTPUT_DIR)
print("fixed_candidate_dir =", FIXED_CANDIDATE_DIR)
print("selected_keys =", SELECTED_KEYS)
'''
    notebook["cells"][5]["source"] = '''import os
import shutil
from pathlib import Path

src = Path(CODE_DATASET_ROOT)
dst = Path(WORK_NOTEBOOK_ROOT)
shutil.copytree(src, dst)

required_files = [
    "starter.py",
    "arc_solver.py",
    "arc_rescoring.py",
    "analyze_qwen_trm_mean_nll.py",
]
for name in required_files:
    assert Path(WORK_CODE_DIR, name).is_file(), f"arc2026 is stale: missing {name}"

starter_source = Path(WORK_CODE_DIR, "starter.py").read_text()
solver_source = Path(WORK_CODE_DIR, "arc_solver.py").read_text()
rescoring_source = Path(WORK_CODE_DIR, "arc_rescoring.py").read_text()
assert "--fixed-candidate-mean-nll" in starter_source
assert "fixed_candidate_mean_nll" in solver_source
assert "normalize_by_answer_tokens" in rescoring_source

candidate_source_dir = Path(WORK_CODE_DIR, "qwen_trm_validation48_fixed_candidates")
assert candidate_source_dir.is_dir(), "arc2026 is stale: missing expanded candidate directory"
shutil.rmtree(FIXED_CANDIDATE_DIR, ignore_errors=True)
shutil.copytree(candidate_source_dir, FIXED_CANDIDATE_DIR)
assert len(list(Path(FIXED_CANDIDATE_DIR).iterdir())) == 1581
print("fixed candidate pool ready:", FIXED_CANDIDATE_DIR)
'''
    notebook["cells"][7]["source"] = '''import json
import subprocess
import sys
import time

cmd = [
    sys.executable,
    "starter.py",
    "--test-path", TEST_PATH,
    "--model-path", MODEL_PATH,
    "--output-dir", OUTPUT_DIR,
    "--nprocs", str(NPROCS),
    "--fixed-candidate-dir", FIXED_CANDIDATE_DIR,
    "--fixed-candidate-mean-nll",
    "--eval-color-permutations", "4",
    "--end-time", str(time.time() + END_TIME_HOURS * 3600),
    "--keys-json", json.dumps(VALIDATION_KEYS),
]
if PROFILE_TIMINGS:
    cmd.append("--profile-timings")
print("running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=WORK_CODE_DIR, env=RUN_ENV, check=True)
'''
    notebook["cells"][8]["source"] = '''import json
import subprocess
import sys

cmd = [
    sys.executable,
    "analyze_qwen_trm_mean_nll.py",
    "--challenges", TEST_PATH,
    "--solutions", SOLUTION_PATH,
    "--candidate-dir", OUTPUT_DIR,
    "--keys-json", json.dumps(VALIDATION_KEYS),
    "--output", SUMMARY_PATH,
]
print("analyzing:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=WORK_CODE_DIR, env=RUN_ENV, check=True)
print(Path(SUMMARY_PATH).read_text())
'''
    notebook["cells"][9]["source"] = '''import shutil
from pathlib import Path

archive_path = shutil.make_archive(
    "/kaggle/working/qwen_trm_mean_nll_scored_candidates",
    "zip",
    root_dir=OUTPUT_DIR,
)
print("scored_candidate_archive =", archive_path)
print("scored_candidate_archive_bytes =", Path(archive_path).stat().st_size)
print("summary_path =", SUMMARY_PATH)
shutil.rmtree(OUTPUT_DIR)
shutil.rmtree(FIXED_CANDIDATE_DIR)
shutil.rmtree(WORK_NOTEBOOK_ROOT)
shutil.rmtree(WRITABLE_UNSLOTH_PARENT)
print("large working directories removed")
'''

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")

    metadata = json.loads(METADATA_TEMPLATE.read_text())
    metadata.update({
        "id": "yuvraj/arc26-qwen-trm-mean-nll-validation48",
        "title": "ARC26 Qwen TRM mean NLL validation48",
        "code_file": OUTPUT_NOTEBOOK.name,
    })
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUTPUT_NOTEBOOK)
    print(OUTPUT_METADATA)


if __name__ == "__main__":
    main()
