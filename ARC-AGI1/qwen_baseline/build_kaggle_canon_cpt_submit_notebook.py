"""Build the Canon-CPT 24-prompt competition submission notebook."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARC_ROOT = HERE.parent
VANILLA = ARC_ROOT / "arc26-vanilla-v2-q9-24-submit-competition.ipynb"
VALIDATION = ARC_ROOT / "arc26-canon-cpt-ttft-24-validation8.ipynb"
OUTPUT = ARC_ROOT / "arc26-canon-cpt-q9-24-submit-competition.ipynb"


def source(cell: dict, text: str) -> None:
    cell["source"] = text
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


vanilla = json.loads(VANILLA.read_text())
validation = json.loads(VALIDATION.read_text())
notebook = deepcopy(vanilla)

source(
    notebook["cells"][0],
    """# ARC26 Canon-CPT q9 with 24 inference prompts

Competition launcher matching the 31.94 Vanilla V2 configuration: q9 multi-token
DFS, threshold 0.2, eight geometries by three colour/order views, `score_kgmon`,
and an 11h50 wall-clock budget. The only model change is the premerged Canon-CPT
checkpoint plus jointly fine-tuned Canon-AC and rank-256 per-task LoRA during TTFT.
""",
)

source(
    notebook["cells"][2],
    """MODE = "submit_competition"  # validation | submit_competition

from pathlib import Path

CODE_DATASET_ROOT = Path("/kaggle/input/datasets/yuvraj/arc2026")
PREMERGED_COMPETITION_ROOT = Path("/kaggle/input/arc26-canon-cpt-premerged-model/canon_cpt_premerged")
PREMERGED_NOTEBOOK_ROOT = Path("/kaggle/input/notebooks/yuvraj/arc26-canon-cpt-premerged-model/canon_cpt_premerged")
COMP_ROOT = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
MODERN_UTILITY_ROOT = Path("/kaggle/usr/lib/notebooks/yuvraj/pip_install_unsloth_ddp_repair")
FA2_ROOT = Path("/kaggle/input/notebooks/yuvraj/flash-attention-cu13-torch-2-11-cp312/flash_attn_cu13_torch211_cp312")

VALIDATION_KEYS = None
NPROCS = 4
DFS_PROB_THRESHOLD = 0.2
UNSLOTH_MULTITOKEN_REPEAT_LEN = 9
EVAL_COLOR_PERMUTATIONS = 3
SELECTION_ALGORITHM = "score_kgmon"
PROFILE_TIMINGS = True

VALIDATION_END_TIME_HOURS = 2.5
SUBMIT_COMPETITION_END_TIME_HOURS = 11 + 50 / 60
RESET_RUN_ARTIFACTS = True

WORK_NOTEBOOK_ROOT = Path("/kaggle/working/arc26_canon_cpt_submit")
WORK_CODE_DIR = WORK_NOTEBOOK_ROOT / "ARC-AGI1/qwen_baseline"
WRITABLE_UNSLOTH_PARENT = Path("/kaggle/working/canon_q9_submit_stack")
""",
)

source(
    notebook["cells"][3],
    """import os


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y", "on"}


IS_KAGGLE_RERUN = _truthy_env("KAGGLE_IS_COMPETITION_RERUN")
EFFECTIVE_MODE = "submit_competition" if IS_KAGGLE_RERUN else MODE
PREMERGED_ROOT = PREMERGED_COMPETITION_ROOT if IS_KAGGLE_RERUN else PREMERGED_NOTEBOOK_ROOT
MODEL_PATH = PREMERGED_ROOT / "model"
CANON_STATE = PREMERGED_ROOT / "canon_ac.pt"

EVAL_CHALLENGES = COMP_ROOT / "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS = COMP_ROOT / "arc-agi_evaluation_solutions.json"
TEST_CHALLENGES = COMP_ROOT / "arc-agi_test_challenges.json"

if IS_KAGGLE_RERUN:
    TEST_PATH = TEST_CHALLENGES
    SOLUTION_PATH = None
    OUTPUT_DIR = Path("/kaggle/working/canon_cpt_q9_24_submit_candidates")
    SUBMISSION_PATH = Path("/kaggle/working/submission.json")
    SELECTED_KEYS = None
    END_TIME_HOURS = SUBMIT_COMPETITION_END_TIME_HOURS
    RUN_INFERENCE = True
elif MODE == "validation":
    TEST_PATH = EVAL_CHALLENGES
    SOLUTION_PATH = EVAL_SOLUTIONS
    OUTPUT_DIR = Path("/kaggle/working/canon_cpt_q9_24_validation_candidates")
    SUBMISSION_PATH = Path("/kaggle/working/canon_cpt_q9_24_validation_submission.json")
    SELECTED_KEYS = VALIDATION_KEYS
    END_TIME_HOURS = VALIDATION_END_TIME_HOURS
    RUN_INFERENCE = True
else:
    TEST_PATH = TEST_CHALLENGES
    SOLUTION_PATH = None
    OUTPUT_DIR = Path("/kaggle/working/canon_cpt_q9_24_shortcut")
    SUBMISSION_PATH = Path("/kaggle/working/submission.json")
    SELECTED_KEYS = None
    END_TIME_HOURS = 0.0
    RUN_INFERENCE = False

print("mode_requested =", MODE)
print("is_kaggle_rerun =", IS_KAGGLE_RERUN)
print("test_path =", TEST_PATH)
print("end_time_hours =", END_TIME_HOURS)
print("run_inference =", RUN_INFERENCE)
""",
)

# Reuse the already exercised Canon validation environment and writable-Unsloth
# setup, replacing only its validation-specific paths and execution cells.
source(notebook["cells"][4], """import importlib.util
import os
import shutil
import subprocess
import sys

required_files = [
    CODE_DATASET_ROOT / "ARC-AGI1/qwen_baseline/starter.py",
    TEST_PATH,
    MODERN_UTILITY_ROOT / "unsloth/__init__.py",
    FA2_ROOT / "flash_attn/__init__.py",
]
if RUN_INFERENCE:
    required_files.extend([MODEL_PATH / "config.json", CANON_STATE])
for required in required_files:
    if not required.is_file():
        raise FileNotFoundError(required)

for path in (WORK_NOTEBOOK_ROOT, OUTPUT_DIR, WRITABLE_UNSLOTH_PARENT):
    shutil.rmtree(path, ignore_errors=True)
if RESET_RUN_ARTIFACTS:
    SUBMISSION_PATH.unlink(missing_ok=True)
shutil.copytree(CODE_DATASET_ROOT, WORK_NOTEBOOK_ROOT)

environment = os.environ.copy()
old_parts = [
    part for part in environment.get("PYTHONPATH", "").split(os.pathsep)
    if part and "pip_install_unsloth_" not in part and "flash_attention_" not in part
]
environment.update({
    "PYTHONPATH": os.pathsep.join([
        str(FA2_ROOT), str(WRITABLE_UNSLOTH_PARENT), str(MODERN_UTILITY_ROOT),
        "/kaggle/working", str(WORK_CODE_DIR), *old_parts,
    ]),
    "PYTHONPYCACHEPREFIX": "/kaggle/working/python_cache",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "OMP_NUM_THREADS": "3",
    "PYTHONUNBUFFERED": "1",
})
print("model =", MODEL_PATH)
print("canon state =", CANON_STATE)
""" )

source(notebook["cells"][5], """writable_unsloth = WRITABLE_UNSLOTH_PARENT / "unsloth"
shutil.copytree(MODERN_UTILITY_ROOT / "unsloth", writable_unsloth)
subprocess.run(
    [
        sys.executable,
        str(WORK_CODE_DIR / "patch_unsloth_qwen3_multitoken.py"),
        "--unsloth-package-dir", str(writable_unsloth),
    ],
    env=environment,
    check=True,
)
print("patched writable Unsloth =", writable_unsloth)
""" )

source(notebook["cells"][6], """# Source-only production preflight; no model load on save-version runs.
starter_source = (WORK_CODE_DIR / "starter.py").read_text()
solver_source = (WORK_CODE_DIR / "arc_solver.py").read_text()
search_source = (WORK_CODE_DIR / "arc_search_multitoken.py").read_text()
for marker in (
    "--canon-ac-state",
    "--use-unsloth-multitoken-dfs",
    "--eval-color-permutations",
):
    if marker not in starter_source:
        raise RuntimeError(f"Missing starter marker: {marker}")
if "Canon TTFT delta_l2=" not in solver_source:
    raise RuntimeError("arc2026 lacks joint Canon TTFT")
if "del outputs" not in search_source:
    raise RuntimeError("arc2026 lacks q9 recursive-output release fix")
print("Canon competition preflight passed")
""" )

source(notebook["cells"][7], """import json
import time

if RUN_INFERENCE:
    command = [
        sys.executable, str(WORK_CODE_DIR / "starter.py"),
        "--test-path", str(TEST_PATH),
        "--model-path", str(MODEL_PATH),
        "--output-dir", str(OUTPUT_DIR),
        "--nprocs", str(NPROCS),
        "--dfs-prob-threshold", str(DFS_PROB_THRESHOLD),
        "--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),
        "--ttft-method", "full_sft",
        "--canon-ac-state", str(CANON_STATE),
        "--use-unsloth-multitoken-dfs",
        "--unsloth-multitoken-repeat-len", str(UNSLOTH_MULTITOKEN_REPEAT_LEN),
        "--end-time", str(time.time() + END_TIME_HOURS * 3600),
    ]
    if PROFILE_TIMINGS:
        command.append("--profile-timings")
    if SELECTED_KEYS is not None:
        command.extend(["--keys-json", json.dumps(SELECTED_KEYS)])
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORK_CODE_DIR, env=environment, check=True)
else:
    print("save-version shortcut: full inference runs only during competition rerun")
""" )

# The vanilla decoder/submission cell is intentionally preserved verbatim; it
# produces /kaggle/working/submission.json with score_kgmon and two guesses.
notebook["cells"][8]["source"] = notebook["cells"][8]["source"].replace(
    "if WORK_CODE_DIR not in sys.path:", "WORK_CODE_DIR = str(WORK_CODE_DIR)\nif WORK_CODE_DIR not in sys.path:"
).replace("Path(OUTPUT_DIR)", "OUTPUT_DIR").replace("open(SUBMISSION_PATH", "open(str(SUBMISSION_PATH)")
notebook["cells"][8]["outputs"] = []
notebook["cells"][8]["execution_count"] = None

OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
