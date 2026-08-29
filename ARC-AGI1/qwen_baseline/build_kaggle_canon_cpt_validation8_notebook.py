"""Build the retained-eight Canon-CPT plus TTFT validation notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-canon-cpt-ttft-24-validation8.ipynb"


def _id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(source),
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(source),
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def main() -> None:
    cells = [
        markdown(
            """# ARC26 Canon-CPT + joint Canon/LoRA TTFT — retained eight

Loads the separately prepared Canon-CPT model with its global rank-256 LoRA already merged. For each retained validation puzzle, a fresh rank-256 per-task LoRA and a fresh copy of the pretrained Canon-AC weights are jointly fine-tuned for vanilla 128-step TTFT. Candidate generation uses 8 geometries x 3 colour/order views, threshold 0.2, q9 multi-token DFS, and `score_kgmon`. Canon is reset between puzzles. No repair or global-evaluation adapter is used."""
        ),
        code(
            """from pathlib import Path

PREMERGED_ROOT = Path('/kaggle/input/notebooks/yuvraj/arc26-canon-cpt-premerged-model/canon_cpt_premerged')
MODEL_PATH = PREMERGED_ROOT / 'model'
CANON_STATE = PREMERGED_ROOT / 'canon_ac.pt'
COMP_ROOT = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2')
CHALLENGES_PATH = COMP_ROOT / 'arc-agi_evaluation_challenges.json'
SOLUTIONS_PATH = COMP_ROOT / 'arc-agi_evaluation_solutions.json'
VALIDATION_KEYS = ['0934a4d8', '135a2760', '136b0064', '13e47133', '142ca369', '16b78196', '16de56c4', '1818057f']

WORK_ROOT = Path('/kaggle/working/arc26_canon_cpt_validation8')
WORK_CODE_DIR = WORK_ROOT / 'ARC-AGI1/qwen_baseline'
OUTPUT_DIR = Path('/kaggle/working/canon_cpt_validation8_candidates')
SUBMISSION_PATH = Path('/kaggle/working/canon_cpt_validation8_submission.json')
SUMMARY_PATH = Path('/kaggle/working/canon_cpt_validation8_summary.json')
MODERN_UTILITY_ROOT = Path('/kaggle/usr/lib/notebooks/yuvraj/pip_install_unsloth_ddp_repair')
WRITABLE_UNSLOTH_PARENT = Path('/kaggle/working/canon_q9_stack')

NPROCS = 4
DFS_PROB_THRESHOLD = 0.2
EVAL_COLOR_PERMUTATIONS = 3
UNSLOTH_MULTITOKEN_REPEAT_LEN = 9
END_TIME_HOURS = 3.0"""
        ),
        code(
            """import json
import os
import shutil
import subprocess
import sys

CODE_ROOT = Path('/kaggle/input/datasets/yuvraj/arc2026')
FA2_ROOT = Path('/kaggle/input/notebooks/yuvraj/flash-attention-cu13-torch-2-11-cp312/flash_attn_cu13_torch211_cp312')
for required in (
    CODE_ROOT / 'ARC-AGI1/qwen_baseline/starter.py',
    MODEL_PATH / 'config.json',
    CANON_STATE,
    MODERN_UTILITY_ROOT / 'unsloth/__init__.py',
    FA2_ROOT / 'flash_attn/__init__.py',
):
    if not required.is_file():
        raise FileNotFoundError(required)

for path in (WORK_ROOT, OUTPUT_DIR, WRITABLE_UNSLOTH_PARENT):
    shutil.rmtree(path, ignore_errors=True)
shutil.copytree(CODE_ROOT, WORK_ROOT)
OUTPUT_DIR.mkdir(parents=True)

environment = os.environ.copy()
old_parts = [
    part for part in environment.get('PYTHONPATH', '').split(os.pathsep)
    if part and 'pip_install_unsloth_' not in part and 'flash_attention_' not in part
]
environment.update({{
    'PYTHONPATH': os.pathsep.join([
        str(FA2_ROOT), str(WRITABLE_UNSLOTH_PARENT), str(MODERN_UTILITY_ROOT), '/kaggle/working',
        str(WORK_CODE_DIR), *old_parts,
    ]),
    'PYTHONPYCACHEPREFIX': '/kaggle/working/python_cache',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'UNSLOTH_DISABLE_STATISTICS': '1',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_ENABLE_HF_TRANSFER': '0',
    'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas',
    'OMP_NUM_THREADS': '3',
    'PYTHONUNBUFFERED': '1',
}})
gpu_names = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
).strip().splitlines()
if len(gpu_names) != 4 or any('L4' not in name for name in gpu_names):
    raise RuntimeError(f'Expected four L4 GPUs, observed {{gpu_names}}')
print('GPUs =', gpu_names)
print('code =', CODE_ROOT)
print('premerged model =', MODEL_PATH)
print('canon state =', CANON_STATE)
print('utility =', MODERN_UTILITY_ROOT)
print('flash attention =', FA2_ROOT)"""
        ),
        code(
            """# The q9 patch modifies two Unsloth source files, so copy only the
# mounted Unsloth package to Kaggle's writable filesystem.
writable_unsloth = WRITABLE_UNSLOTH_PARENT / 'unsloth'
shutil.copytree(MODERN_UTILITY_ROOT / 'unsloth', writable_unsloth)
subprocess.run(
    [
        sys.executable,
        str(WORK_CODE_DIR / 'patch_unsloth_qwen3_multitoken.py'),
        '--unsloth-package-dir', str(writable_unsloth),
    ],
    env=environment,
    check=True,
)
print('patched writable Unsloth =', writable_unsloth)"""
        ),
        code(
            """# Cheap source/operator checks before the scored model run.
subprocess.run(
    [sys.executable, '-m', 'unittest', 'test_arc_canon.py'],
    cwd=WORK_CODE_DIR,
    env=environment,
    check=True,
)
starter_source = (WORK_CODE_DIR / 'starter.py').read_text()
solver_source = (WORK_CODE_DIR / 'arc_solver.py').read_text()
for marker in (
    'Canon TTFT trainable parameters=',
    'load_canon_state_dict(model, default_canon_weights)',
    'Canon TTFT delta_l2=',
):
    if marker not in solver_source:
        raise RuntimeError(f'Missing Canon TTFT marker: {marker}')
print('Canon joint-TTFT preflight passed')"""
        ),
        code(
            """import time

command = [
    sys.executable, str(WORK_CODE_DIR / 'starter.py'),
    '--test-path', str(CHALLENGES_PATH),
    '--model-path', str(MODEL_PATH),
    '--output-dir', str(OUTPUT_DIR),
    '--keys-json', json.dumps(VALIDATION_KEYS),
    '--nprocs', str(NPROCS),
    '--dfs-prob-threshold', str(DFS_PROB_THRESHOLD),
    '--eval-color-permutations', str(EVAL_COLOR_PERMUTATIONS),
    '--ttft-method', 'full_sft',
    '--canon-ac-state', str(CANON_STATE),
    '--use-unsloth-multitoken-dfs',
    '--unsloth-multitoken-repeat-len', str(UNSLOTH_MULTITOKEN_REPEAT_LEN),
    '--end-time', str(time.time() + END_TIME_HOURS * 3600),
    '--profile-timings',
]
started = time.perf_counter()
print('running:', ' '.join(command), flush=True)
subprocess.run(command, cwd=WORK_CODE_DIR, env=environment, check=True)
inference_seconds = time.perf_counter() - started
print('inference seconds =', round(inference_seconds, 2))
print('candidate files =', len([path for path in OUTPUT_DIR.iterdir() if path.is_file()]))"""
        ),
        code(
            """import bz2
import pickle

import numpy as np

sys.path.insert(0, str(WORK_CODE_DIR))
from arc_decoder import ArcDecoder, hashable, score_kgmon
from arc_loader import ArcDataset

data = ArcDataset.from_file(CHALLENGES_PATH, keys=VALIDATION_KEYS).load_replies(SOLUTIONS_PATH)
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
decoder.load_decoded_results(OUTPUT_DIR)
selected = decoder.run_selection_algo(score_kgmon)
submission = data.get_submission(selected)
SUBMISSION_PATH.write_text(json.dumps(submission))
selected_score = data.validate_submission(submission)

num_outputs_per_task = {}
oracle_outputs = set()
candidate_records = 0
for basekey, candidates in decoder.decoded_results.items():
    task, output_index = basekey.split('_')
    num_outputs_per_task[task] = max(num_outputs_per_task.get(task, 0), int(output_index) + 1)
    gold = decoder.dataset.replies[basekey][0]
    candidate_records += len(candidates)
    if any(
        np.shape(record['solution']) == np.shape(gold)
        and np.array_equal(record['solution'], gold)
        for record in candidates.values()
    ):
        oracle_outputs.add(basekey)
oracle_score = sum(
    1 / num_outputs_per_task[basekey.split('_')[0]]
    for basekey in oracle_outputs
)

decoder.benchmark_selection_algos()
summary = {
    'validation_keys': VALIDATION_KEYS,
    'global_lora_merged': True,
    'canon_jointly_ttft_trained': True,
    'color_permutations': EVAL_COLOR_PERMUTATIONS,
    'geometry_views': 8,
    'total_prompts_per_task': 8 * EVAL_COLOR_PERMUTATIONS,
    'dfs': 'canon_q9_multitoken',
    'dfs_prob_threshold': DFS_PROB_THRESHOLD,
    'candidate_files': len([path for path in OUTPUT_DIR.iterdir() if path.is_file()]),
    'candidate_records': candidate_records,
    'decoded_outputs': len(decoder.decoded_results),
    'selected_score': selected_score,
    'oracle_score': oracle_score,
    'oracle_outputs': sorted(oracle_outputs),
    'inference_seconds': inference_seconds,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
shutil.make_archive('/kaggle/working/canon_cpt_validation8_candidates', 'zip', OUTPUT_DIR)
print(json.dumps(summary, indent=2, sort_keys=True))"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1))
    print(OUTPUT)


if __name__ == "__main__":
    main()
