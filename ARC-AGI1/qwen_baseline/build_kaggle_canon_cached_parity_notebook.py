"""Build the one-prompt Canon cached/full inference parity notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-canon-cached-parity.ipynb"


def cell(kind: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main() -> None:
    cells = [
        cell(
            "markdown",
            """# ARC26 Canon cached/full parity

Mechanical one-prompt gate on `0934a4d8`. Loads the premerged Canon-CPT model and compares full-sequence, one-token cached, q9 cached, partial-q9 acceptance, and sibling-backtracking logits. This performs no TTFT and no scored DFS.""",
        ),
        cell(
            "code",
            f"""from pathlib import Path
import os, shutil, subprocess, sys

PREMERGED_ROOT = Path('/kaggle/input/arc26-canon-cpt-premerged-model/canon_cpt_premerged')
MODEL_PATH = PREMERGED_ROOT / 'model'
CANON_STATE = PREMERGED_ROOT / 'canon_ac.pt'
COMP_ROOT = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2')
WORK_ROOT = Path('/kaggle/working/canon_parity_code')
WORK_CODE_DIR = WORK_ROOT / 'ARC-AGI1/qwen_baseline'
SUMMARY_PATH = Path('/kaggle/working/canon_cached_parity_summary.json')
UTILITY_ROOT = Path('/kaggle/usr/lib/notebooks/yuvraj/pip_install_unsloth_ddp_repair')
FLASH_ROOT = Path('/kaggle/input/notebooks/yuvraj/flash-attention-cu13-torch-2-11-cp312/flash_attn_cu13_torch211_cp312')
WRITABLE_UNSLOTH_PARENT = Path('/kaggle/working/canon_q9_parity_stack')

CODE_ROOT = Path('/kaggle/input/datasets/yuvraj/arc2026')
for required in (MODEL_PATH / 'config.json', CANON_STATE, FLASH_ROOT / 'flash_attn/__init__.py'):
    if not required.is_file(): raise FileNotFoundError(required)

for path in (WORK_ROOT, WRITABLE_UNSLOTH_PARENT): shutil.rmtree(path, ignore_errors=True)
shutil.copytree(CODE_ROOT, WORK_ROOT)
shutil.copytree(UTILITY_ROOT / 'unsloth', WRITABLE_UNSLOTH_PARENT / 'unsloth')

environment = os.environ.copy()
old = [p for p in environment.get('PYTHONPATH','').split(os.pathsep) if p and 'pip_install_unsloth_' not in p and 'flash_attention_' not in p]
environment.update({{
    'PYTHONPATH': os.pathsep.join([str(FLASH_ROOT), str(WRITABLE_UNSLOTH_PARENT), str(UTILITY_ROOT), str(WORK_CODE_DIR), *old]),
    'PYTHONPYCACHEPREFIX': '/kaggle/working/python_cache',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'UNSLOTH_DISABLE_STATISTICS': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_ENABLE_HF_TRANSFER': '0',
    'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas',
    'CUDA_VISIBLE_DEVICES': '0',
}})
subprocess.run([
    sys.executable, str(WORK_CODE_DIR / 'patch_unsloth_qwen3_multitoken.py'),
    '--unsloth-package-dir', str(WRITABLE_UNSLOTH_PARENT / 'unsloth'),
], env=environment, check=True)
print('premerged model =', MODEL_PATH)
print('canon state =', CANON_STATE)
print('flash attention =', FLASH_ROOT)""",
        ),
        cell(
            "code",
            """command = [
    sys.executable, str(WORK_CODE_DIR / 'probe_canon_cached_parity.py'),
    '--model-path', str(MODEL_PATH),
    '--canon-state', str(CANON_STATE),
    '--challenges-path', str(COMP_ROOT / 'arc-agi_evaluation_challenges.json'),
    '--solutions-path', str(COMP_ROOT / 'arc-agi_evaluation_solutions.json'),
    '--puzzle-key', '0934a4d8',
    '--color-permutations', '3',
    '--threshold', '0.2',
    '--output-path', str(SUMMARY_PATH),
]
print('running:', ' '.join(command), flush=True)
subprocess.run(command, cwd=WORK_CODE_DIR, env=environment, check=True)
print(SUMMARY_PATH.read_text())""",
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
