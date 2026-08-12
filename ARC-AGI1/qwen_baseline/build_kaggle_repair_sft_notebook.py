"""Build a Kaggle notebook for the repair-SFT integration smoke or full pilot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_NAME = os.environ.get("ARC_REPAIR_SFT_NOTEBOOK_NAME", "arc26-repair-sft-smoke.ipynb")
OUTPUT = HERE.parent / OUTPUT_NAME
DATASET_SLUG = os.environ.get("ARC_REPAIR_SFT_DATASET_SLUG", "arc26-repair-failures-10240")
MAX_TRAIN_EXAMPLES = os.environ.get("ARC_REPAIR_SFT_MAX_TRAIN_EXAMPLES", "8")
EPOCHS = os.environ.get("ARC_REPAIR_SFT_EPOCHS", "0.25")
DIAGNOSTIC_EXAMPLES = int(os.environ.get("ARC_REPAIR_SFT_DIAGNOSTIC_EXAMPLES", "2"))
ROLLOUT_EXAMPLES = int(os.environ.get("ARC_REPAIR_SFT_ROLLOUT_EXAMPLES", "1"))
OUTPUT_DIR_NAME = os.environ.get("ARC_REPAIR_SFT_OUTPUT_DIR", "repair_sft_smoke")
SOURCE_FILES = (
    "repair_mining.py",
    "repair_sft.py",
    "train_repair_adapter.py",
    "arc_solver.py",
)
SOURCE_HASHES = {
    name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES
}


def cell_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(source),
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(source),
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


max_train_literal = "None" if MAX_TRAIN_EXAMPLES.lower() == "none" else str(int(MAX_TRAIN_EXAMPLES))
cells = [
    markdown_cell(
        """# ARC repair SFT

This notebook trains a completion-only LoRA on genuine repair failures, 15% ordinary solve replay, and 1% zero-mask no-ops. The previous wrong assistant candidate is context only and receives no loss. One `<REPAIR>` token is added and initialized from the mean of the existing structural-token embeddings."""
    ),
    code_cell(
        f"""RUN_EXPERIMENT = True
DATASET_SLUG = {DATASET_SLUG!r}
MAX_TRAIN_EXAMPLES = {max_train_literal}
EPOCHS = {float(EPOCHS)!r}
DIAGNOSTIC_EXAMPLES = {DIAGNOSTIC_EXAMPLES}
ROLLOUT_EXAMPLES = {ROLLOUT_EXAMPLES}
EXPECTED_SOURCE_HASHES = {SOURCE_HASHES!r}

MODEL_PATH = '/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
OUTPUT_DIR = '/kaggle/working/{OUTPUT_DIR_NAME}'

print('dataset =', DATASET_SLUG, 'max train =', MAX_TRAIN_EXAMPLES, 'epochs =', EPOCHS)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import hashlib
    import subprocess
    from pathlib import Path

    gpu_names = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        text=True,
    ).strip().splitlines()
    if len(gpu_names) != 4 or any('L4' not in name for name in gpu_names):
        raise RuntimeError(
            f'Repair SFT requires the pinned 4xL4 environment; observed GPUs={gpu_names}'
        )
    print('verified GPUs =', gpu_names)

    code_candidates = [
        Path('/kaggle/input/datasets/yuvraj/arc2026/ARC-AGI1/qwen_baseline'),
        Path('/kaggle/input/arc2026/ARC-AGI1/qwen_baseline'),
    ]
    CODE_DIR = next((path for path in code_candidates if path.is_dir()), None)
    if CODE_DIR is None:
        raise FileNotFoundError(f'Could not find arc2026 code under {code_candidates}')
    observed_hashes = {
        name: hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest()
        for name in EXPECTED_SOURCE_HASHES
    }
    if observed_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(
            'Mounted arc2026 code is stale or unexpected. '
            f'expected={EXPECTED_SOURCE_HASHES} observed={observed_hashes}'
        )

    data_candidates = [
        Path('/kaggle/input/datasets/yuvraj') / DATASET_SLUG,
        Path('/kaggle/input') / DATASET_SLUG,
    ]
    DATA_DIR = next((path for path in data_candidates if path.is_dir()), None)
    if DATA_DIR is None:
        raise FileNotFoundError(f'Could not find repair dataset under {data_candidates}')
    print('verified code =', CODE_DIR)
    print('repair data =', DATA_DIR)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import subprocess
    import sys

    command = [
        sys.executable,
        str(CODE_DIR / 'train_repair_adapter.py'),
        '--model-path', MODEL_PATH,
        '--train-path', str(DATA_DIR / 'repair_failures.train.jsonl'),
        '--dev-path', str(DATA_DIR / 'repair_failures.dev.jsonl'),
        '--output-dir', OUTPUT_DIR,
        '--epochs', str(EPOCHS),
        '--solve-replay-fraction', '0.15',
        '--noop-fraction', '0.01',
        '--diagnostic-examples', str(DIAGNOSTIC_EXAMPLES),
        '--rollout-examples', str(ROLLOUT_EXAMPLES),
    ]
    if MAX_TRAIN_EXAMPLES is not None:
        command.extend(['--max-train-examples', str(MAX_TRAIN_EXAMPLES)])
    print('running:', ' '.join(command))
    subprocess.run(command, cwd=str(CODE_DIR), check=True)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import json
    from pathlib import Path

    manifest_path = Path(OUTPUT_DIR) / 'repair_sft_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['tokenizer']['old_vocab_size'] == 16
    assert manifest['tokenizer']['new_vocab_size'] == 17
    assert manifest['tokenizer']['repair_token_id'] == 16
    assert manifest['mixture']['requested_fractions'] == {
        'repair_failure': 0.84,
        'solve_replay': 0.15,
        'repair_noop': 0.01,
    }
    print(json.dumps(manifest, indent=2))"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(OUTPUT)
