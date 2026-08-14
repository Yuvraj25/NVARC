"""Build the bounded repaired-base versus v17 retained-eight Kaggle notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-repair-ttft-retained8.ipynb"
KEYS = [
    "0934a4d8",
    "135a2760",
    "136b0064",
    "13e47133",
    "142ca369",
    "16b78196",
    "16de56c4",
    "1818057f",
]
SOURCE_FILES = (
    "arc_solver.py",
    "starter.py",
    "repair_sft.py",
    "compare_candidate_runs.py",
    "probe_candidate_selection.py",
    "arc_decoder.py",
    "arc_loader.py",
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


cells = [
    markdown_cell(
        """# Repaired-base ordinary TTFT on the v17 retained eight

This is the bounded transfer test. It merges the offline repair adapter into the original 4B base in each vanilla HF worker, trains fresh rank-256 full-SFT TTFT adapters for the same eight validation puzzles, performs ordinary DFS decoding with the v17 threshold 0.1, and compares mean-KGMON top-2 and any-candidate oracle accuracy against the retained v17 pool. No repair prompt is used."""
    ),
    code_cell(
        f"""RUN_EXPERIMENT = True
KEYS = {KEYS!r}
EXPECTED_SOURCE_HASHES = {SOURCE_HASHES!r}
DFS_PROB_THRESHOLD = 0.1
EXPECTED_BASELINE = {{
    'tasks': 8,
    'outputs': 11,
    'selected_top2_score': 2.0,
    'oracle_score': 2.5,
    'candidate_samples': 184,
}}

MODEL_PATH = '/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
COMP_ROOT = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
OUTPUT_ROOT = '/kaggle/working/repair_ttft_retained8'
INFERENCE_DIR = OUTPUT_ROOT + '/inference_outputs_validation'
print('keys =', KEYS, 'dfs threshold =', DFS_PROB_THRESHOLD)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path

    gpu_names = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
    ).strip().splitlines()
    if len(gpu_names) != 4 or any('L4' not in name for name in gpu_names):
        raise RuntimeError(f'Expected four L4 GPUs, observed {gpu_names}')

    code_candidates = [
        Path('/kaggle/input/datasets/yuvraj/arc2026/ARC-AGI1/qwen_baseline'),
        Path('/kaggle/input/arc2026/ARC-AGI1/qwen_baseline'),
    ]
    CODE_DIR = next((path for path in code_candidates if path.is_dir()), None)
    if CODE_DIR is None:
        raise FileNotFoundError(f'Could not locate arc2026 source under {code_candidates}')
    observed_hashes = {
        name: hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest()
        for name in EXPECTED_SOURCE_HASHES
    }
    if observed_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(
            f'arc2026 source mismatch: expected={EXPECTED_SOURCE_HASHES} observed={observed_hashes}'
        )

    repair_candidates = [
        Path('/kaggle/input/arc26-repair-sft-full/repair_sft_full/adapter'),
        Path('/kaggle/input/notebooks/yuvraj/arc26-repair-sft-full/repair_sft_full/adapter'),
    ]
    REPAIR_ADAPTER = next((path for path in repair_candidates if path.is_dir()), None)
    if REPAIR_ADAPTER is None:
        discovered = [
            path.parent for path in Path('/kaggle/input').rglob('adapter_model.safetensors')
            if 'repair' in str(path).lower()
        ]
        REPAIR_ADAPTER = discovered[0] if len(discovered) == 1 else None
    if REPAIR_ADAPTER is None:
        raise FileNotFoundError(f'Could not uniquely locate repair adapter; candidates={repair_candidates}')
    adapter_config = json.loads((REPAIR_ADAPTER / 'adapter_config.json').read_text())
    if adapter_config.get('r') != 256:
        raise RuntimeError(f'Unexpected repair rank: {adapter_config.get("r")}')
    if set(adapter_config.get('modules_to_save') or []) != {'embed_tokens', 'lm_head'}:
        raise RuntimeError(f'Unexpected modules_to_save: {adapter_config.get("modules_to_save")}')

    baseline_candidates = [
        Path('/kaggle/input/datasets/yuvraj/arc26-v17-retained8-calibration/inference_outputs_validation'),
        Path('/kaggle/input/arc26-v17-retained8-calibration/inference_outputs_validation'),
    ]
    BASELINE_DIR = next((path for path in baseline_candidates if path.is_dir()), None)
    if BASELINE_DIR is None:
        raise FileNotFoundError(f'Could not locate retained v17 pool under {baseline_candidates}')

    CHALLENGES = Path(COMP_ROOT) / 'arc-agi_evaluation_challenges.json'
    SOLUTIONS = Path(COMP_ROOT) / 'arc-agi_evaluation_solutions.json'
    for required in (Path(MODEL_PATH), CHALLENGES, SOLUTIONS):
        if not required.exists():
            raise FileNotFoundError(required)

    output_root = Path(OUTPUT_ROOT)
    if output_root.exists():
        shutil.rmtree(output_root)
    Path(INFERENCE_DIR).mkdir(parents=True)
    print('verified GPUs =', gpu_names)
    print('verified code =', CODE_DIR)
    print('repair adapter =', REPAIR_ADAPTER)
    print('baseline candidates =', BASELINE_DIR)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import subprocess
    import sys

    baseline_json = Path(OUTPUT_ROOT) / 'baseline_selfcheck.json'
    baseline_csv = Path(OUTPUT_ROOT) / 'baseline_selfcheck.csv'
    subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / 'compare_candidate_runs.py'),
            '--baseline-dir', str(BASELINE_DIR),
            '--repaired-dir', str(BASELINE_DIR),
            '--solutions', str(SOLUTIONS),
            '--task-keys', json.dumps(KEYS),
            '--output-json', str(baseline_json),
            '--output-csv', str(baseline_csv),
        ],
        cwd=str(CODE_DIR),
        check=True,
    )
    baseline = json.loads(baseline_json.read_text())
    observed = {
        'tasks': baseline['tasks'],
        'outputs': baseline['outputs'],
        'selected_top2_score': baseline['baseline']['selected_top2_score'],
        'oracle_score': baseline['baseline']['oracle_score'],
        'candidate_samples': baseline['baseline']['candidate_samples'],
    }
    if observed != EXPECTED_BASELINE:
        raise RuntimeError(f'v17 baseline mismatch: expected={EXPECTED_BASELINE} observed={observed}')
    print('v17 baseline preflight passed:', observed)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import time

    os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
    os.environ['OMP_NUM_THREADS'] = '3'
    if not Path(os.environ['TRITON_PTXAS_PATH']).is_file():
        raise FileNotFoundError(os.environ['TRITON_PTXAS_PATH'])

    command = [
        sys.executable,
        str(CODE_DIR / 'starter.py'),
        '--test-path', str(CHALLENGES),
        '--model-path', MODEL_PATH,
        '--initial-adapter-path', str(REPAIR_ADAPTER),
        '--output-dir', INFERENCE_DIR,
        '--keys-json', json.dumps(KEYS),
        '--nprocs', '4',
        '--dfs-prob-threshold', str(DFS_PROB_THRESHOLD),
        '--ttft-method', 'full_sft',
        '--end-time', str(time.time() + 5 * 3600),
        '--profile-timings',
    ]
    print('running:', ' '.join(command))
    subprocess.run(command, cwd=str(CODE_DIR), env=os.environ.copy(), check=True)
    shutil.rmtree('/kaggle/working/unsloth_repair_ttft_offload', ignore_errors=True)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    comparison_json = Path(OUTPUT_ROOT) / 'candidate_comparison.json'
    comparison_csv = Path(OUTPUT_ROOT) / 'candidate_comparison.csv'
    subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / 'compare_candidate_runs.py'),
            '--baseline-dir', str(BASELINE_DIR),
            '--repaired-dir', INFERENCE_DIR,
            '--solutions', str(SOLUTIONS),
            '--task-keys', json.dumps(KEYS),
            '--output-json', str(comparison_json),
            '--output-csv', str(comparison_csv),
        ],
        cwd=str(CODE_DIR),
        check=True,
    )
    comparison = json.loads(comparison_json.read_text())
    if comparison['repaired']['decoded_outputs'] != comparison['repaired']['expected_outputs']:
        raise RuntimeError(f'Incomplete repaired outputs: {comparison["repaired"]}')
    if comparison['repaired']['invalid_candidate_samples'] != 0:
        raise RuntimeError(f'Invalid repaired candidates: {comparison["repaired"]}')
    adapter_digest = hashlib.sha256()
    with (REPAIR_ADAPTER / 'adapter_model.safetensors').open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            adapter_digest.update(chunk)
    run_manifest = {
        'keys': KEYS,
        'dfs_prob_threshold': DFS_PROB_THRESHOLD,
        'ttft_method': 'full_sft',
        'decoder': 'vanilla_hf_single_token_dfs',
        'repair_prompt_used': False,
        'repair_adapter': str(REPAIR_ADAPTER),
        'repair_adapter_sha256': adapter_digest.hexdigest(),
        'source_hashes': EXPECTED_SOURCE_HASHES,
        'comparison': comparison,
    }
    (Path(OUTPUT_ROOT) / 'run_manifest.json').write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + '\\n'
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))"""
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
