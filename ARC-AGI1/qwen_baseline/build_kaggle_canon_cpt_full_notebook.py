"""Build the 11.5-hour Canon-AC continued-pretraining Kaggle notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/Users/banna/kaggle/temp/canon_cpt_full_upload")
NOTEBOOK_PATH = OUTPUT_DIR / "arc26-canon-cpt-11h30.ipynb"
SOURCE_FILES = [
    "arc_canon.py",
    "nvarc_continued_pretraining.py",
    "train_canon_continued_pretraining.py",
]
EXPECTED_HASHES = {
    name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    for name in SOURCE_FILES
}


def notebook_cell(kind: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = [
        notebook_cell(
            "markdown",
            """# ARC Canon-AC continued pretraining — 11h30

Fresh continued pretraining from the untouched published 16-token Qwen ARC checkpoint. The run lazily samples the released 3.2M-record corpus using 40% NVARC full, 35% NVARC training, 15% ARC-AGI-2 training, 5% RE-ARC, 3% ConceptARC, and 2% MiniARC. Every assistant output receives loss. Canon-AC trains alone for the first 30% of the wall-clock budget; Canon-AC and a fresh rank-256 transformer LoRA train jointly for the remainder. There is no repair data, reverse-NLL pass, TTFT, or inference.""",
        ),
        notebook_cell(
            "code",
            f"""from pathlib import Path

EXPECTED_HASHES = {EXPECTED_HASHES!r}
MODEL_PATH = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')
CORPUS_ROOT = Path('/kaggle/input/datasets/sorokin/nvarc-augmented-puzzles')
OUTPUT_DIR = Path('/kaggle/working/canon_continued_pretraining')
MODERN_UTILITY_ROOT = Path('/kaggle/usr/lib/notebooks/yuvraj/pip_install_unsloth_ddp_repair')""",
        ),
        notebook_cell(
            "code",
            """import hashlib
import json
import os
import shutil
import subprocess
import sys

code_candidates = [
    Path('/kaggle/input/datasets/yuvraj/arc2026'),
    Path('/kaggle/input/arc2026'),
]
CODE_ROOT = next((path for path in code_candidates if path.is_dir()), None)
if CODE_ROOT is None:
    raise FileNotFoundError(code_candidates)
SOURCE_DIR = CODE_ROOT / 'ARC-AGI1/qwen_baseline'
observed = {
    name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
    for name in EXPECTED_HASHES
}
if observed != EXPECTED_HASHES:
    raise RuntimeError(f'arc2026 source mismatch: expected={EXPECTED_HASHES} observed={observed}')

fa2_candidates = sorted(set(
    path.parent.parent
    for root in (Path('/kaggle/input'), Path('/kaggle/usr/lib/notebooks'))
    if root.exists()
    for path in root.glob('**/flash_attn/__init__.py')
))
if len(fa2_candidates) != 1:
    raise RuntimeError(f'Expected one FlashAttention root, found {fa2_candidates}')
FA2_ROOT = fa2_candidates[0]

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
environment = os.environ.copy()
old_parts = [
    part for part in environment.get('PYTHONPATH', '').split(os.pathsep)
    if part and 'pip_install_unsloth_' not in part and 'flash_attention_' not in part
]
environment.update({
    'PYTHONPATH': os.pathsep.join([
        str(FA2_ROOT), str(MODERN_UTILITY_ROOT), str(SOURCE_DIR), *old_parts
    ]),
    'PYTHONPYCACHEPREFIX': '/kaggle/working/python_cache',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'UNSLOTH_DISABLE_STATISTICS': '1',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_ENABLE_HF_TRANSFER': '0',
    'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas',
    'OMP_NUM_THREADS': '3',
})
gpu_names = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
).strip().splitlines()
if len(gpu_names) != 4 or any('L4' not in name for name in gpu_names):
    raise RuntimeError(f'Expected four L4 GPUs, observed {gpu_names}')
print('GPUs =', gpu_names)
print('code =', CODE_ROOT)
print('corpus =', CORPUS_ROOT)
print('utility =', MODERN_UTILITY_ROOT)
print('flash attention =', FA2_ROOT)""",
        ),
        notebook_cell(
            "code",
            """import time

command = [
    sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '4',
    str(SOURCE_DIR / 'train_canon_continued_pretraining.py'),
    '--model-path', str(MODEL_PATH),
    '--corpus-root', str(CORPUS_ROOT),
    '--output-dir', str(OUTPUT_DIR),
    '--wall-clock-hours', '11.5',
    '--canon-only-fraction', '0.30',
    '--checkpoint-hours', '3.0',
    '--virtual-records', '3255481',
    '--max-seq-length', '8192',
    '--learning-rate', '3e-5',
    '--lora-rank', '256',
    '--gradient-accumulation-steps', '1',
    '--expected-world-size', '4',
    '--seed', '20260827',
]
started = time.perf_counter()
print('running:', ' '.join(command), flush=True)
subprocess.run(command, env=environment, check=True)
print('notebook training command seconds =', round(time.perf_counter() - started, 2))""",
        ),
        notebook_cell(
            "code",
            """checkpoint = OUTPUT_DIR / 'checkpoint'
required = ['adapter_config.json', 'adapter_model.safetensors', 'canon_ac.pt', 'progress.json']
missing = [name for name in required if not (checkpoint / name).is_file()]
if missing:
    raise FileNotFoundError(missing)
summary = {
    'checkpoint_files': {
        name: (checkpoint / name).stat().st_size for name in required
    },
    'progress': json.loads((checkpoint / 'progress.json').read_text()),
    'manifest': json.loads((OUTPUT_DIR / 'run_manifest.json').read_text()),
}
(Path('/kaggle/working') / 'canon_cpt_final_summary.json').write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\\n'
)
print(json.dumps(summary, indent=2, sort_keys=True))""",
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    metadata = {
        "id": "yuvraj/arc26-canon-cpt-11h30",
        "title": "[ARC26] Canon-AC continued pretraining 11h30",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "yuvraj/arc2026",
            "sorokin/nvarc-augmented-puzzles",
        ],
        "kernel_sources": [
            "yuvraj/pip-install-unsloth-ddp-repair",
            "yuvraj/flash-attention-cu13-torch-2-11-cp312",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": [
            "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4",
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
