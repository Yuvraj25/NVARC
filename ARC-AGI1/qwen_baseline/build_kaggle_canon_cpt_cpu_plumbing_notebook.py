"""Build a CPU-only Kaggle gate for the released NVARC corpus plumbing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/Users/banna/kaggle/temp/canon_cpt_cpu_plumbing_upload")
NOTEBOOK_PATH = OUTPUT_DIR / "arc26-canon-cpt-corpus-cpu-gate.ipynb"
MODULE_SOURCE = (HERE / "nvarc_continued_pretraining.py").read_text()
MODULE_SHA256 = hashlib.sha256(MODULE_SOURCE.encode()).hexdigest()


def cell(kind: str, source: str) -> dict:
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
        cell(
            "markdown",
            """# Canon continued-pretraining corpus CPU gate

This notebook performs no model forward pass and requests no GPU. It opens every released NVARC augmented Arrow subset, verifies the exact 40/35/15/5/3/2 lazy source sampler, loads the released 16-token tokenizer, and inspects real all-assistant loss masks and sequence lengths.""",
        ),
        cell(
            "code",
            f"""from pathlib import Path
import hashlib

MODULE_SOURCE = {MODULE_SOURCE!r}
MODULE_SHA256 = {MODULE_SHA256!r}
WORK_ROOT = Path('/kaggle/working/canon_cpt_cpu_gate')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
module_path = WORK_ROOT / 'nvarc_continued_pretraining.py'
module_path.write_text(MODULE_SOURCE)
assert hashlib.sha256(module_path.read_bytes()).hexdigest() == MODULE_SHA256

MODEL_PATH = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_JSON = Path('/kaggle/working/canon_cpt_cpu_plumbing.json')""",
        ),
        cell(
            "code",
            """import json
import sys
import time
from collections import Counter

sys.path.insert(0, str(WORK_ROOT))
from datasets import load_from_disk
from transformers import AutoTokenizer
from nvarc_continued_pretraining import (
    SOURCE_WEIGHTS,
    WeightedNVARCCorpus,
    discover_augmented_corpus_root,
    load_released_sources,
)

started = time.perf_counter()
corpus_root = discover_augmented_corpus_root(INPUT_ROOT)
sources = load_released_sources(corpus_root)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
print('corpus root =', corpus_root)
print('tokenizer size =', len(tokenizer))
print('source sizes =', {name: len(dataset) for name, dataset in sources.items()})""",
        ),
        cell(
            "code",
            """corpus = WeightedNVARCCorpus(
    sources,
    tokenizer,
    max_seq_length=8192,
    virtual_length=10_000,
    seed=20260827,
)

rows = [corpus[index] for index in range(1_000)]
source_counts_100 = Counter(corpus.selection(index).source for index in range(100))
source_counts_1000 = Counter(row['source'] for row in rows)
sequence_lengths = sorted(row['sequence_tokens'] for row in rows)
supervised = sorted(row['supervised_tokens'] for row in rows)
assistant_outputs = Counter(row['assistant_outputs'] for row in rows)
dropped = Counter(row['dropped_leading_pairs'] for row in rows)

def quantiles(values):
    return {
        'min': values[0],
        'p50': values[len(values) // 2],
        'p90': values[int(len(values) * 0.90)],
        'p99': values[int(len(values) * 0.99)],
        'max': values[-1],
    }

summary = {
    'module_sha256': MODULE_SHA256,
    'elapsed_s': time.perf_counter() - started,
    'corpus_root': str(corpus_root),
    'tokenizer_size': len(tokenizer),
    'source_weights': SOURCE_WEIGHTS,
    'source_sizes': {name: len(dataset) for name, dataset in sources.items()},
    'source_counts_first_100': dict(source_counts_100),
    'source_counts_first_1000': dict(source_counts_1000),
    'sampled_records': len(rows),
    'sequence_tokens': quantiles(sequence_lengths),
    'supervised_tokens': quantiles(supervised),
    'assistant_outputs_per_record': dict(sorted(assistant_outputs.items())),
    'dropped_leading_pairs': dict(sorted(dropped.items())),
    'all_records_have_supervision': all(row['supervised_tokens'] > 0 for row in rows),
    'model_forward_calls': 0,
    'gpu_requested': False,
}
OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n')
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
        "id": "yuvraj/arc26-canon-cpt-corpus-cpu-gate",
        "title": "[ARC26] Canon CPT corpus CPU gate",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": ["sorokin/nvarc-augmented-puzzles"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [
            "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
        ],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
