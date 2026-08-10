#!/usr/bin/env python3
"""Build the private Kaggle notebook for the Unsloth multi-token probe."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
KAGGLE_ROOT = HERE.parents[2]
OUTPUT_DIR = KAGGLE_ROOT / "temp" / "kaggle_unsloth_multitoken_probe"
NOTEBOOK_PATH = OUTPUT_DIR / "arc26-unsloth-multitoken-parity.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    patch_source = (HERE / "patch_unsloth_qwen3_multitoken.py").read_text()
    probe_source = (HERE / "probe_unsloth_qwen3_multitoken.py").read_text()
    arc_search_source = (HERE / "arc_search.py").read_text()
    search_source = (HERE / "arc_search_multitoken.py").read_text()
    dfs_probe_source = (HERE / "probe_unsloth_multitoken_dfs.py").read_text()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "# ARC26 Unsloth cached multi-token parity\n\n"
                "Inference-only complete candidate-pool gate. This notebook patches the "
                "pinned winner Qwen3 fast path before importing Unsloth, then compares "
                "vanilla DFS with a conservative cached multi-token DFS at threshold 0.1. "
                "It reuses a retained adapter and does not train."
            ),
        },
        code_cell(
            """import os
from pathlib import Path

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["PYTHONUNBUFFERED"] = "1"

CODE_DATASET_ROOT = Path("/kaggle/input/datasets/yuvraj/arc2026")
MODEL_PATH = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
TEST_PATH = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
ADAPTER_PATH = Path("/kaggle/input/notebooks/yuvraj/arc26-2025-winning-solution-v1-455107/sglang_adapters/136b0064")
WORK_CODE_DIR = Path("/kaggle/working/qwen_baseline_multitoken_probe")
RESULT_PATH = Path("/kaggle/working/multitoken_probe_output.txt")

for path in (CODE_DATASET_ROOT, MODEL_PATH, TEST_PATH, ADAPTER_PATH):
    assert path.exists(), path
assert Path(os.environ["TRITON_PTXAS_PATH"]).exists()
print("model_path =", MODEL_PATH)
print("test_path =", TEST_PATH)
print("adapter_path =", ADAPTER_PATH)
"""
        ),
        code_cell(
            """import shutil

candidates = [
    CODE_DATASET_ROOT / "ARC-AGI1" / "qwen_baseline",
    CODE_DATASET_ROOT / "qwen_baseline",
]
source_code_dir = next((path for path in candidates if path.is_dir()), None)
assert source_code_dir is not None, candidates
shutil.rmtree(WORK_CODE_DIR, ignore_errors=True)
shutil.copytree(source_code_dir, WORK_CODE_DIR)
print("copied_from =", source_code_dir)
print("copied_to =", WORK_CODE_DIR)
"""
        ),
        code_cell(
            "from pathlib import Path\n"
            f"Path(WORK_CODE_DIR / 'patch_unsloth_qwen3_multitoken.py').write_text({patch_source!r})\n"
            f"Path(WORK_CODE_DIR / 'probe_unsloth_qwen3_multitoken.py').write_text({probe_source!r})\n"
            f"Path(WORK_CODE_DIR / 'arc_search.py').write_text({arc_search_source!r})\n"
            f"Path(WORK_CODE_DIR / 'arc_search_multitoken.py').write_text({search_source!r})\n"
            f"Path(WORK_CODE_DIR / 'probe_unsloth_multitoken_dfs.py').write_text({dfs_probe_source!r})\n"
            "print('embedded probe sources written')\n"
        ),
        code_cell(
            """import importlib.util
import shutil

spec = importlib.util.find_spec("unsloth")
assert spec is not None and spec.submodule_search_locations
MOUNTED_UNSLOTH_PACKAGE_DIR = Path(next(iter(spec.submodule_search_locations)))
QWEN3_SOURCE = MOUNTED_UNSLOTH_PACKAGE_DIR / "models" / "qwen3.py"
source_text = QWEN3_SOURCE.read_text()
print("mounted_unsloth_package_dir =", MOUNTED_UNSLOTH_PACKAGE_DIR)
print("qwen3_source =", QWEN3_SOURCE)
print("winner_flash_patch_present =", "A = flash_attn_func(Qnn, Knn, Vnn)" in source_text)
print("multitoken_patch_present_before_run =", "ARC_QWEN3_MULTITOKEN_CACHE_PATCH_V1" in source_text)
assert "A = flash_attn_func(Qnn, Knn, Vnn)" in source_text, "Winner FlashAttention patch is not mounted"

WRITABLE_UNSLOTH_PARENT = Path("/kaggle/working/unsloth_multitoken_stack")
UNSLOTH_PACKAGE_DIR = WRITABLE_UNSLOTH_PARENT / "unsloth"
shutil.rmtree(WRITABLE_UNSLOTH_PARENT, ignore_errors=True)
shutil.copytree(MOUNTED_UNSLOTH_PACKAGE_DIR, UNSLOTH_PACKAGE_DIR)
assert (UNSLOTH_PACKAGE_DIR / "models" / "qwen3.py").is_file()
print("writable_unsloth_package_dir =", UNSLOTH_PACKAGE_DIR)
"""
        ),
        code_cell(
            """import subprocess
import sys

cmd = [
    sys.executable,
    "probe_unsloth_multitoken_dfs.py",
    "--unsloth-package-dir", str(UNSLOTH_PACKAGE_DIR),
    "--model-path", str(MODEL_PATH),
    "--adapter-path", str(ADAPTER_PATH),
    "--test-path", str(TEST_PATH),
    "--puzzle-key", "136b0064",
    "--batch-size", "4",
    "--prob-threshold", "0.1",
    "--max-new-tokens", "256",
    "--repeat-len", "9",
    "--path-seconds", "240",
]
print("running =", " ".join(cmd), flush=True)
probe_env = os.environ.copy()
probe_env["PYTHONPATH"] = str(WRITABLE_UNSLOTH_PARENT) + os.pathsep + probe_env.get("PYTHONPATH", "")
completed = subprocess.run(
    cmd,
    cwd=WORK_CODE_DIR,
    env=probe_env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
RESULT_PATH.write_text(completed.stdout)
print(completed.stdout)
print("returncode =", completed.returncode)
print("result_path =", RESULT_PATH)
completed.check_returncode()
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))

    metadata = {
        "id": "yuvraj/arc26-unsloth-multi-token-parity",
        "title": "[ARC26] Unsloth Multi-token Parity",
        "code_file": NOTEBOOK_PATH.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": ["yuvraj/arc2026"],
        "kernel_sources": [
            "yuvraj/notebookc4ca2ea220",
            "sorokin/pip-install-unsloth-flash-patch",
            "yuvraj/arc26-2025-winning-solution-v1-455107",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
