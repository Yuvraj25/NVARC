"""Build the clean 48-task global-LoRA training notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-global48-vanilla-lora.ipynb"

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


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    cells = [
        markdown(
            """# ARC26 clean global LoRA on the validation 48

This notebook trains one rank-256 LoRA on leave-one-out records made only from
the provided training pairs of the established 48 validation tasks. It starts
from the published Vanilla V2 model: no Repair adapter, no validation test
outputs, and no per-puzzle TTFT. Twenty deterministic views per eligible task
give 960 records before any one-pair exclusion. The adapter includes the same
seven projection families, `embed_tokens`, and `lm_head` used by Vanilla V2."""
        ),
        code(
            f"""import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["OMP_NUM_THREADS"] = "3"

CODE_DATASET_ROOT = Path("/kaggle/input/datasets/yuvraj/arc2026")
MODEL_PATH = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
COMP_ROOT = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
VALIDATION_KEYS = {VALIDATION_KEYS!r}

WORK_ROOT = Path("/kaggle/working/arc26_global48_vanilla")
WORK_CODE_DIR = WORK_ROOT / "ARC-AGI1/qwen_baseline"
SUBSET_PATH = Path("/kaggle/working/global48_challenges.json")
OUTPUT_ROOT = Path("/kaggle/working/global48_vanilla_lora")

for path in [WORK_ROOT, OUTPUT_ROOT]:
    shutil.rmtree(path, ignore_errors=True)
SUBSET_PATH.unlink(missing_ok=True)
shutil.copytree(CODE_DATASET_ROOT, WORK_ROOT)

challenges = json.loads((COMP_ROOT / "arc-agi_evaluation_challenges.json").read_text())
missing = sorted(set(VALIDATION_KEYS) - set(challenges))
if missing:
    raise KeyError(f"Validation keys missing from evaluation challenges: {{missing}}")
subset = {{key: challenges[key] for key in VALIDATION_KEYS}}
SUBSET_PATH.write_text(json.dumps(subset))
print("global task count =", len(subset))
print("global nominal records =", len(subset) * 20)
"""
        ),
        code(
            """MODERN_UTILITY_ROOT = Path(
    "/kaggle/usr/lib/notebooks/yuvraj/pip_install_unsloth_ddp_repair"
)
fa2_candidates = sorted(set(
    path.parent.parent
    for root in (Path("/kaggle/input"), Path("/kaggle/usr/lib/notebooks"))
    if root.exists()
    for path in root.glob("**/flash_attn/__init__.py")
))
if not MODERN_UTILITY_ROOT.is_dir():
    raise FileNotFoundError(MODERN_UTILITY_ROOT)
if len(fa2_candidates) != 1:
    raise RuntimeError(f"Expected one Torch-2.11 FlashAttention root, found {fa2_candidates}")
FA2_ROOT = fa2_candidates[0]

environment = os.environ.copy()
old_parts = [
    part for part in environment.get("PYTHONPATH", "").split(os.pathsep)
    if part
    and "pip_install_unsloth_ddp_repair" not in part
    and "pip_install_unsloth_flash_patch" not in part
    and "flash_attention_cu13_torch_2_11_cp312" not in part
]
environment["PYTHONPATH"] = os.pathsep.join([
    str(FA2_ROOT), str(MODERN_UTILITY_ROOT), "/kaggle/working",
    str(WORK_CODE_DIR), *old_parts,
])
environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

command = [
    sys.executable, "-m", "torch.distributed.run", "--standalone",
    "--nproc_per_node", "4",
    str(WORK_CODE_DIR / "train_global_eval_curriculum.py"),
    "--model-path", str(MODEL_PATH),
    "--challenges-path", str(SUBSET_PATH),
    "--output-dir", str(OUTPUT_ROOT),
    "--views-per-task", "20",
    "--score-batch-size", "2",
    "--max-seq-length", "8192",
    "--epochs", "1.0",
    "--learning-rate", "2.5e-5",
    "--lora-rank", "256",
    "--expected-world-size", "4",
    "--seed", "20260815",
]
print("running:", " ".join(command), flush=True)
started = time.perf_counter()
subprocess.run(command, env=environment, check=True)
elapsed = time.perf_counter() - started
print("global wall_s =", round(elapsed, 3))
(OUTPUT_ROOT / "global_notebook_runtime.json").write_text(json.dumps({
    "global_notebook_wall_s": elapsed,
    "task_count": len(subset),
    "views_per_task": 20,
}, indent=2))
"""
        ),
        code(
            """manifest_path = OUTPUT_ROOT / "global_eval_curriculum_manifest.json"
adapter_path = OUTPUT_ROOT / "adapter"
manifest = json.loads(manifest_path.read_text())
print(json.dumps({
    "global_wall_s": elapsed,
    "used_task_count": manifest["data"]["used_task_count"],
    "skipped_task_count": manifest["data"]["skipped_lt2_train_pair_task_count"],
    "formatted_records": manifest["data"]["formatted_records"],
    "starting_adapter": manifest["model"]["repair_adapter_merged_before_scoring"],
    "lora_rank": manifest["model"]["fresh_global_lora_rank"],
    "modules_to_save": manifest["model"]["fresh_global_modules_to_save"],
}, indent=2))
for name in ["adapter_config.json", "adapter_model.safetensors"]:
    path = adapter_path / name
    if not path.is_file():
        raise FileNotFoundError(path)
print("adapter GiB =", round(sum(
    path.stat().st_size for path in adapter_path.rglob("*") if path.is_file()
) / 2**30, 3))
print("runtime =", (OUTPUT_ROOT / "global_notebook_runtime.json").read_text())
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
