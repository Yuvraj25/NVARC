"""Build the descending-NLL validation-48 run with late bias-corrected TTFT EMA."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_BUILDER = HERE / "build_kaggle_descending_nll_validation48_notebook.py"
BASE_DIR = Path("/Users/banna/kaggle/temp/kaggle_descending_nll_validation48")
OUTPUT_DIR = Path("/Users/banna/kaggle/temp/kaggle_descending_nll_ema98_validation48")
NOTEBOOK_NAME = "arc26-desc-nll-ema98-b20-q9-24-validation48.ipynb"


def main():
    spec = importlib.util.spec_from_file_location("descending_nll_builder", BASE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()

    notebook = json.loads((BASE_DIR / module.NOTEBOOK_NAME).read_text())
    notebook["cells"][0]["source"] = """# ARC26 descending-NLL plus TTFT EMA validation48

Matches the completed descending-NLL q9/24 validation-48 run, including TTFT
batch size 2 and inference batch 6. The only behavioral addition is a
bias-corrected FP32 EMA of trainable LoRA weights with decay 0.98, sampled from
optimizer steps 20 through 64 inclusive. The normalized EMA replaces the live
LoRA weights after training, and its shadow is released before DFS inference.
"""

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"].replace(
            "descending_nll_validation48",
            "descending_nll_ema98_validation48",
        )
        if '"--ttft-order", "descending_nll",' in source:
            source = source.replace(
                '        "--ttft-order", "descending_nll",\n',
                '        "--ttft-order", "descending_nll",\n'
                '        "--ttft-ema-decay", "0.98",\n'
                '        "--ttft-ema-start-step", "20",\n',
            )
        if 'assert "--ttft-order" in starter_source' in source:
            source = source.replace(
                'assert "--ttft-order" in starter_source\n',
                'assert "--ttft-order" in starter_source\n'
                'assert "--ttft-ema-decay" in starter_source\n'
                'assert "--ttft-ema-start-step" in starter_source\n',
            )
        cell["source"] = source

    metadata = json.loads((BASE_DIR / "kernel-metadata.json").read_text())
    metadata.update(
        {
            "id": "yuvraj/arc26-desc-nll-ema98-b20-q9-24-validation48",
            "title": "ARC26 Desc NLL EMA98 b20 q9 24 validation48",
            "code_file": NOTEBOOK_NAME,
        }
    )
    metadata["dataset_sources"] = ["yuvraj/arc2026"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=1) + "\n")
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
