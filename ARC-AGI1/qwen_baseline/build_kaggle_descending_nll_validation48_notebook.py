"""Build the Vanilla V2 descending-initial-NLL validation-48 experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_BUILDER = HERE / "build_kaggle_vanilla_v2_current_smoke4_notebook.py"
BASE_DIR = Path("/Users/banna/kaggle/temp/kaggle_vanilla_v2_current_smoke4")
OUTPUT_DIR = Path("/Users/banna/kaggle/temp/kaggle_descending_nll_validation48")
NOTEBOOK_NAME = "arc26-vanilla-v2-descending-nll-q9-24-validation48.ipynb"
SOURCE_FILES = ("arc_solver.py", "starter.py")
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


def main():
    spec = importlib.util.spec_from_file_location("vanilla_smoke4_builder", BASE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()

    notebook = json.loads((BASE_DIR / module.NOTEBOOK_NAME).read_text())
    notebook["cells"][0]["source"] = """# ARC26 Vanilla V2 descending-NLL validation48

Scores each puzzle's fixed 128 augmented TTFT rows once with the global model,
sorts them by descending assistant-token mean NLL, and trains the local rank-256
LoRA in that fixed sequential order. Evaluation uses q9, 24 views, threshold
0.2, score_kgmon, and inference batch 6.
"""

    base_keys = repr(module.SMOKE_KEYS)
    validation_keys = repr(VALIDATION_KEYS)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"].replace(base_keys, validation_keys)
        source = source.replace("smoke4", "descending_nll_validation48")
        source = source.replace("VALIDATION_END_TIME_HOURS = 1.5", "VALIDATION_END_TIME_HOURS = 5.0")
        if "--eval-color-permutations" in source and "--eval-batch-size" not in source:
            source = source.replace(
                '        "--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),\n',
                '        "--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),\n'
                '        "--eval-batch-size", "6",\n'
                '        "--ttft-order", "descending_nll",\n',
            )
        if 'assert "--eval-color-permutations" in starter_source' in source:
            source = source.replace(
                'assert "--eval-color-permutations" in starter_source\n',
                'assert "--eval-color-permutations" in starter_source\n'
                'assert "--eval-batch-size" in starter_source\n'
                'assert "--ttft-order" in starter_source\n',
            )
        cell["source"] = source

    source_hashes = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    notebook["cells"][2]["source"] += f'\nEXPECTED_DESCENDING_NLL_SOURCE_HASHES = {source_hashes!r}\n'

    copy_cell = notebook["cells"][5]["source"]
    copy_cell = copy_cell.replace(
        "shutil.copytree(src, dst)\n",
        "shutil.copytree(src, dst)\n\n"
        "import hashlib\n"
        "observed_hashes = {}\n"
        "for name, expected_hash in EXPECTED_DESCENDING_NLL_SOURCE_HASHES.items():\n"
        "    source_path = Path(WORK_CODE_DIR, name)\n"
        "    observed_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()\n"
        "    assert observed_hash == expected_hash, (name, observed_hash, expected_hash)\n"
        "    observed_hashes[name] = observed_hash\n"
        "print('arc2026 descending-NLL source hash check passed', observed_hashes)\n",
    )
    notebook["cells"][5]["source"] = copy_cell

    scoring = notebook["cells"][8]["source"]
    scoring += """

order_dir = Path(f"{OUTPUT_DIR}_ttft_descending_nll")
order_files = sorted(order_dir.glob("*.json")) if order_dir.exists() else []
print("descending_nll_manifest_count =", len(order_files))
assert len(order_files) == len(SELECTED_KEYS), (len(order_files), len(SELECTED_KEYS))
"""
    notebook["cells"][8]["source"] = scoring

    metadata = json.loads((BASE_DIR / "kernel-metadata.json").read_text())
    metadata.update(
        {
            "id": "yuvraj/arc26-vanilla-v2-descending-nll-q9-24-validation48",
            "title": "[ARC26] Vanilla V2 descending NLL q9 24 validation48",
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
