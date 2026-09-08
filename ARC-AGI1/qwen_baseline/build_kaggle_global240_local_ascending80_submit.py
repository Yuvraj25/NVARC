"""Build Global-x20 plus 160-row ascending-NLL local TTFT submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE = Path(
    "/Users/banna/kaggle/temp/arc26-global240-local80-q9-24-submit/"
    "arc26-global240-local80-q9-24-submit.ipynb"
)
EXPECTED_SOURCE_CELLS_SHA256 = "4d69adc2813d7abf9f5411387dda0cdcf4381deff63207240d964f8d10f8d130"
OUTPUT_DIR = Path("/Users/banna/kaggle/temp/kaggle_global240_local_ascending80_submit")
NOTEBOOK_NAME = "arc26-global240-x20-local-ascending80-q9-24-submit.ipynb"
EXPECTED_CODE_HASHES = {
    "starter.py": "08382592376ea6cdc347eca5580b4af8ebfd17d88e990c9be608d651bd87ca50",
    "arc_solver.py": "fada5cefe52044d07a5d80b0fd64c07f89a8a485c715de915be8978f56078e63",
    "train_global_eval_curriculum.py": "299c72cdf43fc734cc7752aad03f1a7bddd097acab16d5639c21a62e0b855757",
    "global_eval_curriculum.py": "9764f4a397ca79e5fbb6f9042831911bfd452981772b23ec8a0307a06d050c27",
}


def source_cells_sha256(notebook):
    payload = json.dumps(
        [(cell["cell_type"], cell["source"]) for cell in notebook["cells"]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def main():
    notebook = json.loads(SOURCE.read_text())
    observed_source_hash = source_cells_sha256(notebook)
    if observed_source_hash != EXPECTED_SOURCE_CELLS_SHA256:
        raise RuntimeError(f"Global240 source drift: {observed_source_hash}")

    notebook["cells"][0]["source"] = """# ARC26 Global x20 plus ascending-NLL local80 submission

The completed Global-x20 plus local80 production stack, with the local stage
changed to 160 newly sampled TTFT augmentations, batch size 2, one ascending-NLL
easy-to-hard pass (80 optimizer steps), and 10% warmup followed by constant LR.
Global training, merge, q9 24-view inference, threshold 0.2, and score_kgmon are
unchanged. EMA is disabled.
"""

    config = notebook["cells"][2]["source"]
    config = replace_once(
        config,
        "LOCAL_TTFT_STEPS = 80\n",
        "LOCAL_TRAIN_COLOR_PERMUTATIONS = 20\n"
        "LOCAL_TRAIN_BATCH_SIZE = 2\n"
        "LOCAL_TTFT_STEPS = 8 * LOCAL_TRAIN_COLOR_PERMUTATIONS // LOCAL_TRAIN_BATCH_SIZE\n"
        "LOCAL_TTFT_ORDER = \"ascending_nll\"\n"
        "LOCAL_TTFT_LR_SCHEDULER_TYPE = \"constant_with_warmup\"\n",
    )
    config = config.replace("global240_local80_q9_24", "global240_local_ascending80_q9_24")
    config += f"\nEXPECTED_CODE_HASHES = {EXPECTED_CODE_HASHES!r}\n"
    notebook["cells"][2]["source"] = config

    setup = notebook["cells"][4]["source"].replace(
        "/kaggle/working/global240_adapter_compat",
        "/kaggle/working/global240_ascending80_adapter_compat",
    )
    notebook["cells"][4]["source"] = setup

    preflight = notebook["cells"][5]["source"]
    preflight = replace_once(
        preflight,
        "required_files = [\n",
        "observed_code_hashes = {\n"
        "    name: __import__('hashlib').sha256(Path(WORK_CODE_DIR, name).read_bytes()).hexdigest()\n"
        "    for name in EXPECTED_CODE_HASHES\n"
        "}\n"
        "if observed_code_hashes != EXPECTED_CODE_HASHES:\n"
        "    raise RuntimeError({'expected': EXPECTED_CODE_HASHES, 'observed': observed_code_hashes})\n"
        "\nrequired_files = [\n",
    )
    preflight = replace_once(
        preflight,
        'assert "--eval-color-permutations" in starter_source\n',
        'assert "--eval-color-permutations" in starter_source\n'
        'assert "--train-color-permutations" in starter_source\n'
        'assert "--train-batch-size" in starter_source\n'
        'assert "ascending_nll" in starter_source\n'
        'assert "--ttft-lr-scheduler-type" in starter_source\n',
    )
    notebook["cells"][5]["source"] = preflight

    merge_cell = notebook["cells"][7]["source"].replace(
        "/kaggle/working/global240_adapter_compat",
        "/kaggle/working/global240_ascending80_adapter_compat",
    )
    notebook["cells"][7]["source"] = merge_cell

    command = notebook["cells"][9]["source"]
    command = replace_once(
        command,
        '        "--full-sft-total-steps", str(LOCAL_TTFT_STEPS),\n',
        '        "--train-color-permutations", str(LOCAL_TRAIN_COLOR_PERMUTATIONS),\n'
        '        "--train-batch-size", str(LOCAL_TRAIN_BATCH_SIZE),\n'
        '        "--ttft-order", LOCAL_TTFT_ORDER,\n'
        '        "--ttft-lr-scheduler-type", LOCAL_TTFT_LR_SCHEDULER_TYPE,\n',
    )
    notebook["cells"][9]["source"] = command

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    assert all(
        flag in notebook["cells"][9]["source"] for flag in LOCAL_EXPECTED_FLAGS
    )
    assert "--ttft-ema" not in notebook["cells"][9]["source"]

    metadata = {
        "id": "yuvraj/arc26-global-x20-local-asc80-q9-24-submit",
        "title": "ARC26 Global x20 local asc80 q9 24 submit",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": ["yuvraj/arc2026"],
        "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": [
            "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
        ],
        "docker_image_pinning_type": "original",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / NOTEBOOK_NAME).write_text(json.dumps(notebook, indent=1) + "\n")
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUTPUT_DIR)


LOCAL_EXPECTED_FLAGS = {
    '"--train-color-permutations", str(LOCAL_TRAIN_COLOR_PERMUTATIONS)',
    '"--train-batch-size", str(LOCAL_TRAIN_BATCH_SIZE)',
    '"--ttft-order", LOCAL_TTFT_ORDER',
    '"--ttft-lr-scheduler-type", LOCAL_TTFT_LR_SCHEDULER_TYPE',
}


if __name__ == "__main__":
    main()
