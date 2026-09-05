"""Build the retained-eight validation clone of the dual puzzle-LoRA notebook."""

from __future__ import annotations

import json
from pathlib import Path


ARC_ROOT = Path(__file__).resolve().parents[1]
SOURCE = ARC_ROOT / "arc26-dual-lora-48x2-q9-16x2-submit-competition.ipynb"
TARGET = ARC_ROOT / "arc26-dual-lora-48x2-q9-16x2-validation8.ipynb"
UPLOAD_DIR = Path("/Users/banna/kaggle/temp/kaggle_dual_lora_48x2_validation8")
VALIDATION_KEYS = [
    "0934a4d8",
    "135a2760",
    "136b0064",
    "13e47133",
    "142ca369",
    "16b78196",
    "16de56c4",
    "1818057f",
]


def clean(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def main():
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = """# ARC26 dual puzzle-LoRA retained-eight validation

Validation-only clone of the submission-ready dual-LoRA Version 5. The production stack and recipe are unchanged: two independently seeded rank-256 puzzle LoRAs, 48 updates per branch, 16 q9 views per branch, and one pooled score_kgmon selection.
"""

    config = notebook["cells"][2]["source"]
    config = config.replace('MODE = "submit_competition"', 'MODE = "validation"')
    config = config.replace("VALIDATION_KEYS = None", f"VALIDATION_KEYS = {VALIDATION_KEYS!r}")
    config = config.replace(
        'MANIFEST_PATH = "/kaggle/working/dual_lora_48x2_manifest.json"',
        'MANIFEST_PATH = "/kaggle/working/dual_lora_48x2_validation8_manifest.json"\n'
        'SUMMARY_PATH = "/kaggle/working/dual_lora_48x2_validation8_summary.json"',
    )
    notebook["cells"][2]["source"] = config

    scoring = notebook["cells"][8]["source"]
    scoring = scoring.replace(
        "data = ArcDataset.from_file(TEST_PATH)\n",
        "data = ArcDataset.from_file(TEST_PATH, keys=SELECTED_KEYS)\n",
    )
    scoring = scoring.replace(
        '        print("validation_score =", data.validate_submission(submission))\n',
        '''        import numpy as np

        selected_score = data.validate_submission(submission)
        outputs_per_task = {
            key: len(data.queries[key]["test"]) for key in VALIDATION_KEYS
        }
        oracle_outputs = []
        candidate_records = 0
        for base_key, candidates in combined.decoded_results.items():
            gold = split_data.replies[base_key][0]
            candidate_records += len(candidates)
            if any(np.array_equal(record["solution"], gold) for record in candidates.values()):
                oracle_outputs.append(base_key)
        oracle_score = sum(
            1 / outputs_per_task[base_key.rsplit("_", 1)[0]]
            for base_key in oracle_outputs
        )

        selected_correct_outputs = []
        per_key = {}
        for key in VALIDATION_KEYS:
            correct = 0
            for output_index, gold in enumerate(data.replies[key]):
                attempts = submission[key][output_index]
                if any(np.array_equal(attempt, gold) for attempt in attempts.values()):
                    selected_correct_outputs.append(f"{key}_{output_index}")
                    correct += 1
            task_oracle = [value for value in oracle_outputs if value.startswith(key + "_")]
            per_key[key] = {
                "selected_correct_outputs": correct,
                "oracle_correct_outputs": len(task_oracle),
                "outputs": outputs_per_task[key],
            }

        summary = {
            "recipe": "dual_lora_48x2_q9_16x2",
            "validation_keys": VALIDATION_KEYS,
            "requested_tasks": len(VALIDATION_KEYS),
            "requested_outputs": len(split_data.keys),
            "branch_output_counts": {
                name: len(decoder.decoded_results)
                for name, decoder in branch_decoders.items()
            },
            "combined_output_keys": len(combined.decoded_results),
            "candidate_records": candidate_records,
            "selected_score": selected_score,
            "selected_correct_outputs": sorted(selected_correct_outputs),
            "oracle_score": oracle_score,
            "oracle_outputs": sorted(oracle_outputs),
            "per_key": per_key,
            "dual_branch_wall_seconds": dual_wall_seconds,
            "train_steps_per_branch": 8 * TRAIN_COLOR_PERMUTATIONS,
            "inference_views_per_branch": 8 * EVAL_COLOR_PERMUTATIONS,
        }
        Path(SUMMARY_PATH).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
        print("validation_score =", selected_score)
        print(json.dumps(summary, indent=2, sort_keys=True))
''',
    )
    notebook["cells"][8]["source"] = scoring

    clean(notebook)
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / TARGET.name).write_bytes(TARGET.read_bytes())
    metadata = {
        "id": "yuvraj/arc26-dual-lora-48x2-q9-16x2-validation8",
        "title": "[ARC26] Dual LoRA 48x2 q9 16x2 validation8",
        "code_file": TARGET.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": ["yuvraj/arc2026"],
        "kernel_sources": [
            "yuvraj/notebookc4ca2ea220",
            "sorokin/pip-install-unsloth-flash-patch",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": [
            "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
        ],
        "docker_image_pinning_type": "original",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    }
    (UPLOAD_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(TARGET)
    print(UPLOAD_DIR)


if __name__ == "__main__":
    main()
