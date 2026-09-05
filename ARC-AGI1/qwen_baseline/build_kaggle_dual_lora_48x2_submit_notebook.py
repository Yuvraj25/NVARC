"""Build the compute-bounded dual puzzle-LoRA competition notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARC_ROOT = Path(__file__).resolve().parents[1]
SOURCE = ARC_ROOT / "arc26-vanilla-v2-q9-24-submit-competition.ipynb"
TARGET = ARC_ROOT / "arc26-dual-lora-48x2-q9-16x2-submit-competition.ipynb"
UPLOAD_DIR = Path("/Users/banna/kaggle/temp/kaggle_dual_lora_48x2_submit")
CODE_FILES = ("starter.py", "arc_solver.py")
EXPECTED_CODE_HASHES = {
    name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    for name in CODE_FILES
}


def clean(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def main():
    notebook = json.loads(SOURCE.read_text())

    notebook["cells"][0]["source"] = """# ARC26 dual puzzle-LoRA 48x2 / q9 16x2 submission

Controlled self-ensemble derived from the 32.08 Vanilla V2 Version 4 launcher. Two independently seeded rank-256 per-puzzle LoRAs run concurrently: branch A on GPUs 0-1 and branch B on GPUs 2-3. Each LoRA trains on 8 geometries x 6 colour/order augmentations (48 optimizer steps) and decodes 8 geometries x 2 colour/order views (16 prompts). Candidate grids from both branches are pooled and ranked once with the unchanged score_kgmon selector. There is no LoRA-weight averaging, token-logit averaging, global LoRA, adaptive search, or silent one-branch fallback.
"""

    config = notebook["cells"][2]["source"]
    config = config.replace(
        "NPROCS = 4\n",
        "BRANCH_NPROCS = 2\n",
    )
    config = config.replace(
        "EVAL_COLOR_PERMUTATIONS = 3\n",
        "TRAIN_COLOR_PERMUTATIONS = 6\n"
        "EVAL_COLOR_PERMUTATIONS = 2\n"
        "BRANCHES = [\n"
        "    dict(name=\"a\", cuda_offset=0, train_seed=1, eval_seed=2, trainer_seed=42),\n"
        "    dict(name=\"b\", cuda_offset=2, train_seed=1001, eval_seed=1002, trainer_seed=1042),\n"
        "]\n",
    )
    config = config.replace(
        "WORK_NOTEBOOK_ROOT = \"/kaggle/working/arc2026_run_vanilla_v2_q9_24\"\n",
        "WORK_NOTEBOOK_ROOT = \"/kaggle/working/arc2026_run_dual_lora_48x2_q9_16x2\"\n",
    ).replace(
        "WORK_CODE_DIR = \"/kaggle/working/arc2026_run_vanilla_v2_q9_24/ARC-AGI1/qwen_baseline\"\n",
        "WORK_CODE_DIR = \"/kaggle/working/arc2026_run_dual_lora_48x2_q9_16x2/ARC-AGI1/qwen_baseline\"\n",
    ).replace(
        "WRITABLE_UNSLOTH_PARENT = \"/kaggle/working/vanilla_v2_q9_24_stack\"\n",
        "WRITABLE_UNSLOTH_PARENT = \"/kaggle/working/dual_lora_48x2_q9_16x2_stack\"\n"
        f"EXPECTED_CODE_HASHES = {EXPECTED_CODE_HASHES!r}\n"
        "MANIFEST_PATH = \"/kaggle/working/dual_lora_48x2_manifest.json\"\n",
    )
    if any(line.startswith("NPROCS =") for line in config.splitlines()):
        raise RuntimeError("Failed to replace Vanilla process configuration")
    notebook["cells"][2]["source"] = config

    mode = notebook["cells"][3]["source"]
    mode = mode.replace(
        "    OUTPUT_DIR = \"/kaggle/working/inference_outputs_vanilla_v2_q9_24_submit\"\n",
        "    OUTPUT_DIR_A = \"/kaggle/working/inference_outputs_dual_lora_branch_a\"\n"
        "    OUTPUT_DIR_B = \"/kaggle/working/inference_outputs_dual_lora_branch_b\"\n",
    ).replace(
        "    OUTPUT_DIR = \"/kaggle/working/inference_outputs_vanilla_v2_q9_24_validation\"\n",
        "    OUTPUT_DIR_A = \"/kaggle/working/inference_outputs_dual_lora_branch_a_validation\"\n"
        "    OUTPUT_DIR_B = \"/kaggle/working/inference_outputs_dual_lora_branch_b_validation\"\n",
    ).replace(
        "    OUTPUT_DIR = \"/kaggle/working/inference_outputs_vanilla_v2_q9_24_shortcut\"\n",
        "    OUTPUT_DIR_A = \"/kaggle/working/inference_outputs_dual_lora_branch_a_shortcut\"\n"
        "    OUTPUT_DIR_B = \"/kaggle/working/inference_outputs_dual_lora_branch_b_shortcut\"\n",
    ).replace(
        'print("output_dir =", OUTPUT_DIR)\n',
        'print("output_dirs =", OUTPUT_DIR_A, OUTPUT_DIR_B)\n',
    )
    if any(line.lstrip().startswith("OUTPUT_DIR =") for line in mode.splitlines()):
        raise RuntimeError("Failed to replace Vanilla output configuration")
    notebook["cells"][3]["source"] = mode

    setup = notebook["cells"][4]["source"]
    setup = setup.replace(
        "for path in [WORK_NOTEBOOK_ROOT, OUTPUT_DIR, WRITABLE_UNSLOTH_PARENT]:",
        "for path in [WORK_NOTEBOOK_ROOT, OUTPUT_DIR_A, OUTPUT_DIR_B, WRITABLE_UNSLOTH_PARENT]:",
    ).replace(
        'for path in Path("/kaggle/working").glob("worker_train_*"):',
        'for path in Path("/kaggle/working").glob("worker_train_dual_*"): ',
    )
    notebook["cells"][4]["source"] = setup

    preflight = notebook["cells"][5]["source"]
    preflight = preflight.replace(
        "required_files = [\n",
        "observed_code_hashes = {\n"
        "    name: __import__('hashlib').sha256(Path(WORK_CODE_DIR, name).read_bytes()).hexdigest()\n"
        "    for name in EXPECTED_CODE_HASHES\n"
        "}\n"
        "if observed_code_hashes != EXPECTED_CODE_HASHES:\n"
        "    raise RuntimeError({\"expected_code_hashes\": EXPECTED_CODE_HASHES, \"observed\": observed_code_hashes})\n"
        "\n"
        "required_files = [\n",
    ).replace(
        'assert "--eval-color-permutations" in starter_source\n',
        'assert "--eval-color-permutations" in starter_source\n'
        'assert "--train-color-permutations" in starter_source\n'
        'assert "--train-augmentation-seed" in starter_source\n'
        'assert "--eval-augmentation-seed" in starter_source\n'
        'assert "--trainer-seed" in starter_source\n'
        'assert "--sentinel-tag" in starter_source\n',
    ).replace(
        'print("arc2026 multi-token production preflight passed")\n',
        'print("arc2026 dual-LoRA production preflight passed")\n'
        'print("code_hashes =", observed_code_hashes)\n',
    )
    notebook["cells"][5]["source"] = preflight

    notebook["cells"][7]["source"] = """import json
import subprocess
import sys
import time


def branch_command(branch, output_dir, absolute_end_time):
    command = [
        sys.executable,
        "starter.py",
        "--test-path", TEST_PATH,
        "--model-path", MODEL_PATH,
        "--output-dir", output_dir,
        "--nprocs", str(BRANCH_NPROCS),
        "--cuda-device-offset", str(branch["cuda_offset"]),
        "--use-unsloth-multitoken-dfs",
        "--unsloth-multitoken-repeat-len", str(UNSLOTH_MULTITOKEN_REPEAT_LEN),
        "--dfs-prob-threshold", str(DFS_PROB_THRESHOLD),
        "--train-color-permutations", str(TRAIN_COLOR_PERMUTATIONS),
        "--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),
        "--train-augmentation-seed", str(branch["train_seed"]),
        "--eval-augmentation-seed", str(branch["eval_seed"]),
        "--trainer-seed", str(branch["trainer_seed"]),
        "--sentinel-tag", f"dual_{branch['name']}",
        "--end-time", str(absolute_end_time),
    ]
    if PROFILE_TIMINGS:
        command.append("--profile-timings")
    if SELECTED_KEYS is not None:
        command.extend(["--keys-json", json.dumps(SELECTED_KEYS)])
    return command


if RUN_INFERENCE:
    absolute_end_time = time.time() + END_TIME_HOURS * 3600
    dual_started_at = time.perf_counter()
    branch_specs = [
        (BRANCHES[0], OUTPUT_DIR_A),
        (BRANCHES[1], OUTPUT_DIR_B),
    ]
    processes = []
    for branch, output_dir in branch_specs:
        command = branch_command(branch, output_dir, absolute_end_time)
        print(f"starting branch {branch['name']}:", " ".join(command), flush=True)
        processes.append((branch["name"], subprocess.Popen(command, cwd=WORK_CODE_DIR, env=RUN_ENV)))

    pending = dict(processes)
    while pending:
        for name, process in list(pending.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del pending[name]
            print(f"branch {name} exit_code={return_code}", flush=True)
            if return_code != 0:
                for other in pending.values():
                    other.terminate()
                for other in pending.values():
                    other.wait()
                raise subprocess.CalledProcessError(return_code, f"dual-LoRA branch {name}")
        if pending:
            time.sleep(5)
    dual_wall_seconds = time.perf_counter() - dual_started_at
    print("dual_branch_wall_seconds =", dual_wall_seconds, flush=True)
else:
    dual_wall_seconds = 0.0
    print("save-version shortcut: full inference runs only during the competition rerun")
"""

    notebook["cells"][8]["source"] = """import json
import sys
from pathlib import Path

if WORK_CODE_DIR not in sys.path:
    sys.path.insert(0, WORK_CODE_DIR)

from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, score_kgmon


data = ArcDataset.from_file(TEST_PATH)
if EFFECTIVE_MODE == "validation" and SOLUTION_PATH is not None:
    data = data.load_replies(SOLUTION_PATH)
split_data = data.split_multi_replies()
expected_outputs = set(split_data.keys)

if not RUN_INFERENCE:
    submission = data.get_submission()
    Path(SUBMISSION_PATH).write_text(json.dumps(submission))
    print("save-version shortcut submission only; competition rerun performs dual-LoRA inference")
    print("submission_path =", SUBMISSION_PATH)
else:
    branch_decoders = {}
    branch_missing = {}
    for branch_name, output_dir in (("a", OUTPUT_DIR_A), ("b", OUTPUT_DIR_B)):
        decoder = ArcDecoder(split_data, n_guesses=2)
        path = Path(output_dir)
        if path.exists() and any(path.iterdir()):
            decoder.load_decoded_results(output_dir, run_name=f".branch_{branch_name}")
        branch_decoders[branch_name] = decoder
        branch_missing[branch_name] = sorted(expected_outputs - set(decoder.decoded_results))

    manifest = {
        "recipe": "dual_lora_48x2_q9_16x2",
        "expected_outputs": len(expected_outputs),
        "branch_output_counts": {
            name: len(decoder.decoded_results)
            for name, decoder in branch_decoders.items()
        },
        "branch_missing": branch_missing,
        "combined_missing": sorted(
            expected_outputs
            - set().union(
                *(set(decoder.decoded_results) for decoder in branch_decoders.values())
            )
        ),
        "train_steps_per_branch": 8 * TRAIN_COLOR_PERMUTATIONS,
        "inference_views_per_branch": 8 * EVAL_COLOR_PERMUTATIONS,
        "dual_branch_wall_seconds": dual_wall_seconds,
        "branches": BRANCHES,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    empty_branches = [
        name for name, decoder in branch_decoders.items()
        if not decoder.decoded_results
    ]
    if empty_branches:
        raise RuntimeError(
            "Dual-LoRA branch produced no candidate outputs; refusing a one-branch "
            f"submission for branches {empty_branches}. See {MANIFEST_PATH}"
        )

    combined = ArcDecoder(split_data, n_guesses=2)
    combined.load_decoded_results(OUTPUT_DIR_A, run_name=".branch_a")
    combined.load_decoded_results(OUTPUT_DIR_B, run_name=".branch_b")
    selected = combined.run_selection_algo(score_kgmon)
    submission = data.get_submission(selected)
    Path(SUBMISSION_PATH).write_text(json.dumps(submission))

    print("combined_output_keys =", len(combined.decoded_results))
    print("submission_path =", SUBMISSION_PATH)
    print("submission_tasks =", len(submission))

    if EFFECTIVE_MODE == "validation" and SOLUTION_PATH is not None:
        for branch_name, decoder in branch_decoders.items():
            print(f"branch_{branch_name}_validation")
            decoder.benchmark_selection_algos()
        combined.benchmark_selection_algos()
        print("validation_score =", data.validate_submission(submission))
"""

    clean(notebook)
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_notebook = UPLOAD_DIR / TARGET.name
    upload_notebook.write_bytes(TARGET.read_bytes())
    metadata = {
        "id": "yuvraj/arc26-dual-lora-48x2-q9-16x2-submit",
        "title": "[ARC26] Dual LoRA 48x2 q9 16x2 submit",
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
