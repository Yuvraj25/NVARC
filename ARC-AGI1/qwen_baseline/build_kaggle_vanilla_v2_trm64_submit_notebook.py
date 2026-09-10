import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_NOTEBOOK = REPO_ROOT / "ARC-AGI1/arc26-vanilla-v2-q9-24-submit-competition.ipynb"
TRM_NOTEBOOK = Path(
    "/Users/banna/kaggle/temp/kaggle_trm_arc26_validation48/"
    "arc26-trm-v18-recipe-validation48.ipynb"
)
OUTPUT_NOTEBOOK = REPO_ROOT / "ARC-AGI1/arc26-vanilla-v2-q9-24-trm64-submit-competition.ipynb"
METADATA_TEMPLATE = Path(
    "/Users/banna/kaggle/temp/kaggle_vanilla_v2_q9_24_submit/kernel-metadata.json"
)
OUTPUT_METADATA = Path(
    "/Users/banna/kaggle/temp/kaggle_vanilla_v2_trm64_submit/kernel-metadata.json"
)
EXPECTED_SOURCE_CELLS_SHA256 = "80753fa51e47f214d8272a2ec6293c6f1199b41d178ff654adc967367f756991"


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def source_cells_sha256(notebook):
    payload = json.dumps(
        [(cell["cell_type"], cell["source"]) for cell in notebook["cells"]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def conditional_source(source):
    return "if RUN_INFERENCE:\n" + "\n".join(
        f"    {line}" if line else "" for line in source.splitlines()
    ) + "\n"


def main():
    notebook = json.loads(QWEN_NOTEBOOK.read_text())
    trm_notebook = json.loads(TRM_NOTEBOOK.read_text())
    observed_source_hash = source_cells_sha256(notebook)
    if observed_source_hash != EXPECTED_SOURCE_CELLS_SHA256:
        raise RuntimeError(
            f"Vanilla V2 Version 4 source drift: {observed_source_hash}"
        )

    notebook["cells"][0]["source"] = """# ARC26 Vanilla V2 q9/24 plus bounded TRM-64

The exact 32.08 Vanilla V2 production path, with inference batching changed from
4 to 6. After Qwen finishes, the released TRM v18 recipe uses 64 augmentations
and trains until the shared 11h25 deadline, then evaluates every task. Final
attempts are Qwen rank 1 plus TRM attempt 1, falling back to Qwen rank 2 only
when the TRM grid is invalid or duplicates Qwen rank 1.
"""
    notebook["cells"].insert(1, code_cell("""import time

NOTEBOOK_START_TIME = time.time()
TRM_AUGMENTATIONS = 64
TRM_TRAIN_STOP_HOURS = 11 + 25 / 60
FINAL_TARGET_HOURS = 11 + 50 / 60
print("notebook_start_time =", NOTEBOOK_START_TIME)
"""))

    config_cell = notebook["cells"][3]["source"]
    config_cell = config_cell.replace(
        'EVAL_COLOR_PERMUTATIONS = 3\n',
        'EVAL_COLOR_PERMUTATIONS = 3\nEVAL_BATCH_SIZE = 6\n',
    )
    notebook["cells"][3]["source"] = config_cell

    routing_cell = notebook["cells"][4]["source"]
    routing_cell = routing_cell.replace(
        'SUBMISSION_PATH = "/kaggle/working/submission.json"',
        'SUBMISSION_PATH = "/kaggle/working/qwen_submission.json"',
    )
    notebook["cells"][4]["source"] = routing_cell

    launch_cell = notebook["cells"][8]["source"]
    launch_cell = launch_cell.replace(
        '"--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),\n',
        '"--eval-color-permutations", str(EVAL_COLOR_PERMUTATIONS),\n'
        '        "--eval-batch-size", str(EVAL_BATCH_SIZE),\n',
    )
    launch_cell = launch_cell.replace(
        '"--end-time", str(time.time() + END_TIME_HOURS * 3600),',
        '"--end-time", str(NOTEBOOK_START_TIME + TRM_TRAIN_STOP_HOURS * 3600),',
    )
    notebook["cells"][8]["source"] = launch_cell

    notebook["cells"].append(code_cell("""import shutil
from pathlib import Path

QWEN_SUBMISSION_PATH = Path("/kaggle/working/qwen_submission.json")
assert QWEN_SUBMISSION_PATH.is_file(), QWEN_SUBMISSION_PATH
for path in [WORK_NOTEBOOK_ROOT, OUTPUT_DIR, WRITABLE_UNSLOTH_PARENT]:
    shutil.rmtree(path, ignore_errors=True)
print("qwen phase complete; elapsed_hours =", (time.time() - NOTEBOOK_START_TIME) / 3600)
"""))

    wheel_source = trm_notebook["cells"][1]["source"]
    notebook["cells"].append(code_cell(conditional_source(wheel_source)))
    notebook["cells"].append(code_cell(conditional_source("""import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRM_WORK = Path("/kaggle/working/trm64_phase")
shutil.rmtree(TRM_WORK, ignore_errors=True)
TRM_WORK.mkdir(parents=True)
os.chdir(TRM_WORK)

TRM_ROOT = Path("/kaggle/input/trm-code/20251211_trm_handover/TinyRecursiveModels")
assert TRM_ROOT.is_dir(), TRM_ROOT
observed_trm_hash = hashlib.sha256(
    (TRM_ROOT / "models/recursive_reasoning/trm.py").read_bytes()
).hexdigest()
assert observed_trm_hash == "3454818b64a1abb45c062782d380aa4bb560e2cb35db9cc309cf367ca3ac262c"
for name in ["models", "puzzle_dataset.py", "dataset", "utils", "evaluators", "assets", "config", "kaggle"]:
    os.symlink(TRM_ROOT / name, name)

data_dir = Path("data1")
data_dir.mkdir()
shutil.copy2(TEST_PATH, data_dir / "arc-agi_test_challenges.json")
subprocess.run([
    sys.executable, "-m", "dataset.build_arc_dataset",
    "--input-file-prefix", "./data1/arc-agi",
    "--output-dir", "data1/arc2test-aug-64",
    "--subsets", "test",
    "--test-set-name", "test",
    "--num-aug", str(TRM_AUGMENTATIONS),
], check=True)
print("TRM dataset ready; elapsed_hours =", (time.time() - NOTEBOOK_START_TIME) / 3600)
""")))

    eval_source = trm_notebook["cells"][4]["source"]
    assert eval_source.startswith("%%writefile eval-arc.py\n")
    eval_source = eval_source.removeprefix("%%writefile eval-arc.py\n")
    eval_source = eval_source.replace("import copy\n", "import copy\nimport time\n", 1)
    marker = """            if config.ema:
                ema_helper.update(train_state.model)
"""
    replacement = marker + """
            deadline = float(os.environ.get("TRM_TRAIN_DEADLINE", "inf"))
            stop = torch.tensor([int(time.time() >= deadline)], device="cuda")
            if dist.is_initialized():
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                if RANK == 0:
                    print(f"TRM training deadline reached at step {train_state.step}")
                break
"""
    assert marker in eval_source
    eval_source = eval_source.replace(marker, replacement, 1)
    notebook["cells"].append(code_cell(conditional_source(
        "from pathlib import Path\n"
        f"Path('eval-arc.py').write_text({eval_source!r})\n"
        "print('Wrote eval-arc.py')\n"
    )))

    notebook["cells"].append(code_cell(conditional_source("""import os
import subprocess
import time
from pathlib import Path

trm_deadline = NOTEBOOK_START_TIME + TRM_TRAIN_STOP_HOURS * 3600
env = os.environ.copy()
env["TRM_TRAIN_DEADLINE"] = str(trm_deadline)
cmd = [
    "torchrun", "--standalone", "--nnodes=1", "--nproc-per-node", "4",
    "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0", "eval-arc.py",
    "arch=trm", "data_paths=[./data1/arc2test-aug-64]",
    "arch.L_layers=2", "arch.H_cycles=4", "arch.L_cycles=4",
    "arch.halt_max_steps=10", "freeze_weights=False",
    "+load_checkpoint=/kaggle/input/arc-prize-trm-031/step_220708",
    "+checkpoint_path=./eval_checkpoint", "eval_interval=4000", "epochs=4000",
    "global_batch_size=128", "ema=True", "lr_warmup_steps=200", "lr=0.0001",
]
print("TRM seconds available for setup/training =", max(0, trm_deadline - time.time()))
result = subprocess.run(cmd, env=env, check=False)
print("TRM returncode =", result.returncode)
""")))

    notebook["cells"].append(code_cell("""import json
import shutil
from pathlib import Path

def valid_grid(grid):
    return (
        isinstance(grid, list) and 1 <= len(grid) <= 30
        and all(isinstance(row, list) and len(row) == len(grid[0]) for row in grid)
        and 1 <= len(grid[0]) <= 30
        and all(isinstance(cell, int) and 0 <= cell <= 9 for row in grid for cell in row)
    )

FINAL_SUBMISSION_PATH = Path("/kaggle/working/submission.json")
if not RUN_INFERENCE:
    shutil.copy2(QWEN_SUBMISSION_PATH, FINAL_SUBMISSION_PATH)
    print("Shortcut run: copied Qwen placeholder submission")
else:
    qwen_submission = json.loads(QWEN_SUBMISSION_PATH.read_text())
    submission_files = sorted(Path("eval_checkpoint").glob("evaluator_*/submission.json"))
    trm_submission = json.loads(submission_files[-1].read_text()) if submission_files else {}

    trm_used = 0
    fallback_used = 0
    for task, outputs in qwen_submission.items():
        trm_outputs = trm_submission.get(task, [])
        for output_index, qwen_output in enumerate(outputs):
            qwen_first = qwen_output["attempt_1"]
            qwen_second = qwen_output["attempt_2"]
            trm_first = None
            if output_index < len(trm_outputs):
                trm_first = trm_outputs[output_index].get("attempt_1")
            if valid_grid(trm_first) and trm_first != qwen_first:
                qwen_output["attempt_2"] = trm_first
                trm_used += 1
            else:
                qwen_output["attempt_2"] = qwen_second
                fallback_used += 1

    FINAL_SUBMISSION_PATH.write_text(json.dumps(qwen_submission))
    assert set(qwen_submission) == set(json.loads(Path(TEST_PATH).read_text()))
    print("TRM attempt-1 slots used =", trm_used)
    print("Qwen rank-2 fallbacks =", fallback_used)
print("final elapsed_hours =", (time.time() - NOTEBOOK_START_TIME) / 3600)
print("submission_path =", FINAL_SUBMISSION_PATH)
"""))

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook_payload = json.dumps(notebook, indent=1) + "\n"
    OUTPUT_NOTEBOOK.write_text(notebook_payload)

    metadata = json.loads(METADATA_TEMPLATE.read_text())
    metadata.update({
        "id": "yuvraj/arc26-vanilla-v2-q9-24-trm64-submit",
        "title": "ARC26 Vanilla V2 b6 q9 24 plus TRM64",
        "code_file": OUTPUT_NOTEBOOK.name,
        "dataset_sources": [
            "yuvraj/arc2026",
            "chanhainguyen/trm-code",
            "cpmpml/arc-prize-trm-031",
        ],
    })
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT_METADATA.parent / OUTPUT_NOTEBOOK.name).write_text(notebook_payload)
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(OUTPUT_NOTEBOOK)
    print(OUTPUT_METADATA)


if __name__ == "__main__":
    main()
