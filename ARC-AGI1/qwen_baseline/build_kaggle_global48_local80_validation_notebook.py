"""Build the clean global-then-local 48-task validation notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "arc26-vanilla-v2-q9-24-submit-competition.ipynb"
OUTPUT = HERE.parent / "arc26-global48-local80-q9-24-validation.ipynb"

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


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = """# ARC26 clean global-LoRA + local-80 validation

Controlled 48-task comparison against the 31.94 Vanilla V2 stack. A separately
trained rank-256 global LoRA is merged into the untouched published model, then
each puzzle receives ordinary 80-step local TTFT. Candidate generation and
selection are unchanged: q_len=9, threshold 0.2, eight geometries by three
colour/order variants, and `score_kgmon`. No Repair or scheduled sampling is
used.
"""
    notebook["cells"][2]["source"] = f'''MODE = "validation"

CODE_DATASET_ROOT = "/kaggle/input/datasets/yuvraj/arc2026"
BASE_MODEL_PATH = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
GLOBAL_RUN_ROOT = "/kaggle/input/notebooks/yuvraj/arc26-global48-vanilla-lora/global48_vanilla_lora"
GLOBAL_ADAPTER = GLOBAL_RUN_ROOT + "/adapter"
MODEL_PATH = "/kaggle/working/global48_vanilla_merged_model"
COMP_ROOT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"

VALIDATION_KEYS = {VALIDATION_KEYS!r}
NPROCS = 4
LOCAL_TTFT_STEPS = 80
DFS_PROB_THRESHOLD = 0.2
UNSLOTH_MULTITOKEN_REPEAT_LEN = 9
EVAL_COLOR_PERMUTATIONS = 3
SELECTION_ALGORITHM = "score_kgmon"
PROFILE_TIMINGS = True

VALIDATION_END_TIME_HOURS = 4.0
SUBMIT_COMPETITION_END_TIME_HOURS = 11 + 50 / 60
RESET_RUN_ARTIFACTS = True

WORK_NOTEBOOK_ROOT = "/kaggle/working/arc2026_run_global48_local80_q9_24"
WORK_CODE_DIR = "/kaggle/working/arc2026_run_global48_local80_q9_24/ARC-AGI1/qwen_baseline"
WRITABLE_UNSLOTH_PARENT = "/kaggle/working/global48_local80_q9_24_stack"
'''

    setup = notebook["cells"][4]["source"]
    setup = setup.replace(
        'assert Path(MODEL_PATH).exists(), f"Missing model path: {MODEL_PATH}"',
        'assert Path(BASE_MODEL_PATH).exists(), f"Missing base model path: {BASE_MODEL_PATH}"\n'
        'assert Path(GLOBAL_ADAPTER).exists(), f"Missing global adapter: {GLOBAL_ADAPTER}"',
    )
    setup = setup.replace(
        'for path in [WORK_NOTEBOOK_ROOT, OUTPUT_DIR, WRITABLE_UNSLOTH_PARENT]:',
        'for path in [WORK_NOTEBOOK_ROOT, OUTPUT_DIR, WRITABLE_UNSLOTH_PARENT, MODEL_PATH, "/kaggle/working/global48_adapter_compat"]:',
    )
    notebook["cells"][4]["source"] = setup

    merge_cell = code(r'''import subprocess
import textwrap
import time

merge_code = textwrap.dedent(r"""
import json
import os
import shutil
from pathlib import Path

import unsloth
from peft import PeftModel
from unsloth import FastLanguageModel

base_path = os.environ["ARC_BASE_MODEL_PATH"]
adapter_path = Path(os.environ["ARC_GLOBAL_ADAPTER_PATH"])
compat_path = Path("/kaggle/working/global48_adapter_compat")
output_path = Path(os.environ["ARC_MERGED_MODEL_PATH"])

# Modern PEFT writes extra config fields which the pinned Vanilla V2 PEFT does
# not recognize. The weights and all training-relevant fields are unchanged.
supported = {
    "base_model_name_or_path", "bias", "fan_in_fan_out", "inference_mode",
    "init_lora_weights", "layers_pattern", "layers_to_transform", "loftq_config",
    "lora_alpha", "lora_dropout", "megatron_config", "megatron_core",
    "modules_to_save", "peft_type", "r", "rank_pattern", "revision",
    "target_modules", "task_type", "use_dora", "use_rslora",
}
raw = json.loads((adapter_path / "adapter_config.json").read_text())
config = {key: value for key, value in raw.items() if key in supported}
config["base_model_name_or_path"] = base_path
compat_path.mkdir(parents=True)
(compat_path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
os.symlink(adapter_path / "adapter_model.safetensors", compat_path / "adapter_model.safetensors")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_path,
    full_finetuning=False,
    load_in_4bit=False,
    local_files_only=True,
    use_gradient_checkpointing=False,
    max_seq_length=8192,
)
model = PeftModel.from_pretrained(
    model, str(compat_path), is_trainable=False, local_files_only=True
)
model = model.merge_and_unload(safe_merge=True)
output_path.mkdir(parents=True)
model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(output_path)
print("saved global-merged model:", output_path)
""")

merge_env = os.environ.copy()
merge_env.update({
    "ARC_BASE_MODEL_PATH": BASE_MODEL_PATH,
    "ARC_GLOBAL_ADAPTER_PATH": GLOBAL_ADAPTER,
    "ARC_MERGED_MODEL_PATH": MODEL_PATH,
})
merge_started = time.perf_counter()
subprocess.run([sys.executable, "-c", merge_code], env=merge_env, check=True)
MERGE_WALL_SECONDS = time.perf_counter() - merge_started
print("merge_wall_s =", round(MERGE_WALL_SECONDS, 3))
''')
    # Merge under the untouched, pinned Vanilla environment before making the
    # writable q9-patched Unsloth copy.
    notebook["cells"].insert(6, merge_cell)

    command_cell = notebook["cells"][8]
    command_cell["source"] = command_cell["source"].replace(
        '"--nprocs", str(NPROCS),',
        '"--nprocs", str(NPROCS),\n        "--full-sft-total-steps", str(LOCAL_TTFT_STEPS),',
    )
    command_cell["source"] = command_cell["source"].replace(
        'subprocess.run(cmd, cwd=WORK_CODE_DIR, env=RUN_ENV, check=True)',
        'inference_started = time.perf_counter()\n'
        '    subprocess.run(cmd, cwd=WORK_CODE_DIR, env=RUN_ENV, check=True)\n'
        '    INFERENCE_WALL_SECONDS = time.perf_counter() - inference_started\n'
        '    print("inference_wall_s =", round(INFERENCE_WALL_SECONDS, 3))\n'
        '    global_runtime = json.loads(Path(GLOBAL_RUN_ROOT, "global_notebook_runtime.json").read_text())\n'
        '    print("merge_plus_local_wall_s =", round(MERGE_WALL_SECONDS + INFERENCE_WALL_SECONDS, 3))\n'
        '    print("global_plus_merge_plus_local_wall_s =", round(\n'
        '        global_runtime["global_notebook_wall_s"] + MERGE_WALL_SECONDS + INFERENCE_WALL_SECONDS,\n'
        '        3,\n'
        '    ))',
    )

    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
