"""Build the bounded end-to-end Canon-AC Kaggle smoke notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-canon-ac-end-to-end-smoke.ipynb"
SOURCE_FILES = [
    "arc_canon.py",
    "test_arc_canon.py",
    "arc_search.py",
    "arc_solver.py",
    "starter.py",
    "global_eval_curriculum.py",
    "train_global_eval_curriculum.py",
    "arc_loader.py",
    "arc_decoder.py",
    "arc_rescoring.py",
    "arc_opsd.py",
    "arc_repair_ttft.py",
    "arc_selected_augmentations.py",
    "arc_scheduled_sampling.py",
    "arc_sglang.py",
    "train_repair_adapter.py",
    "repair_mining.py",
    "repair_sft.py",
]
EXPECTED_SOURCE_HASHES = {
    name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    for name in SOURCE_FILES
}


def cell_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(source),
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(source),
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


cells = [
    markdown(
        """# ARC Canon-AC end-to-end smoke

This is a bounded mechanical gate, not a score experiment. It uses the real four-L4 global-training stack on 8 public-evaluation tasks x 4 views, trains zero-initialized residual Canon-AC alone for the first 30% of records and Canon plus a fresh rank-256 global LoRA for the rest, merges the ordinary adapters, then runs vanilla 128-step TTFT and one-token DFS on one validation puzzle. It verifies the released-paper boundary semantics, Canon weight movement, sidecar reload, branch-local incremental state, valid candidate production, and validation scoring. No speculative decoding is used in this first gate."""
    ),
    code(
        f"""from pathlib import Path

EXPECTED_SOURCE_HASHES = {EXPECTED_SOURCE_HASHES!r}
MODEL_PATH = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')
COMP_ROOT = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2')
CHALLENGES_PATH = COMP_ROOT / 'arc-agi_evaluation_challenges.json'
SOLUTIONS_PATH = COMP_ROOT / 'arc-agi_evaluation_solutions.json'
SMOKE_KEY = '0934a4d8'

WORK_ROOT = Path('/kaggle/working/arc26_canon_ac_smoke')
WORK_CODE_DIR = WORK_ROOT / 'ARC-AGI1/qwen_baseline'
GLOBAL_ROOT = Path('/kaggle/working/canon_global_smoke')
GLOBAL_ADAPTER = GLOBAL_ROOT / 'adapter'
MERGED_MODEL = Path('/kaggle/working/canon_merged_smoke')
INFERENCE_OUTPUT = Path('/kaggle/working/canon_smoke_candidates')

MODERN_UTILITY_ROOT = Path('/kaggle/usr/lib/notebooks/yuvraj/pip_install_unsloth_ddp_repair')
print('source files =', len(EXPECTED_SOURCE_HASHES))"""
    ),
    code(
        """import hashlib
import json
import os
import shutil
import subprocess
import sys

gpu_names = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
).strip().splitlines()
if len(gpu_names) != 4 or any('L4' not in name for name in gpu_names):
    raise RuntimeError(f'Expected four L4 GPUs, observed {gpu_names}')

code_candidates = [
    Path('/kaggle/input/datasets/yuvraj/arc2026'),
    Path('/kaggle/input/arc2026'),
]
CODE_DATASET_ROOT = next((path for path in code_candidates if path.is_dir()), None)
if CODE_DATASET_ROOT is None:
    raise FileNotFoundError(code_candidates)
source_root = CODE_DATASET_ROOT / 'ARC-AGI1/qwen_baseline'
observed = {
    name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
    for name in EXPECTED_SOURCE_HASHES
}
if observed != EXPECTED_SOURCE_HASHES:
    raise RuntimeError(f'arc2026 source mismatch: expected={EXPECTED_SOURCE_HASHES} observed={observed}')

repair_candidates = [
    Path('/kaggle/input/notebooks/yuvraj/arc26-repair-sft-continuation/repair_sft_continued/adapter'),
    Path('/kaggle/input/arc26-repair-sft-continuation/repair_sft_continued/adapter'),
]
REPAIR_ADAPTER = next((path for path in repair_candidates if path.is_dir()), None)
if REPAIR_ADAPTER is None:
    raise FileNotFoundError(repair_candidates)

fa2_candidates = sorted(set(
    path.parent.parent
    for root in (Path('/kaggle/input'), Path('/kaggle/usr/lib/notebooks'))
    if root.exists()
    for path in root.glob('**/flash_attn/__init__.py')
))
if len(fa2_candidates) != 1:
    raise RuntimeError(f'Expected one FlashAttention root, found {fa2_candidates}')
FA2_ROOT = fa2_candidates[0]

for path in (WORK_ROOT, GLOBAL_ROOT, MERGED_MODEL, INFERENCE_OUTPUT):
    shutil.rmtree(path, ignore_errors=True)
shutil.copytree(CODE_DATASET_ROOT, WORK_ROOT)
INFERENCE_OUTPUT.mkdir(parents=True)

environment = os.environ.copy()
environment.update({
    'UNSLOTH_DISABLE_STATISTICS': '1',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_ENABLE_HF_TRANSFER': '0',
    'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas',
    'OMP_NUM_THREADS': '3',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})
old_parts = [
    part for part in environment.get('PYTHONPATH', '').split(os.pathsep)
    if part and 'pip_install_unsloth_' not in part and 'flash_attention_' not in part
]
environment['PYTHONPATH'] = os.pathsep.join([
    str(FA2_ROOT), str(MODERN_UTILITY_ROOT), '/kaggle/working', str(WORK_CODE_DIR), *old_parts
])
print('GPUs =', gpu_names)
print('code =', CODE_DATASET_ROOT)
print('repair =', REPAIR_ADAPTER)
print('utility =', MODERN_UTILITY_ROOT)
print('FA2 =', FA2_ROOT)"""
    ),
    code(
        """# Cheap operator tests run before model loading.
subprocess.run(
    [sys.executable, '-m', 'unittest', 'test_arc_canon.py'],
    cwd=WORK_CODE_DIR,
    env=environment,
    check=True,
)"""
    ),
    code(
        """import time

train_command = [
    sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '4',
    str(WORK_CODE_DIR / 'train_global_eval_curriculum.py'),
    '--model-path', str(MODEL_PATH),
    '--repair-adapter-path', str(REPAIR_ADAPTER),
    '--challenges-path', str(CHALLENGES_PATH),
    '--output-dir', str(GLOBAL_ROOT),
    '--max-tasks', '8',
    '--views-per-task', '4',
    '--score-batch-size', '2',
    '--epochs', '1.0',
    '--learning-rate', '2.5e-5',
    '--lora-rank', '256',
    '--gradient-accumulation-steps', '1',
    '--expected-world-size', '4',
    '--canon-ac',
    '--canon-kernel-size', '4',
    '--canon-only-warmup-fraction', '0.30',
]
started = time.perf_counter()
print('running:', ' '.join(train_command))
subprocess.run(train_command, env=environment, check=True)
global_seconds = time.perf_counter() - started
manifest = json.loads((GLOBAL_ROOT / 'global_eval_curriculum_manifest.json').read_text())
print('global seconds =', round(global_seconds, 2))
print('stages =', json.dumps(manifest['train_stages'], indent=2))
print('canon l2 =', manifest['model']['canon_l2_before'], '->', manifest['model']['canon_l2_after'])
for required in ('adapter_model.safetensors', 'adapter_config.json', 'canon_ac.pt'):
    if not (GLOBAL_ADAPTER / required).is_file():
        raise FileNotFoundError(GLOBAL_ADAPTER / required)"""
    ),
    code(
        """import textwrap

merge_code = textwrap.dedent(r'''\
import json
import os
import shutil
from pathlib import Path

import unsloth
from peft import PeftModel
from unsloth import FastLanguageModel

model_path = os.environ['ARC_BASE_MODEL_PATH']
repair_path = Path(os.environ['ARC_REPAIR_ADAPTER_PATH'])
global_path = Path(os.environ['ARC_GLOBAL_ADAPTER_PATH'])
output_path = Path(os.environ['ARC_MERGED_MODEL_PATH'])
supported = {
    'base_model_name_or_path', 'bias', 'fan_in_fan_out', 'inference_mode',
    'init_lora_weights', 'layers_pattern', 'layers_to_transform', 'loftq_config',
    'lora_alpha', 'lora_dropout', 'megatron_config', 'megatron_core',
    'modules_to_save', 'peft_type', 'r', 'rank_pattern', 'revision',
    'target_modules', 'task_type', 'use_dora', 'use_rslora',
}

def compatible_adapter(source, name):
    destination = Path('/kaggle/working') / name
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    raw = json.loads((source / 'adapter_config.json').read_text())
    config = {key: value for key, value in raw.items() if key in supported}
    config['base_model_name_or_path'] = model_path
    (destination / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    os.symlink(source / 'adapter_model.safetensors', destination / 'adapter_model.safetensors')
    return destination

repair_compat = compatible_adapter(repair_path, 'canon_repair_compat')
global_compat = compatible_adapter(global_path, 'canon_global_compat')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path, full_finetuning=False, load_in_4bit=False,
    local_files_only=True, use_gradient_checkpointing=False, max_seq_length=8192,
)
existing = getattr(tokenizer, 'additional_special_tokens', None)
if existing is None:
    existing = (getattr(tokenizer, 'special_tokens_map', {}) or {}).get('additional_special_tokens', [])
existing = list(existing)
if '<REPAIR>' not in existing:
    existing.append('<REPAIR>')
    if tokenizer.add_special_tokens({'additional_special_tokens': existing}) != 1:
        raise RuntimeError('Failed to add exactly one repair token')
try:
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
except TypeError:
    model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, str(repair_compat), is_trainable=False, local_files_only=True)
model = model.merge_and_unload(safe_merge=True)
model = PeftModel.from_pretrained(model, str(global_compat), is_trainable=False, local_files_only=True)
model = model.merge_and_unload(safe_merge=True)
output_path.mkdir(parents=True)
model.save_pretrained(output_path, safe_serialization=True, max_shard_size='5GB')
tokenizer.save_pretrained(output_path)
print('saved merged ordinary weights:', output_path)
''')

merge_env = environment.copy()
merge_env.update({
    'ARC_BASE_MODEL_PATH': str(MODEL_PATH),
    'ARC_REPAIR_ADAPTER_PATH': str(REPAIR_ADAPTER),
    'ARC_GLOBAL_ADAPTER_PATH': str(GLOBAL_ADAPTER),
    'ARC_MERGED_MODEL_PATH': str(MERGED_MODEL),
})
subprocess.run([sys.executable, '-c', merge_code], env=merge_env, check=True)"""
    ),
    code(
        """import time

inference_command = [
    sys.executable, str(WORK_CODE_DIR / 'starter.py'),
    '--test-path', str(CHALLENGES_PATH),
    '--model-path', str(MERGED_MODEL),
    '--output-dir', str(INFERENCE_OUTPUT),
    '--keys-json', json.dumps([SMOKE_KEY]),
    '--nprocs', '1',
    '--dfs-prob-threshold', '0.2',
    '--eval-color-permutations', '2',
    '--ttft-method', 'full_sft',
    '--canon-ac-state', str(GLOBAL_ADAPTER / 'canon_ac.pt'),
    '--end-time', str(time.time() + 45 * 60),
    '--profile-timings',
]
started = time.perf_counter()
print('running:', ' '.join(inference_command))
subprocess.run(inference_command, cwd=WORK_CODE_DIR, env=environment, check=True)
inference_seconds = time.perf_counter() - started
print('inference seconds =', round(inference_seconds, 2))
print('candidate files =', len([path for path in INFERENCE_OUTPUT.iterdir() if path.is_file()]))"""
    ),
    code(
        """import bz2
import pickle

import numpy as np

sys.path.insert(0, str(WORK_CODE_DIR))
from arc_decoder import ArcDecoder, score_kgmon
from arc_loader import ArcDataset

data = ArcDataset.from_file(CHALLENGES_PATH, keys=[SMOKE_KEY]).load_replies(SOLUTIONS_PATH)
split_data = data.split_multi_replies()
decoder = ArcDecoder(split_data, n_guesses=2)
candidate_records = 0
for path in sorted(INFERENCE_OUTPUT.iterdir()):
    if not path.is_file():
        continue
    with bz2.BZ2File(path, 'rb') as handle:
        rows = pickle.load(handle)
    base_key = path.name.split('.')[0]
    decoder.decoded_results.setdefault(base_key, {})
    for index, row in enumerate(rows):
        decoder.decoded_results[base_key][f'{path.name}.out{index}'] = row
        candidate_records += 1

selected = decoder.run_selection_algo(score_kgmon)
submission = data.get_submission(selected)
score = data.validate_submission(submission)
summary = {
    'key': SMOKE_KEY,
    'candidate_files': len([path for path in INFERENCE_OUTPUT.iterdir() if path.is_file()]),
    'candidate_records': candidate_records,
    'decoded_outputs': len(decoder.decoded_results),
    'selected_score': score,
    'global_seconds': global_seconds,
    'inference_seconds': inference_seconds,
    'canon_l2_before': manifest['model']['canon_l2_before'],
    'canon_l2_after': manifest['model']['canon_l2_after'],
}
(Path('/kaggle/working') / 'canon_ac_smoke_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
print(json.dumps(summary, indent=2))"""
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
OUTPUT.write_text(json.dumps(notebook, indent=1))
print(OUTPUT)
