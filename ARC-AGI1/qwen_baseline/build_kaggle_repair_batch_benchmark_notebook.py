"""Build the self-contained Kaggle batched repair-mining benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-repair-batch-benchmark.ipynb"


def cell_id(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(source),
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(source),
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


module_source = (HERE / "repair_mining.py").read_text()

expected = {
    "913fb3ed_d94c3b52": (False, 2, 21, "049f94bb849b0d725e67ac387b37b900c30d971e9bde0ecd21cea21b51885f25"),
    "d406998b_62b74c02": (True, 0, None, None),
    "770cc55f_794b24be": (False, 1, 101, "f855c8c3647260799ecdbfcb9b594ad483a1a4fc9c68b5c5dc44ff6fddf03bd0"),
    "212895b5_f1cefba8": (False, 1, 104, "8ff98ae47bdf658c9375717258b28df50d82756b872e86e2dd9c2336a11ef4e9"),
    "d90796e8_05a7bcf2": (False, 2, 237, "be3b62901279585fcb77768a0b2d31563e0e6b50d0f4f75ca204cb0ec4c33e1e"),
    "0b148d64_a85d4709": (False, 2, 0, "deb02927c18a816edc90e5f09461944e70e5a511647cfc8cee046fd1bde13754"),
    "6ea4a07e_f8ff0b80": (False, 2, 2, "c5c456a4e7bce0f86e00333ce178481178c5fccb51f66e0495637e5a6ea04152"),
    "5a5a2103_505fff84": (False, 1, 9, "4f8aa950c7e264a0e8aff4e0100bf04c0e848adc63e47fe17eadab999b10ab18"),
    "22eb0ac0_1990f7a8": (False, 6, 2, "a2d46d410927234ba7c9f67ce8c012089bd1c4983b41eed246a6d3f4ff9ab05b"),
    "a04b2602_d4c90558": (False, 3, 16, "4586dd27b5b9b18a4746e732bc7f35faa00ba45fd7c4f1b47ba61e1f64de6ff3"),
    "0a2355a6_0f63c0b9": (False, 4, 0, "2f63cf3b89923cb2ee53328a9ab1934dcd8c40cff75229ab5c0f28d7ef9c9471"),
    "f3cdc58f_80af3007": (False, 13, 1, "ed4ee37ee02446aedb25363dd0a049ee40a6a8b790b6a8794e79256c410b8054"),
    "aee291af_e7b06bea": (True, 0, None, None),
    "c0f76784_3f7978a0": (True, 0, None, None),
    "3ac3eb23_d13f3404": (False, 5, 1, "b071e711c5ec501095674a9c05e2da93b0be3010cce80f0da89ace8291a3179b"),
    "150deff5_82819916": (False, 1, 547, "b576deddb332a0fc6fbf28a46c82ef9bd952f8e535c790152c61411ed295c2aa"),
}

cells = [
    markdown_cell(
        """# ARC repair-mining batch benchmark

This benchmark reruns the 16 clean `nvarc_training` probes from the successful batch-size-1 smoke. It uses length-bucketed teacher-forced batches of 8 and rollout batches of 4, verifies exact parity against the saved v3 predictions, and constructs failure, no-op, and ordinary-replay records. It performs no training and installs no packages."""
    ),
    code_cell(
        f"""RUN_EXPERIMENT = True
NUM_PROBES = 16
TF_BATCH_SIZE = 8
ROLLOUT_BATCH_SIZE = 4
SEED = 20260811
MAX_SEQ_LENGTH = 8192
BATCH1_ROLLOUT_SECONDS = 198.78762031300005

MODEL_PATH = '/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
COMP_ROOT = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
OUTPUT_JSON = '/kaggle/working/repair_batch_benchmark.json'

EXPECTED = {expected!r}
print('TF batch =', TF_BATCH_SIZE, 'rollout batch =', ROLLOUT_BATCH_SIZE)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import os, sys
    os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
    os.environ['OMP_NUM_THREADS'] = '12'

    import unsloth
    import numpy as np
    import torch
    import transformers
    print('python =', sys.version)
    print('numpy =', np.__version__, np.__file__)
    print('torch =', torch.__version__, torch.__file__)
    print('transformers =', transformers.__version__, transformers.__file__)
    print('unsloth =', unsloth.__file__)
    print('cuda devices =', torch.cuda.device_count())"""
    ),
    code_cell(module_source),
    code_cell(
        """if RUN_EXPERIMENT:
    import json
    from pathlib import Path

    input_root = Path('/kaggle/input')
    validation_ids = set(json.loads((Path(COMP_ROOT) / 'arc-agi_evaluation_challenges.json').read_text()))
    training_root = discover_subset_root(input_root, 'nvarc_training')
    full_root = discover_subset_root(input_root, 'nvarc_full')
    full_anchors = {path.name for path in full_root.iterdir() if path.is_dir()}
    assert full_anchors == validation_ids

    paths = deterministic_sample_paths(
        training_root,
        count=NUM_PROBES,
        seed=SEED,
        excluded_anchor_ids=validation_ids,
    )
    probes = [load_probe_from_path(path, subset='nvarc_training', seed=SEED) for path in paths]
    assert [probe.puzzle_id for probe in probes] == list(EXPECTED)
    prompts = [format_prompt(probe) for probe in probes]
    replies = [format_reply(probe.gold_output) for probe in probes]
    print('unique clean probes =', len(probes))"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import time
    from unsloth import FastLanguageModel

    started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model = FastLanguageModel.for_inference(model)
    stabilize_inference_state(model)
    assert len(tokenizer) == 16
    print('model load seconds =', round(time.perf_counter() - started, 2))"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import gc, hashlib, json, time
    from pathlib import Path

    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    tf_metrics = [None] * len(probes)
    tf_started = time.perf_counter()
    tf_batches = length_bucket_batches(prompts, batch_size=TF_BATCH_SIZE, key=lambda prompt: len(tokenizer.encode(prompt)))
    for indices in tf_batches:
        batch_metrics = teacher_forced_metrics_batch(
            model,
            tokenizer,
            [prompts[index] for index in indices],
            [replies[index] for index in indices],
        )
        for index, metrics in zip(indices, batch_metrics):
            tf_metrics[index] = metrics
    tf_seconds = time.perf_counter() - tf_started

    for probe, metrics in zip(probes, tf_metrics):
        expected_exact, expected_wrong, expected_first, _ = EXPECTED[probe.puzzle_id]
        assert metrics['restricted_greedy_exact'] == expected_exact, (probe.puzzle_id, metrics)
        assert metrics['wrong_argmax_tokens'] == expected_wrong, (probe.puzzle_id, metrics)
        assert metrics['first_wrong_token'] == expected_first, (probe.puzzle_id, metrics)

    failure_indices = [index for index, metrics in enumerate(tf_metrics) if not metrics['restricted_greedy_exact']]
    rollout_prompts = [prompts[index] for index in failure_indices]
    rollout_results = [None] * len(failure_indices)
    rollout_batches = length_bucket_batches(
        rollout_prompts,
        batch_size=ROLLOUT_BATCH_SIZE,
        key=lambda prompt: len(tokenizer.encode(prompt)),
    )
    rollout_started = time.perf_counter()
    for batch_number, local_indices in enumerate(rollout_batches, 1):
        batch_prompts = [rollout_prompts[index] for index in local_indices]
        maximum_prompt = max(len(tokenizer.encode(prompt)) for prompt in batch_prompts)
        batch_rollouts = restricted_greedy_rollout_batch(
            model,
            tokenizer,
            batch_prompts,
            max_new_tokens=min(930, MAX_SEQ_LENGTH - maximum_prompt),
        )
        for local_index, token_ids in zip(local_indices, batch_rollouts):
            rollout_results[local_index] = token_ids
        print('rollout batch', batch_number, '/', len(rollout_batches), 'size', len(local_indices))
        gc.collect()
        torch.cuda.empty_cache()
    rollout_seconds = time.perf_counter() - rollout_started

    records = []
    failure_training_records = []
    rollout_mismatches = []
    invalid_rollouts = []
    for local_index, probe_index in enumerate(failure_indices):
        probe = probes[probe_index]
        token_ids = rollout_results[local_index]
        prediction, invalid_reason = parse_rollout_grid(tokenizer, token_ids)
        if prediction is None:
            invalid_rollouts.append({'puzzle_id': probe.puzzle_id, 'reason': invalid_reason})
            records.append({
                'puzzle_id': probe.puzzle_id,
                **tf_metrics[probe_index],
                'rollout_tokens': len(token_ids),
                'rollout_invalid_reason': invalid_reason,
                'prediction_sha256': None,
                'expected_prediction_sha256': EXPECTED[probe.puzzle_id][3],
                'batch1_parity': False,
            })
            continue
        prediction_hash = hashlib.sha256(json.dumps(prediction, separators=(',', ':')).encode()).hexdigest()
        expected_hash = EXPECTED[probe.puzzle_id][3]
        matches_batch1 = prediction_hash == expected_hash
        if not matches_batch1:
            rollout_mismatches.append({
                'puzzle_id': probe.puzzle_id,
                'prediction_sha256': prediction_hash,
                'expected_prediction_sha256': expected_hash,
            })
        training_record = build_repair_training_record(probe, prediction)
        failure_training_records.append(training_record)
        records.append({
            'puzzle_id': probe.puzzle_id,
            **tf_metrics[probe_index],
            'rollout_tokens': len(token_ids),
            'prediction_sha256': prediction_hash,
            'expected_prediction_sha256': expected_hash,
            'batch1_parity': matches_batch1,
            **error_mask_diagnostics(prediction, probe.gold_output),
        })

    exact_indices = [index for index, metrics in enumerate(tf_metrics) if metrics['restricted_greedy_exact']]
    noop_records = [build_repair_training_record(probes[index], probes[index].gold_output) for index in exact_indices]
    solve_replay_records = [build_solve_replay_record(probe) for probe in probes]
    result = {
        'config': {
            'num_probes': NUM_PROBES,
            'tf_batch_size': TF_BATCH_SIZE,
            'rollout_batch_size': ROLLOUT_BATCH_SIZE,
            'seed': SEED,
        },
        'parity': {
            'teacher_forced': True,
            'rollouts': not rollout_mismatches and not invalid_rollouts,
            'compared_failures': len(failure_indices),
            'rollout_mismatches': rollout_mismatches,
            'invalid_rollouts': invalid_rollouts,
        },
        'counts': {
            'failure_repairs': len(failure_training_records),
            'repair_noops': len(noop_records),
            'solve_replay_available_from_source': len(solve_replay_records),
        },
        'timings': {
            'teacher_forced_batched_s': tf_seconds,
            'rollout_batched_s': rollout_seconds,
            'rollout_batch1_reference_s': BATCH1_ROLLOUT_SECONDS,
            'rollout_speedup_vs_batch1': BATCH1_ROLLOUT_SECONDS / rollout_seconds,
        },
        'records': records,
        'record_schema_examples': {
            'repair_failure': failure_training_records[0],
            'repair_noop': noop_records[0],
            'solve_replay': solve_replay_records[0],
        },
    }
    Path(OUTPUT_JSON).write_text(json.dumps(result, indent=2))
    print('BATCH BENCHMARK SUMMARY')
    print(json.dumps({key: value for key, value in result.items() if key not in {'records', 'record_schema_examples'}}, indent=2))"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(OUTPUT)
