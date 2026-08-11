"""Build the self-contained Kaggle repair-mining GPU smoke notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-repair-mining-smoke.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().splitlines()],
    }


module_source = (HERE / "repair_mining.py").read_text()

cells = [
    markdown_cell(
        """# ARC-AGI-2 error-mask repair mining smoke

This notebook spends no time training. It verifies the clean raw `nvarc_training` pool, unique-puzzle A,B→C construction, teacher-forced restricted-argmax screen, and actual restricted-greedy failure generation in the pinned NVARC model environment. Only teacher-forced failures are autoregressively generated. `nvarc_full` is deliberately excluded because its 120 top-level anchors are exactly the 120 validation IDs."""
    ),
    code_cell(
        """RUN_EXPERIMENT = True
NUM_PROBES = 16
SEED = 20260811
MAX_SEQ_LENGTH = 8192

MODEL_PATH = '/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
COMP_ROOT = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
OUTPUT_JSON = '/kaggle/working/repair_mining_smoke.json'

print('RUN_EXPERIMENT =', RUN_EXPERIMENT)
print('unique nvarc_training probes =', NUM_PROBES)"""
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

    import unsloth  # Must precede transformers in the pinned Kaggle stack.
    import numpy as np
    import torch
    import transformers

    print('python =', sys.version)
    print('numpy =', np.__version__, np.__file__)
    print('torch =', torch.__version__, torch.__file__)
    print('transformers =', transformers.__version__, transformers.__file__)
    print('unsloth =', unsloth.__file__)
    print('cuda devices =', torch.cuda.device_count())
else:
    print('Dry upload only; no GPU work was started.')"""
    ),
    code_cell(module_source),
    code_cell(
        """if RUN_EXPERIMENT:
    import json
    from pathlib import Path

    input_root = Path('/kaggle/input')
    validation_ids = set(json.loads((Path(COMP_ROOT) / 'arc-agi_evaluation_challenges.json').read_text()))
    subset_roots = {'nvarc_training': discover_subset_root(input_root, 'nvarc_training')}
    full_root = discover_subset_root(input_root, 'nvarc_full')
    full_anchors = {path.name for path in full_root.iterdir() if path.is_dir()}
    assert full_anchors == validation_ids, (len(full_anchors), len(validation_ids))
    print('nvarc_full deliberately excluded: all', len(full_anchors), 'anchors are validation IDs')
    print('subset roots =', {key: str(value) for key, value in subset_roots.items()})
    print('validation IDs excluded as anchors =', len(validation_ids))

    probes = []
    inventory = {}
    for subset, root in subset_roots.items():
        paths = deterministic_sample_paths(
            root,
            count=NUM_PROBES,
            seed=SEED,
            excluded_anchor_ids=validation_ids,
        )
        inventory[subset] = [str(path.relative_to(root)) for path in paths]
        for path in paths:
            probes.append(load_probe_from_path(path, subset=subset, seed=SEED))

    assert len(probes) == NUM_PROBES
    assert len({(probe.subset, probe.puzzle_id) for probe in probes}) == len(probes)
    assert all(probe.anchor_id not in validation_ids for probe in probes)
    print('unique probes =', len(probes))
    print('inventory =', json.dumps(inventory, indent=2))"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import time
    from unsloth import FastLanguageModel

    load_started = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    model = FastLanguageModel.for_inference(model)
    model.eval()
    print('model load seconds =', round(time.perf_counter() - load_started, 2))
    print('tokenizer size =', len(tokenizer))
    assert len(tokenizer) == 16, len(tokenizer)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import gc, json, time
    from collections import Counter, defaultdict
    from pathlib import Path

    records = []
    timings = defaultdict(float)
    for index, probe in enumerate(probes):
        prompt = format_prompt(probe)
        gold_reply = format_reply(probe.gold_output)
        prompt_tokens = tokenizer.encode(prompt)
        gold_tokens = tokenizer.encode(gold_reply)
        if len(prompt_tokens) + len(gold_tokens) > MAX_SEQ_LENGTH:
            records.append({**probe.to_json(), 'status': 'sequence_too_long'})
            continue

        started = time.perf_counter()
        metrics = teacher_forced_metrics(model, tokenizer, prompt, gold_reply)
        timings['teacher_forced_s'] += time.perf_counter() - started
        record = {**probe.to_json(), **metrics}

        if metrics['restricted_greedy_exact']:
            record['status'] = 'teacher_forced_exact'
        else:
            started = time.perf_counter()
            # 30x30 is the largest legal ARC grid: 900 cells, 29 newlines,
            # and one EOS token.  Do not leak the gold output length.
            rollout_limit = min(930, MAX_SEQ_LENGTH - len(prompt_tokens))
            rollout_ids = restricted_greedy_rollout(
                model,
                tokenizer,
                prompt,
                max_new_tokens=rollout_limit,
            )
            timings['rollout_s'] += time.perf_counter() - started
            prediction, invalid_reason = parse_rollout_grid(tokenizer, rollout_ids)
            record.update({
                'rollout_token_ids': rollout_ids,
                'rollout_tokens': len(rollout_ids),
                'rollout_invalid_reason': invalid_reason,
                'prediction': prediction,
            })
            if prediction is None:
                record['status'] = 'invalid_failure_rollout'
            else:
                record['status'] = 'usable_repair_failure'
                record.update(error_mask_diagnostics(prediction, probe.gold_output))

        records.append(record)
        Path(OUTPUT_JSON).write_text(json.dumps({
            'config': {'num_probes': NUM_PROBES, 'subset': 'nvarc_training', 'seed': SEED},
            'records': records,
            'timings': dict(timings),
        }, indent=2))
        print(index + 1, '/', len(probes), probe.subset, probe.puzzle_id, record['status'],
              'tf_exact =', metrics['restricted_greedy_exact'],
              'wrong_argmax =', metrics['wrong_argmax_tokens'],
              'mean_nll =', round(metrics['gold_mean_nll'], 5))
        gc.collect()
        torch.cuda.empty_cache()

    status_counts = Counter(record['status'] for record in records)
    subset_status = {
        subset: Counter(record['status'] for record in records if record['subset'] == subset)
        for subset in subset_roots
    }
    summary = {
        'config': {'num_probes': NUM_PROBES, 'subset': 'nvarc_training', 'seed': SEED},
        'unique_probes': len(probes),
        'status_counts': dict(status_counts),
        'subset_status': {key: dict(value) for key, value in subset_status.items()},
        'timings': dict(timings),
        'records': records,
    }
    Path(OUTPUT_JSON).write_text(json.dumps(summary, indent=2))
    print('SMOKE SUMMARY')
    print(json.dumps({key: value for key, value in summary.items() if key != 'records'}, indent=2))"""
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
