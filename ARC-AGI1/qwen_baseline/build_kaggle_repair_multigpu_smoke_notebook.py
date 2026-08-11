"""Build a self-contained four-GPU smoke test for repair dataset mining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-repair-multigpu-smoke.ipynb"


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
worker_source = (HERE / "mine_repair_dataset.py").read_text()

cells = [
    markdown_cell(
        """# ARC repair mining: four-GPU smoke

This notebook validates the production sharding path. Four independent workers each see one L4, reconstruct the same 16-puzzle global sample, take disjoint modulo shards, screen with batch size 1, and generate failures in batches of up to 4. Only expensive repair failures are materialized; clean solve replay and zero-mask repair examples remain reconstructible from the clean source corpus."""
    ),
    code_cell(
        """RUN_EXPERIMENT = True
NUM_PROBES = 16
WORLD_SIZE = 4
ROLLOUT_BATCH_SIZE = 4
SEED = 20260811

MODEL_PATH = '/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
VALIDATION_PATH = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json'
CODE_DIR = '/kaggle/working/repair_miner_code'
OUTPUT_DIR = '/kaggle/working/repair_multigpu_shards'
FINAL_JSON = '/kaggle/working/repair_multigpu_smoke.json'

print('probes =', NUM_PROBES, 'workers =', WORLD_SIZE, 'rollout batch =', ROLLOUT_BATCH_SIZE)"""
    ),
    code_cell(
        f"""if RUN_EXPERIMENT:
    from pathlib import Path
    code_dir = Path(CODE_DIR)
    code_dir.mkdir(parents=True, exist_ok=False)
    (code_dir / 'repair_mining.py').write_text({module_source!r})
    (code_dir / 'mine_repair_dataset.py').write_text({worker_source!r})
    print('wrote production worker sources')"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import os, subprocess, sys, time
    from pathlib import Path

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    handles = []
    started = time.perf_counter()
    for rank in range(WORLD_SIZE):
        log_path = output_dir / f'worker{rank}.log'
        handle = log_path.open('w')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(rank)
        env['PYTHONUNBUFFERED'] = '1'
        command = [
            sys.executable,
            str(Path(CODE_DIR) / 'mine_repair_dataset.py'),
            '--validation-path', VALIDATION_PATH,
            '--model-path', MODEL_PATH,
            '--output-dir', OUTPUT_DIR,
            '--num-probes', str(NUM_PROBES),
            '--seed', str(SEED),
            '--rank', str(rank),
            '--world-size', str(WORLD_SIZE),
            '--rollout-batch-size', str(ROLLOUT_BATCH_SIZE),
        ]
        processes.append((rank, subprocess.Popen(command, cwd=CODE_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        handles.append(handle)

    failures = []
    for rank, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append((rank, return_code))
    for handle in handles:
        handle.close()
    wall_seconds = time.perf_counter() - started

    if failures:
        for rank, return_code in failures:
            print('FAILED WORKER', rank, 'return code', return_code)
            print((output_dir / f'worker{rank}.log').read_text()[-12000:])
        raise RuntimeError(f'worker failures: {failures}')
    print('all workers complete in', round(wall_seconds, 2), 'seconds')"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import json
    from pathlib import Path

    output_dir = Path(OUTPUT_DIR)
    summaries = [json.loads((output_dir / f'summary.rank{rank}.json').read_text()) for rank in range(WORLD_SIZE)]
    records = []
    for rank in range(WORLD_SIZE):
        path = output_dir / f'repair_failures.rank{rank}.jsonl'
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    records.sort(key=lambda record: record['global_index'])

    aggregate = {
        key: sum(summary['counts'][key] for summary in summaries)
        for key in summaries[0]['counts']
    }
    assert aggregate == {
        'assigned_probes': 16,
        'sequence_too_long': 0,
        'teacher_forced_exact': 3,
        'teacher_forced_failures': 13,
        'usable_repair_failures': 13,
        'invalid_rollouts': 0,
    }, aggregate
    assert len(records) == 13
    assert len({record['global_index'] for record in records}) == 13
    assert all(record['record_type'] == 'repair_failure' for record in records)
    assert all('<REPAIR>\\n' in record['input'] for record in records)
    assert all(record['global_index'] in record['decoder']['batch_member_global_indices'] for record in records)
    assert all(len(record['decoder']['batch_member_global_indices']) <= ROLLOUT_BATCH_SIZE for record in records)

    result = {
        'config': {
            'num_probes': NUM_PROBES,
            'world_size': WORLD_SIZE,
            'rollout_batch_size': ROLLOUT_BATCH_SIZE,
            'seed': SEED,
        },
        'wall_seconds': wall_seconds,
        'aggregate_counts': aggregate,
        'worker_summaries': summaries,
        'record_global_indices': [record['global_index'] for record in records],
        'record_schema_example': records[0],
    }
    Path(FINAL_JSON).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'record_schema_example'}, indent=2))"""
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
