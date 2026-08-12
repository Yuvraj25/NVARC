"""Build the bounded four-GPU production ARC repair mining notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "arc26-repair-production-mining.ipynb"
SOURCE_FILES = ("repair_mining.py", "mine_repair_dataset.py")
SOURCE_HASHES = {
    name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    for name in SOURCE_FILES
}


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


cells = [
    markdown_cell(
        """# ARC repair dataset: bounded production mining

This run mines 512 deterministic leave-one-out probes from clean `nvarc_training` using four L4 workers. It mounts the refreshed `yuvraj/arc2026` repository code, verifies exact source hashes before launching workers, excludes ARC-AGI-2 validation anchors, and writes only real valid-grid repair failures. Ordinary solve replay and zero-mask no-op examples remain reconstructible from the immutable source corpus.

This notebook mines training data only. It does not train or evaluate a repair model."""
    ),
    code_cell(
        f"""RUN_EXPERIMENT = True
NUM_PROBES = 512
WORLD_SIZE = 4
ROLLOUT_BATCH_SIZE = 4
SEED = 20260811
EXPECTED_SOURCE_HASHES = {SOURCE_HASHES!r}

MODEL_PATH = '/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
VALIDATION_PATH = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json'
OUTPUT_DIR = '/kaggle/working/repair_mining_512'
MANIFEST_PATH = '/kaggle/working/repair_mining_512/repair_mining_manifest.json'

print('probes =', NUM_PROBES, 'workers =', WORLD_SIZE, 'rollout batch =', ROLLOUT_BATCH_SIZE)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import hashlib
    from pathlib import Path

    candidates = [
        Path('/kaggle/input/datasets/yuvraj/arc2026/ARC-AGI1/qwen_baseline'),
        Path('/kaggle/input/arc2026/ARC-AGI1/qwen_baseline'),
    ]
    CODE_DIR = next((path for path in candidates if path.is_dir()), None)
    if CODE_DIR is None:
        raise FileNotFoundError(f'Could not find refreshed arc2026 code under {candidates}')

    observed_hashes = {
        name: hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest()
        for name in EXPECTED_SOURCE_HASHES
    }
    if observed_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(
            'Mounted arc2026 code is stale or unexpected. '
            f'expected={EXPECTED_SOURCE_HASHES} observed={observed_hashes}'
        )
    print('verified mounted code =', CODE_DIR)
    print('source hashes =', observed_hashes)"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import os
    import subprocess
    import sys
    import time
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
            str(CODE_DIR / 'mine_repair_dataset.py'),
            '--validation-path', VALIDATION_PATH,
            '--model-path', MODEL_PATH,
            '--output-dir', OUTPUT_DIR,
            '--num-probes', str(NUM_PROBES),
            '--seed', str(SEED),
            '--rank', str(rank),
            '--world-size', str(WORLD_SIZE),
            '--rollout-batch-size', str(ROLLOUT_BATCH_SIZE),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(CODE_DIR),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((rank, process))
        handles.append(handle)
        print('started rank', rank, 'pid', process.pid)

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
            print((output_dir / f'worker{rank}.log').read_text()[-16000:])
        raise RuntimeError(f'worker failures: {failures}')
    print('all workers complete in', round(wall_seconds, 2), 'seconds')"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    import hashlib
    import json
    import math
    from collections import Counter
    from pathlib import Path

    output_dir = Path(OUTPUT_DIR)
    validation_anchors = set(json.loads(Path(VALIDATION_PATH).read_text()))
    summaries = [
        json.loads((output_dir / f'summary.rank{rank}.json').read_text())
        for rank in range(WORLD_SIZE)
    ]
    records = []
    shard_hashes = {}
    for rank in range(WORLD_SIZE):
        path = output_dir / f'repair_failures.rank{rank}.jsonl'
        if path.exists():
            shard_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            records.extend(
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            )
    records.sort(key=lambda record: record['global_index'])

    aggregate = {
        key: sum(summary['counts'][key] for summary in summaries)
        for key in summaries[0]['counts']
    }
    assert aggregate['assigned_probes'] == NUM_PROBES, aggregate
    assert (
        aggregate['sequence_too_long']
        + aggregate['teacher_forced_exact']
        + aggregate['teacher_forced_failures']
        == NUM_PROBES
    ), aggregate
    assert (
        aggregate['usable_repair_failures'] + aggregate['invalid_rollouts']
        == aggregate['teacher_forced_failures']
    ), aggregate
    assert len(records) == aggregate['usable_repair_failures']
    assert len({record['global_index'] for record in records}) == len(records)
    assert len({record['source_relpath'] for record in records}) == len(records)
    assert all(record['record_type'] == 'repair_failure' for record in records)
    assert all('<REPAIR>\\n' in record['input'] for record in records)
    assert all(record['anchor_id'] not in validation_anchors for record in records)
    assert all(record['global_index'] in record['decoder']['batch_member_global_indices'] for record in records)
    assert all(len(record['decoder']['batch_member_global_indices']) <= ROLLOUT_BATCH_SIZE for record in records)

    def split_for(anchor_id):
        value = int(hashlib.sha256(f'{SEED}:split:{anchor_id}'.encode()).hexdigest()[:16], 16)
        bucket = value / float(16 ** 16)
        if bucket < 0.8:
            return 'train'
        if bucket < 0.9:
            return 'dev'
        return 'test'

    for record in records:
        record['split'] = split_for(record['anchor_id'])

    split_anchors = {
        split: {record['anchor_id'] for record in records if record['split'] == split}
        for split in ('train', 'dev', 'test')
    }
    assert not (split_anchors['train'] & split_anchors['dev'])
    assert not (split_anchors['train'] & split_anchors['test'])
    assert not (split_anchors['dev'] & split_anchors['test'])

    combined_path = output_dir / 'repair_failures.all.jsonl'
    combined_path.write_text(''.join(json.dumps(record, separators=(',', ':')) + '\\n' for record in records))
    split_counts = {}
    for split in ('train', 'dev', 'test'):
        split_records = [record for record in records if record['split'] == split]
        split_path = output_dir / f'repair_failures.{split}.jsonl'
        split_path.write_text(
            ''.join(json.dumps(record, separators=(',', ':')) + '\\n' for record in split_records)
        )
        split_counts[split] = {
            'records': len(split_records),
            'anchors': len(split_anchors[split]),
            'sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
        }

    def distribution(values):
        if not values:
            return None
        ordered = sorted(values)
        def percentile(q):
            position = q * (len(ordered) - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return float(ordered[lower])
            return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
        return {
            'min': float(ordered[0]),
            'p10': percentile(0.10),
            'median': percentile(0.50),
            'p90': percentile(0.90),
            'max': float(ordered[-1]),
            'mean': float(sum(ordered) / len(ordered)),
        }

    wrong_cells = [record['total_wrong_missing_or_extra_cells'] for record in records]
    shape_errors = [record for record in records if not record['shape_equal']]
    manifest = {
        'config': {
            'num_probes': NUM_PROBES,
            'world_size': WORLD_SIZE,
            'rollout_batch_size': ROLLOUT_BATCH_SIZE,
            'seed': SEED,
            'source_subset': 'nvarc_training',
            'code_dir': str(CODE_DIR),
            'source_hashes': observed_hashes,
        },
        'wall_seconds': wall_seconds,
        'aggregate_counts': aggregate,
        'records': {
            'combined': len(records),
            'unique_anchors': len({record['anchor_id'] for record in records}),
            'unique_puzzles': len({record['puzzle_id'] for record in records}),
            'shape_errors': len(shape_errors),
            'wrong_cells': distribution(wrong_cells),
            'prediction_shapes': dict(Counter(str(record['prediction_shape']) for record in records)),
            'gold_shapes': dict(Counter(str(record['gold_shape']) for record in records)),
        },
        'splits': split_counts,
        'shard_sha256': shard_hashes,
        'combined_sha256': hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        'worker_summaries': summaries,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in manifest.items() if key != 'worker_summaries'}, indent=2))"""
    ),
    code_cell(
        """if RUN_EXPERIMENT:
    from pathlib import Path
    output_dir = Path(OUTPUT_DIR)
    print('OUTPUT FILES')
    for path in sorted(output_dir.iterdir()):
        print(path.name, path.stat().st_size)
    print('manifest =', MANIFEST_PATH)"""
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
