#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from arc_decoder import score_full_probmul_3, score_kgmon, ArcDecoder
from arc_loader import ArcDataset


ROOT_DIR = Path(__file__).resolve().parent

SELECTION_ALGORITHMS = {
    "score_kgmon": score_kgmon,
    "score_full_probmul_3": score_full_probmul_3,
}


def _safe_path_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
        tmp_path = Path(f.name)
    tmp_path.replace(path)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def _load_selected_keys(args) -> list[str]:
    with open(args.test_path, "r") as f:
        data = json.load(f)
    if args.keys_json:
        keys = json.loads(args.keys_json)
        assert isinstance(keys, list), "--keys-json must decode to a JSON list"
    elif args.keys_file:
        with open(args.keys_file, "r") as f:
            keys = [line.strip() for line in f if line.strip()]
    else:
        keys = sorted(data.keys())
        if args.limit_keys is not None:
            keys = keys[: args.limit_keys]
    for key in keys:
        assert key in data, f"Unknown puzzle key: {key}"
    return keys


def _chunked(keys: list[str], chunk_size: int):
    for i in range(0, len(keys), chunk_size):
        yield keys[i : i + chunk_size]


def _default_manifest_path(adapter_dir: Path) -> Path:
    return adapter_dir / "adapter_manifest.json"


def _default_state_path(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}_chunk_state.json"


def _default_submission_path(output_dir: Path) -> Path:
    return output_dir.parent / "submission.json"


def _clear_worker_sentinels() -> None:
    for path in ROOT_DIR.parent.glob("worker*"):
        if path.is_file():
            path.unlink()


def _starter_common_args(args, chunk_keys: list[str]) -> list[str]:
    cmd = [
        "--use-sglang",
        "--test-path",
        args.test_path,
        "--model-path",
        args.model_path,
        "--output-dir",
        args.output_dir,
        "--keys-json",
        json.dumps(chunk_keys),
        "--sglang-tp-size",
        str(args.sglang_tp_size),
        "--sglang-adapter-dir",
        args.sglang_adapter_dir,
        "--sglang-adapter-manifest",
        args.sglang_adapter_manifest,
        "--dfs-prob-threshold",
        str(args.dfs_prob_threshold),
        "--sglang-speculative-repeat-len",
        str(args.sglang_speculative_repeat_len),
    ]
    if args.end_time is not None:
        cmd.extend(["--end-time", str(args.end_time)])
    if args.sglang_mem_fraction_static is not None:
        cmd.extend(["--sglang-mem-fraction-static", str(args.sglang_mem_fraction_static)])
    if args.profile_timings:
        cmd.append("--profile-timings")
    if args.use_speculative_dfs:
        cmd.append("--use-speculative-dfs")
    if args.sglang_dynamic_repeat:
        cmd.append("--sglang-dynamic-repeat")
    return cmd


def _run_starter(args, chunk_keys: list[str], phase: str) -> None:
    cmd = [sys.executable, "starter.py"]
    cmd.extend(_starter_common_args(args, chunk_keys))
    if phase == "train":
        cmd.extend(
            [
                "--sglang-train-adapters-only",
                "--nprocs",
                str(args.train_nprocs),
            ]
        )
    elif phase == "infer":
        cmd.extend(
            [
                "--sglang-infer-from-manifest",
                args.sglang_adapter_manifest,
                "--sglang-infer-workers",
                str(args.infer_workers),
                "--nprocs",
                "1",
            ]
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")
    _clear_worker_sentinels()
    print(f"[chunked] running {phase}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True, env=os.environ.copy())


def _prune_manifest(manifest_path: Path, chunk_keys: list[str]) -> None:
    manifest = _load_json(manifest_path, {"version": 1, "entries": []})
    chunk_set = set(chunk_keys)
    manifest["entries"] = [entry for entry in manifest.get("entries", []) if entry.get("key") not in chunk_set]
    _save_json_atomic(manifest_path, manifest)


def _cleanup_adapters(adapter_dir: Path, chunk_keys: list[str]) -> None:
    for key in chunk_keys:
        path = adapter_dir / _safe_path_key(key)
        shutil.rmtree(path, ignore_errors=True)


def _expected_output_base_keys(test_path: str, puzzle_keys: list[str]) -> dict[str, set[str]]:
    with open(test_path, "r") as f:
        data = json.load(f)
    return {key: {f"{key}_{i}" for i in range(len(data[key]["test"]))} for key in puzzle_keys}


def _observed_output_base_keys(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    return {path.name.split(".")[0] for path in output_dir.iterdir() if path.is_file()}


def _partition_completed_keys(test_path: str, output_dir: Path, puzzle_keys: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    expected = _expected_output_base_keys(test_path, puzzle_keys)
    observed = _observed_output_base_keys(output_dir)
    completed = []
    missing = {}
    for key in puzzle_keys:
        missing_outputs = sorted(expected[key] - observed)
        if missing_outputs:
            missing[key] = missing_outputs
        else:
            completed.append(key)
    return completed, missing


def _write_submission(test_path: str, output_dir: Path, submission_path: Path, selection_algorithm: str) -> None:
    dataset = ArcDataset.from_file(test_path)
    decoder = ArcDecoder(dataset, n_guesses=2)
    if output_dir.exists():
        output_files = [path for path in output_dir.iterdir() if path.is_file()]
        if output_files:
            decoder.load_decoded_results(str(output_dir))
    results = None
    if decoder.decoded_results:
        results = decoder.run_selection_algo(SELECTION_ALGORITHMS[selection_algorithm])
    submission = dataset.get_submission(results)
    _save_json_atomic(submission_path, submission)
    print(f"[chunked] wrote submission to {submission_path}", flush=True)


def _load_state(state_path: Path) -> dict:
    return _load_json(state_path, {"version": 1, "done_keys": [], "history": []})


def _save_state(state_path: Path, state: dict) -> None:
    _save_json_atomic(state_path, state)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--test-path", type=str, default="../input/arc-prize-2024/arc-agi_evaluation_challenges.json")
    parser.add_argument("--model-path", type=str, default="../input/qwen3_4b_grids15_sft139/")
    parser.add_argument("--output-dir", type=str, default="../inference_outputs")
    parser.add_argument("--keys-file", type=str, default=None)
    parser.add_argument("--keys-json", type=str, default=None)
    parser.add_argument("--limit-keys", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--train-nprocs", type=int, default=4)
    parser.add_argument("--infer-workers", type=int, default=4)
    parser.add_argument("--use-speculative-dfs", action="store_true")
    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=None)
    parser.add_argument("--sglang-adapter-dir", type=str, default="../sglang_adapters")
    parser.add_argument("--sglang-adapter-manifest", type=str, default=None)
    parser.add_argument("--sglang-speculative-repeat-len", type=int, default=5)
    parser.add_argument("--sglang-dynamic-repeat", action="store_true")
    parser.add_argument("--dfs-prob-threshold", type=float, default=0.2)
    parser.add_argument("--profile-timings", action="store_true")
    parser.add_argument("--selection-algorithm", choices=sorted(SELECTION_ALGORITHMS), default="score_kgmon")
    parser.add_argument("--state-path", type=str, default=None)
    parser.add_argument("--submission-path", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--keep-adapters", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    test_path = Path(args.test_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    adapter_dir = Path(args.sglang_adapter_dir).resolve()
    manifest_path = Path(args.sglang_adapter_manifest).resolve() if args.sglang_adapter_manifest else _default_manifest_path(adapter_dir)
    state_path = Path(args.state_path).resolve() if args.state_path else _default_state_path(output_dir)
    submission_path = Path(args.submission_path).resolve() if args.submission_path else _default_submission_path(output_dir)

    args.test_path = str(test_path)
    args.model_path = str(model_path)
    args.output_dir = str(output_dir)
    args.sglang_adapter_dir = str(adapter_dir)
    args.sglang_adapter_manifest = str(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    done_keys = set(state.get("done_keys", []))
    selected_keys = _load_selected_keys(args)
    pending_keys = [key for key in selected_keys if key not in done_keys]

    print(
        f"[chunked] total_keys={len(selected_keys)} done_keys={len(done_keys)} pending_keys={len(pending_keys)} "
        f"chunk_size={args.chunk_size} train_nprocs={args.train_nprocs} infer_workers={args.infer_workers}",
        flush=True,
    )

    for chunk_index, chunk_keys in enumerate(_chunked(pending_keys, args.chunk_size), start=1):
        if args.max_chunks is not None and chunk_index > args.max_chunks:
            print(f"[chunked] reached max_chunks={args.max_chunks}; stopping", flush=True)
            break
        if args.end_time is not None and time.time() > args.end_time:
            print("[chunked] reached end_time before starting next chunk; stopping", flush=True)
            break

        print(f"[chunked] chunk {chunk_index}: {chunk_keys}", flush=True)
        _run_starter(args, chunk_keys, phase="train")
        _run_starter(args, chunk_keys, phase="infer")
        _write_submission(args.test_path, output_dir, submission_path, args.selection_algorithm)
        completed_keys, missing_outputs = _partition_completed_keys(args.test_path, output_dir, chunk_keys)
        incomplete_keys = [key for key in chunk_keys if key not in completed_keys]

        state.setdefault("history", []).append(
            {
                "chunk_index": chunk_index,
                "keys": chunk_keys,
                "completed_keys": completed_keys,
                "incomplete_keys": incomplete_keys,
                "missing_outputs": missing_outputs,
                "completed_at": time.time(),
            }
        )
        done_keys.update(completed_keys)
        state["done_keys"] = sorted(done_keys)
        _save_state(state_path, state)

        if incomplete_keys:
            print(
                f"[chunked] chunk {chunk_index} incomplete; completed_keys={completed_keys} "
                f"incomplete_keys={incomplete_keys} missing_outputs={missing_outputs}",
                flush=True,
            )
        else:
            print(f"[chunked] chunk {chunk_index} complete for all keys", flush=True)

        if not args.keep_adapters and completed_keys:
            _cleanup_adapters(adapter_dir, completed_keys)
            _prune_manifest(manifest_path, completed_keys)
            print(f"[chunked] cleaned adapters for completed keys in chunk {chunk_index}", flush=True)

        if incomplete_keys:
            print("[chunked] stopping after partial chunk so unfinished keys can be resumed safely", flush=True)
            break

    _write_submission(args.test_path, output_dir, submission_path, args.selection_algorithm)
    print(f"[chunked] complete; final submission at {submission_path}", flush=True)


if __name__ == "__main__":
    main()
