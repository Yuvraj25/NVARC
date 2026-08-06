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

from arc_decoder import score_full_probmul_3, score_kgmon, score_kgmon_median, ArcDecoder
from arc_loader import ArcDataset


ROOT_DIR = Path(__file__).resolve().parent

SELECTION_ALGORITHMS = {
    "score_kgmon": score_kgmon,
    "score_kgmon_median": score_kgmon_median,
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


def _clear_worker_sentinels(prefix: str) -> None:
    for path in ROOT_DIR.parent.glob(f"{prefix}*"):
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


def _start_starter(
    args,
    chunk_keys: list[str],
    phase: str,
    *,
    cuda_device_offset: int = 0,
    producer_done_path: Path | None = None,
) -> subprocess.Popen:
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
        sentinel_prefix = "worker_train_"
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
        sentinel_prefix = "worker_infer_"
    elif phase == "stream-infer":
        if producer_done_path is None:
            raise ValueError("stream-infer requires producer_done_path")
        cmd.extend(
            [
                "--sglang-stream-manifest",
                args.sglang_adapter_manifest,
                "--sglang-producer-done",
                str(producer_done_path),
                "--sglang-consume-adapters",
                "--sglang-infer-workers",
                str(args.infer_workers),
                "--nprocs",
                "1",
            ]
        )
        sentinel_prefix = "worker_infer_"
    else:
        raise ValueError(f"Unknown phase: {phase}")
    cmd.extend(["--cuda-device-offset", str(cuda_device_offset)])
    _clear_worker_sentinels(sentinel_prefix)
    print(f"[chunked] running {phase}: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    stack_path = env.get("ARC_SGLANG_STACK_PATH")
    if stack_path:
        pythonpath = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
        excluded = {stack_path, "/kaggle/working/arc_stack"}
        pythonpath = [entry for entry in pythonpath if entry not in excluded]
        if phase in {"infer", "stream-infer"}:
            pythonpath.append(stack_path)
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        else:
            env.pop("PYTHONPATH", None)
    return subprocess.Popen(cmd, cwd=ROOT_DIR, env=env)


def _run_starter(args, chunk_keys: list[str], phase: str) -> None:
    process = _start_starter(args, chunk_keys, phase)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


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


def _validate_inference_manifest(manifest_path: Path, selected_keys: list[str]) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Inference-only adapter manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path, {"entries": []})
    entries = {
        entry.get("key"): entry
        for entry in manifest.get("entries", [])
        if entry.get("status") == "ready" and entry.get("key")
    }
    missing_keys = [key for key in selected_keys if key not in entries]
    if missing_keys:
        raise ValueError(f"Inference-only manifest has no ready adapter for keys: {missing_keys}")

    missing_paths = [
        (key, entries[key].get("adapter_path"))
        for key in selected_keys
        if not entries[key].get("adapter_path") or not Path(entries[key]["adapter_path"]).is_dir()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Inference-only adapter paths are missing. Rebase paths from the original "
            f"/kaggle/working location to the attached input: {missing_paths}"
        )


def _resident_adapter_count(adapter_dir: Path) -> int:
    return sum(path.is_dir() for path in adapter_dir.iterdir()) if adapter_dir.exists() else 0


def _wait_for_adapter_capacity(
    adapter_dir: Path,
    max_resident: int,
    incoming: int,
    inference_process: subprocess.Popen,
    end_time: float | None,
) -> bool:
    last_report = 0.0
    while _resident_adapter_count(adapter_dir) + incoming > max_resident:
        return_code = inference_process.poll()
        if return_code is not None:
            if return_code:
                raise subprocess.CalledProcessError(return_code, inference_process.args)
            raise RuntimeError("Streaming inference exited before adapter production completed")
        if end_time is not None and time.time() > end_time:
            return False
        if time.time() - last_report >= 30:
            print(
                f"[stream] waiting for adapter capacity: resident={_resident_adapter_count(adapter_dir)} "
                f"incoming={incoming} max={max_resident}",
                flush=True,
            )
            last_report = time.time()
        time.sleep(1)
    return True


def _run_streaming_pipeline(
    args,
    pending_keys: list[str],
    output_dir: Path,
    adapter_dir: Path,
    manifest_path: Path,
    state_path: Path,
    submission_path: Path,
    state: dict,
    done_keys: set[str],
) -> None:
    batches = list(_chunked(pending_keys, args.chunk_size))
    if args.max_chunks is not None:
        batches = batches[: args.max_chunks]
    streaming_keys = [key for batch in batches for key in batch]
    if not streaming_keys:
        _write_submission(args.test_path, output_dir, submission_path, args.selection_algorithm)
        return

    producer_done_path = adapter_dir / "stream_producer_done"
    try:
        producer_done_path.unlink()
    except FileNotFoundError:
        pass

    inference_process = _start_starter(
        args,
        streaming_keys,
        phase="stream-infer",
        cuda_device_offset=args.train_nprocs,
        producer_done_path=producer_done_path,
    )
    trained_keys = []
    training_error = None
    try:
        for batch_index, batch_keys in enumerate(batches, 1):
            if args.end_time is not None and time.time() > args.end_time:
                print("[stream] reached end_time before starting next training batch", flush=True)
                break
            if not _wait_for_adapter_capacity(
                adapter_dir,
                args.max_ready_adapters,
                len(batch_keys),
                inference_process,
                args.end_time,
            ):
                print("[stream] reached end_time while waiting for adapter capacity", flush=True)
                break
            print(f"[stream] training batch {batch_index}: {batch_keys}", flush=True)
            training_process = _start_starter(args, batch_keys, phase="train", cuda_device_offset=0)
            return_code = training_process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, training_process.args)
            trained_keys.extend(batch_keys)
    except BaseException as exc:
        training_error = exc
    finally:
        producer_done_path.write_text("done\n")

    inference_return_code = inference_process.wait()
    if inference_return_code:
        raise subprocess.CalledProcessError(inference_return_code, inference_process.args)
    if training_error is not None:
        raise training_error

    _write_submission(args.test_path, output_dir, submission_path, args.selection_algorithm)
    completed_keys, missing_outputs = _partition_completed_keys(args.test_path, output_dir, trained_keys)
    incomplete_keys = [key for key in trained_keys if key not in completed_keys]
    state.setdefault("history", []).append(
        {
            "mode": "overlap_train_infer",
            "keys": trained_keys,
            "completed_keys": completed_keys,
            "incomplete_keys": incomplete_keys,
            "missing_outputs": missing_outputs,
            "completed_at": time.time(),
        }
    )
    done_keys.update(completed_keys)
    state["done_keys"] = sorted(done_keys)
    _save_state(state_path, state)

    _cleanup_adapters(adapter_dir, trained_keys)
    _prune_manifest(manifest_path, trained_keys)
    print(
        f"[stream] finished: trained={len(trained_keys)} completed={len(completed_keys)} "
        f"incomplete={len(incomplete_keys)}",
        flush=True,
    )
    if incomplete_keys:
        print(f"[stream] incomplete_keys={incomplete_keys} missing_outputs={missing_outputs}", flush=True)


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
    parser.add_argument(
        "--overlap-train-infer",
        action="store_true",
        help="Run adapter training and persistent SGLang inference concurrently on disjoint GPUs.",
    )
    parser.add_argument(
        "--max-ready-adapters",
        type=int,
        default=16,
        help="Maximum resident adapter directories in overlap mode.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip adapter training and infer from the supplied adapter manifest.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.sglang_tp_size > 1:
        args.train_nprocs = 1
        args.infer_workers = 1

    if args.overlap_train_infer:
        if args.inference_only:
            parser.error("--overlap-train-infer cannot be combined with --inference-only")
        if args.keep_adapters:
            parser.error("--overlap-train-infer requires adapter consumption; remove --keep-adapters")
        if args.sglang_tp_size != 1:
            parser.error("--overlap-train-infer currently requires --sglang-tp-size 1")
        if args.train_nprocs + args.infer_workers > 4:
            parser.error("overlap workers exceed the four Kaggle GPUs")
        if args.chunk_size > args.max_ready_adapters:
            parser.error("--chunk-size cannot exceed --max-ready-adapters in overlap mode")

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
    if args.inference_only:
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Inference-only adapter directory not found: {adapter_dir}")
    else:
        adapter_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    done_keys = set(state.get("done_keys", []))
    selected_keys = _load_selected_keys(args)
    if args.inference_only:
        _validate_inference_manifest(manifest_path, selected_keys)
    pending_keys = [key for key in selected_keys if key not in done_keys]

    print(
        f"[chunked] total_keys={len(selected_keys)} done_keys={len(done_keys)} pending_keys={len(pending_keys)} "
        f"chunk_size={args.chunk_size} train_nprocs={args.train_nprocs} infer_workers={args.infer_workers} "
        f"inference_only={args.inference_only}",
        flush=True,
    )

    if args.overlap_train_infer:
        _run_streaming_pipeline(
            args,
            pending_keys,
            output_dir,
            adapter_dir,
            manifest_path,
            state_path,
            submission_path,
            state,
            done_keys,
        )
        _write_submission(args.test_path, output_dir, submission_path, args.selection_algorithm)
        print(f"[chunked] complete; final submission at {submission_path}", flush=True)
        return

    for chunk_index, chunk_keys in enumerate(_chunked(pending_keys, args.chunk_size), start=1):
        if args.max_chunks is not None and chunk_index > args.max_chunks:
            print(f"[chunked] reached max_chunks={args.max_chunks}; stopping", flush=True)
            break
        if args.end_time is not None and time.time() > args.end_time:
            print("[chunked] reached end_time before starting next chunk; stopping", flush=True)
            break

        print(f"[chunked] chunk {chunk_index}: {chunk_keys}", flush=True)
        if not args.inference_only:
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

        if not args.inference_only and not args.keep_adapters and chunk_keys:
            _cleanup_adapters(adapter_dir, chunk_keys)
            _prune_manifest(manifest_path, chunk_keys)
            print(f"[chunked] cleaned adapters for processed keys in chunk {chunk_index}", flush=True)

        if incomplete_keys:
            print("[chunked] continuing after partial chunk; incomplete keys remain pending for future reruns", flush=True)

    _write_submission(args.test_path, output_dir, submission_path, args.selection_algorithm)
    print(f"[chunked] complete; final submission at {submission_path}", flush=True)


if __name__ == "__main__":
    main()
