import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    use_sglang = os.environ.get("ARC_USE_SGLANG") == "1"
    sglang_tp_size = int(os.environ.get("ARC_SGLANG_TP_SIZE", "1"))
    if not (use_sglang and sglang_tp_size > 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")

    if rank > 0:
        while not os.path.exists(f"../worker{rank - 1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"../worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!")


def _load_selected_keys(args, data):
    if args.sglang_infer_from_manifest:
        raise ValueError("selected keys are not used with --sglang-infer-from-manifest")
    if args.keys_json:
        selected = json.loads(args.keys_json)
        assert isinstance(selected, list), "--keys-json must decode to a JSON list"
        return selected
    if args.keys_file:
        with open(args.keys_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    keys = sorted(data.keys())
    if args.limit_keys is not None:
        keys = keys[: args.limit_keys]
    return keys


def _load_manifest_jobs(args):
    with open(args.sglang_infer_from_manifest, "r") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    jobs = []
    for entry in entries:
        if entry.get("status") != "ready":
            continue
        key = entry.get("key")
        adapter_path = entry.get("adapter_path")
        if not key or not adapter_path:
            raise ValueError(f"Invalid manifest entry: {entry}")
        jobs.append({"key": key, "adapter_path": adapter_path})
    jobs.sort(key=lambda item: item["key"])
    if args.limit_keys is not None:
        jobs = jobs[: args.limit_keys]
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--test-path", type=str, default="../input/arc-prize-2024/arc-agi_evaluation_challenges.json")
    parser.add_argument("--model-path", type=str, default="../input/qwen3_4b_grids15_sft139/")
    parser.add_argument("--output-dir", type=str, default="../inference_outputs")
    parser.add_argument("--keys-file", type=str, default=None)
    parser.add_argument("--keys-json", type=str, default=None)
    parser.add_argument("--limit-keys", type=int, default=None)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--use-speculative-dfs", action="store_true")
    parser.add_argument("--use-sglang", action="store_true")
    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=None)
    parser.add_argument("--sglang-adapter-dir", type=str, default="../sglang_adapters")
    parser.add_argument("--sglang-adapter-manifest", type=str, default=None)
    parser.add_argument("--sglang-train-adapters-only", action="store_true")
    parser.add_argument("--sglang-reuse-adapters", action="store_true")
    parser.add_argument("--sglang-infer-from-manifest", type=str, default=None)
    parser.add_argument("--sglang-infer-workers", type=int, default=None)
    parser.add_argument("--sglang-speculative-repeat-len", type=int, default=5)
    parser.add_argument("--sglang-dynamic-repeat", action="store_true")
    parser.add_argument("--dfs-prob-threshold", type=float, default=0.2)
    parser.add_argument("--profile-timings", action="store_true")
    args = parser.parse_args()
    if args.sglang_train_adapters_only and args.sglang_reuse_adapters:
        raise ValueError("--sglang-train-adapters-only and --sglang-reuse-adapters are mutually exclusive")
    if args.sglang_infer_from_manifest and not args.use_sglang:
        raise ValueError("--sglang-infer-from-manifest requires --use-sglang")
    if args.sglang_infer_from_manifest and args.sglang_train_adapters_only:
        raise ValueError("--sglang-infer-from-manifest cannot be combined with --sglang-train-adapters-only")
    effective_nprocs = args.sglang_infer_workers if args.sglang_infer_workers is not None else args.nprocs
    if args.use_sglang and args.sglang_tp_size > 1 and effective_nprocs != 1:
        raise ValueError("--use-sglang with --sglang-tp-size > 1 must run with exactly one worker")
    end_time = args.end_time if args.end_time is not None else time.time() + 12 * 3600
    os.environ["ARC_USE_SPECULATIVE_DFS"] = "1" if args.use_speculative_dfs else "0"
    os.environ["ARC_USE_SGLANG"] = "1" if args.use_sglang else "0"
    os.environ["ARC_SGLANG_TP_SIZE"] = str(args.sglang_tp_size)
    if args.sglang_mem_fraction_static is not None:
        os.environ["ARC_SGLANG_MEM_FRACTION_STATIC"] = str(args.sglang_mem_fraction_static)
    else:
        os.environ.pop("ARC_SGLANG_MEM_FRACTION_STATIC", None)
    os.environ["ARC_SGLANG_ADAPTER_DIR"] = args.sglang_adapter_dir
    os.environ["ARC_SGLANG_ADAPTER_MANIFEST"] = (
        args.sglang_adapter_manifest or os.path.join(args.sglang_adapter_dir, "adapter_manifest.json")
    )
    os.environ["ARC_SGLANG_TRAIN_ADAPTERS_ONLY"] = "1" if args.sglang_train_adapters_only else "0"
    os.environ["ARC_SGLANG_REUSE_ADAPTERS"] = "1" if args.sglang_reuse_adapters else "0"
    os.environ["ARC_SGLANG_PERSISTENT_INFER"] = "1" if args.sglang_infer_from_manifest else "0"
    os.environ["ARC_SGLANG_SPECULATIVE_REPEAT_LEN"] = str(args.sglang_speculative_repeat_len)
    os.environ["ARC_SGLANG_DYNAMIC_REPEAT"] = "1" if args.sglang_dynamic_repeat else "0"
    os.environ["ARC_DFS_PROB_THRESHOLD"] = str(args.dfs_prob_threshold)
    os.environ["ARC_PROFILE_TIMINGS"] = "1" if args.profile_timings else "0"
    os.environ["ARC_TEST_PATH"] = args.test_path
    os.environ["ARC_MODEL_PATH"] = args.model_path
    os.environ["ARC_OUTPUT_DIR"] = args.output_dir
    print(
        "runtime flags:",
        f"speculative_dfs={os.environ['ARC_USE_SPECULATIVE_DFS']}",
        f"use_sglang={os.environ['ARC_USE_SGLANG']}",
        f"sglang_tp_size={os.environ['ARC_SGLANG_TP_SIZE']}",
        f"sglang_mem_fraction_static={os.environ.get('ARC_SGLANG_MEM_FRACTION_STATIC')}",
        f"sglang_adapter_dir={os.environ['ARC_SGLANG_ADAPTER_DIR']}",
        f"sglang_adapter_manifest={os.environ['ARC_SGLANG_ADAPTER_MANIFEST']}",
        f"sglang_train_adapters_only={os.environ['ARC_SGLANG_TRAIN_ADAPTERS_ONLY']}",
        f"sglang_reuse_adapters={os.environ['ARC_SGLANG_REUSE_ADAPTERS']}",
        f"sglang_persistent_infer={os.environ['ARC_SGLANG_PERSISTENT_INFER']}",
        f"sglang_speculative_repeat_len={os.environ['ARC_SGLANG_SPECULATIVE_REPEAT_LEN']}",
        f"sglang_dynamic_repeat={os.environ['ARC_SGLANG_DYNAMIC_REPEAT']}",
        f"dfs_prob_threshold={os.environ['ARC_DFS_PROB_THRESHOLD']}",
        f"profile_timings={os.environ['ARC_PROFILE_TIMINGS']}",
        f"test_path={os.environ['ARC_TEST_PATH']}",
        f"model_path={os.environ['ARC_MODEL_PATH']}",
        f"output_dir={os.environ['ARC_OUTPUT_DIR']}",
    )

    rerun_mode = True
    queue = mp.Manager().Queue()
    if args.sglang_infer_from_manifest:
        jobs = _load_manifest_jobs(args)
        for job in jobs:
            queue.put(job)
    else:
        with open(args.test_path, "r") as f:
            data = json.load(f)
        selected_keys = _load_selected_keys(args, data)
        for key in selected_keys:
            assert key in data, f"Unknown puzzle key: {key}"
            queue.put(key)
    for _ in range(effective_nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=effective_nprocs)
