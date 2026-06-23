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
    parser.add_argument("--use-prefix-cached-rescoring", action="store_true")
    parser.add_argument("--use-speculative-dfs", action="store_true")
    parser.add_argument("--use-sglang", action="store_true")
    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=None)
    parser.add_argument("--sglang-adapter-dir", type=str, default="../sglang_adapters")
    parser.add_argument("--dfs-prob-threshold", type=float, default=0.2)
    parser.add_argument("--profile-timings", action="store_true")
    args = parser.parse_args()
    if args.use_sglang and args.sglang_tp_size > 1 and args.nprocs != 1:
        raise ValueError("--use-sglang with --sglang-tp-size > 1 must run with --nprocs 1 so one SGLang engine can see all GPUs")
    end_time = args.end_time if args.end_time is not None else time.time() + 12 * 3600
    os.environ["ARC_USE_PREFIX_CACHED_RESCORING"] = "1" if args.use_prefix_cached_rescoring else "0"
    os.environ["ARC_USE_SPECULATIVE_DFS"] = "1" if args.use_speculative_dfs else "0"
    os.environ["ARC_USE_SGLANG"] = "1" if args.use_sglang else "0"
    os.environ["ARC_SGLANG_TP_SIZE"] = str(args.sglang_tp_size)
    if args.sglang_mem_fraction_static is not None:
        os.environ["ARC_SGLANG_MEM_FRACTION_STATIC"] = str(args.sglang_mem_fraction_static)
    else:
        os.environ.pop("ARC_SGLANG_MEM_FRACTION_STATIC", None)
    os.environ["ARC_SGLANG_ADAPTER_DIR"] = args.sglang_adapter_dir
    os.environ["ARC_DFS_PROB_THRESHOLD"] = str(args.dfs_prob_threshold)
    os.environ["ARC_PROFILE_TIMINGS"] = "1" if args.profile_timings else "0"
    os.environ["ARC_TEST_PATH"] = args.test_path
    os.environ["ARC_MODEL_PATH"] = args.model_path
    os.environ["ARC_OUTPUT_DIR"] = args.output_dir
    print(
        "runtime flags:",
        f"prefix_cached_rescoring={os.environ['ARC_USE_PREFIX_CACHED_RESCORING']}",
        f"speculative_dfs={os.environ['ARC_USE_SPECULATIVE_DFS']}",
        f"use_sglang={os.environ['ARC_USE_SGLANG']}",
        f"sglang_tp_size={os.environ['ARC_SGLANG_TP_SIZE']}",
        f"sglang_mem_fraction_static={os.environ.get('ARC_SGLANG_MEM_FRACTION_STATIC')}",
        f"sglang_adapter_dir={os.environ['ARC_SGLANG_ADAPTER_DIR']}",
        f"dfs_prob_threshold={os.environ['ARC_DFS_PROB_THRESHOLD']}",
        f"profile_timings={os.environ['ARC_PROFILE_TIMINGS']}",
        f"test_path={os.environ['ARC_TEST_PATH']}",
        f"model_path={os.environ['ARC_MODEL_PATH']}",
        f"output_dir={os.environ['ARC_OUTPUT_DIR']}",
    )

    rerun_mode = True
    with open(args.test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()
    selected_keys = _load_selected_keys(args, data)
    for key in selected_keys:
        assert key in data, f"Unknown puzzle key: {key}"
        queue.put(key)
    for _ in range(args.nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=args.nprocs)
