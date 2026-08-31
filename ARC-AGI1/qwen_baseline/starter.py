import argparse
import json
import os
import tempfile
import time

import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, cuda_device_offset, sentinel_prefix):
    use_sglang = os.environ.get("ARC_USE_SGLANG") == "1"
    sglang_tp_size = int(os.environ.get("ARC_SGLANG_TP_SIZE", "1"))
    if not (use_sglang and sglang_tp_size > 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_offset + rank)
    torch.set_default_device("cpu")

    # Unsloth dynamically generates RL trainer modules during import.
    # With multiple spawned workers, the shared default compile cache can race
    # and intermittently produce missing Unsloth*Trainer attributes.
    compile_root = os.path.join(tempfile.gettempdir(), f"unsloth_compile_rank{rank}_pid{os.getpid()}")
    os.makedirs(compile_root, exist_ok=True)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = compile_root

    if rank > 0:
        while not os.path.exists(f"../{sentinel_prefix}{rank - 1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"../{sentinel_prefix}{rank}", "w") as f:
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


def _estimated_task_work(task):
    def grid_tokens(grid):
        return len(grid) * (len(grid[0]) + 1)

    train_tokens = sum(
        grid_tokens(pair["input"]) + grid_tokens(pair["output"])
        for pair in task["train"]
    )
    ratios = sorted(
        grid_tokens(pair["output"]) / max(1, grid_tokens(pair["input"]))
        for pair in task["train"]
    )
    median_ratio = ratios[len(ratios) // 2]
    test_tokens = sum(
        grid_tokens(item["input"]) * (1 + median_ratio)
        for item in task["test"]
    )
    return train_tokens * 16 + test_tokens * 8 * len(task["test"])


def _load_manifest_jobs(args):
    with open(args.sglang_infer_from_manifest, "r") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    selected = None
    if args.keys_json:
        selected = json.loads(args.keys_json)
        assert isinstance(selected, list), "--keys-json must decode to a JSON list"
        selected = set(selected)
    elif args.keys_file:
        with open(args.keys_file, "r") as f:
            selected = {line.strip() for line in f if line.strip()}
    jobs = []
    for entry in entries:
        if entry.get("status") != "ready":
            continue
        key = entry.get("key")
        adapter_path = entry.get("adapter_path")
        if not key or not adapter_path:
            raise ValueError(f"Invalid manifest entry: {entry}")
        if selected is not None and key not in selected:
            continue
        jobs.append({"key": key, "adapter_path": adapter_path})
    jobs.sort(key=lambda item: item["key"])
    if args.limit_keys is not None:
        jobs = jobs[: args.limit_keys]
    return jobs


def _load_streaming_manifest_jobs(args, queued_keys):
    if not os.path.isfile(args.sglang_stream_manifest):
        return []
    with open(args.sglang_stream_manifest, "r") as f:
        data = json.load(f)
    selected = None
    if args.keys_json:
        selected = set(json.loads(args.keys_json))
    elif args.keys_file:
        with open(args.keys_file, "r") as f:
            selected = {line.strip() for line in f if line.strip()}
    jobs = []
    for entry in data.get("entries", []):
        key = entry.get("key")
        if entry.get("status") != "ready" or not key or key in queued_keys:
            continue
        if selected is not None and key not in selected:
            continue
        adapter_path = entry.get("adapter_path")
        if adapter_path and os.path.isdir(adapter_path):
            jobs.append({"key": key, "adapter_path": adapter_path})
    jobs.sort(key=lambda item: item["key"])
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
    parser.add_argument("--cuda-device-offset", type=int, default=0)
    parser.add_argument("--use-speculative-dfs", action="store_true")
    parser.add_argument("--use-unsloth-multitoken-dfs", action="store_true")
    parser.add_argument("--use-unsloth-structured-rows", action="store_true")
    parser.add_argument("--unsloth-multitoken-repeat-len", type=int, default=9)
    parser.add_argument("--use-sglang", action="store_true")
    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=None)
    parser.add_argument("--sglang-adapter-dir", type=str, default="../sglang_adapters")
    parser.add_argument("--sglang-adapter-manifest", type=str, default=None)
    parser.add_argument("--sglang-train-adapters-only", action="store_true")
    parser.add_argument("--sglang-reuse-adapters", action="store_true")
    parser.add_argument("--sglang-infer-from-manifest", type=str, default=None)
    parser.add_argument("--sglang-stream-manifest", type=str, default=None)
    parser.add_argument("--sglang-producer-done", type=str, default=None)
    parser.add_argument("--sglang-consume-adapters", action="store_true")
    parser.add_argument("--sglang-infer-workers", type=int, default=None)
    parser.add_argument("--sglang-speculative-repeat-len", type=int, default=5)
    parser.add_argument("--sglang-dynamic-repeat", action="store_true")
    parser.add_argument("--dfs-prob-threshold", type=float, default=0.2)
    parser.add_argument("--eval-color-permutations", type=int, default=2)
    parser.add_argument("--shared-eval-augmentations", action="store_true")
    parser.add_argument("--adaptive-output-dir", type=str, default=None)
    parser.add_argument("--adaptive-dfs-prob-threshold", type=float, default=0.1)
    parser.add_argument("--adaptive-color-permutations", type=int, default=3)
    parser.add_argument("--adaptive-min-unique-candidates", type=int, default=2)
    parser.add_argument("--cheap-first", action="store_true")
    parser.add_argument("--profile-timings", action="store_true")
    parser.add_argument(
        "--ttft-method",
        choices=[
            "full_sft",
            "one_pass_ss",
            "reduced_sft",
            "reduced_plus_sft_c",
            "reduced_plus_opsd",
            "loo_repair_mix",
            "warm_repair_mix",
        ],
        default="full_sft",
    )
    parser.add_argument("--fixed-candidate-dir", type=str, default=None)
    parser.add_argument("--selected-augmentations-path", type=str, default=None)
    parser.add_argument("--opsd-log-dir", type=str, default="../opsd_logs")
    parser.add_argument("--opsd-min-train-pairs", type=int, default=3)
    parser.add_argument("--opsd-color-permutations", type=int, default=2)
    parser.add_argument("--opsd-cross-view-probability", type=float, default=0.2)
    parser.add_argument("--opsd-max-updates", type=int, default=16)
    parser.add_argument("--opsd-learning-rate", type=float, default=5e-5)
    parser.add_argument("--opsd-temperature", type=float, default=1.0)
    parser.add_argument("--opsd-top-p", type=float, default=1.0)
    parser.add_argument("--opsd-lambda-ce", type=float, default=0.0)
    parser.add_argument("--repair-ttft-log-dir", type=str, default="../repair_ttft_logs")
    parser.add_argument("--repair-ttft-total-steps", type=int, default=128)
    parser.add_argument("--repair-ttft-loo-stage1-steps", type=int, default=64)
    parser.add_argument("--repair-ttft-warm-stage1-steps", type=int, default=32)
    parser.add_argument("--repair-ttft-stage2-repair-fraction", type=float, default=0.5)
    parser.add_argument("--repair-ttft-loo-heldout-views", type=int, default=16)
    parser.add_argument("--repair-ttft-loo-seen-views", type=int, default=4)
    parser.add_argument("--repair-ttft-warm-views-per-pair", type=int, default=8)
    parser.add_argument("--scheduled-sampling-warmup-steps", type=int, default=32)
    parser.add_argument("--scheduled-sampling-total-steps", type=int, default=128)
    parser.add_argument("--scheduled-sampling-mix-probability", type=float, default=0.5)
    parser.add_argument("--scheduled-sampling-log-dir", type=str, default="../scheduled_sampling_logs")
    parser.add_argument(
        "--canon-ac-state",
        type=str,
        default=None,
        help="Path to canon_ac.pt produced by staged global Canon training.",
    )
    args = parser.parse_args()
    if args.sglang_train_adapters_only and args.sglang_reuse_adapters:
        raise ValueError("--sglang-train-adapters-only and --sglang-reuse-adapters are mutually exclusive")
    if args.sglang_infer_from_manifest and not args.use_sglang:
        raise ValueError("--sglang-infer-from-manifest requires --use-sglang")
    if args.sglang_infer_from_manifest and args.sglang_train_adapters_only:
        raise ValueError("--sglang-infer-from-manifest cannot be combined with --sglang-train-adapters-only")
    if args.selected_augmentations_path and not args.sglang_infer_from_manifest:
        raise ValueError("--selected-augmentations-path requires --sglang-infer-from-manifest")
    if args.sglang_stream_manifest and not args.use_sglang:
        raise ValueError("--sglang-stream-manifest requires --use-sglang")
    if args.sglang_stream_manifest and (args.sglang_infer_from_manifest or args.sglang_train_adapters_only):
        raise ValueError("--sglang-stream-manifest is a distinct persistent-inference mode")
    if args.sglang_stream_manifest and not args.sglang_producer_done:
        raise ValueError("--sglang-stream-manifest requires --sglang-producer-done")
    effective_nprocs = args.sglang_infer_workers if args.sglang_infer_workers is not None else args.nprocs
    if args.use_sglang and args.sglang_tp_size > 1 and effective_nprocs != 1:
        raise ValueError("--use-sglang with --sglang-tp-size > 1 must run with exactly one worker")
    if args.ttft_method != "full_sft" and args.use_sglang:
        raise ValueError("Reduced-pair TTFT methods are only supported by the Unsloth/HF worker")
    if args.use_unsloth_multitoken_dfs and args.use_sglang:
        raise ValueError("--use-unsloth-multitoken-dfs is only supported by the Unsloth/HF worker")
    if args.canon_ac_state and args.use_sglang:
        raise ValueError("Canon-AC is initially supported only by the Unsloth/HF worker")
    if args.unsloth_multitoken_repeat_len < 2:
        raise ValueError("--unsloth-multitoken-repeat-len must be at least 2")
    if args.use_unsloth_structured_rows and not args.use_unsloth_multitoken_dfs:
        raise ValueError(
            "--use-unsloth-structured-rows requires --use-unsloth-multitoken-dfs"
        )
    if args.use_unsloth_structured_rows and args.canon_ac_state:
        raise ValueError(
            "--use-unsloth-structured-rows is initially a Vanilla-only experiment"
        )
    if args.eval_color_permutations < 1:
        raise ValueError("--eval-color-permutations must be positive")
    if args.adaptive_color_permutations < 1:
        raise ValueError("--adaptive-color-permutations must be positive")
    if args.adaptive_min_unique_candidates < 1:
        raise ValueError("--adaptive-min-unique-candidates must be positive")
    if not 0.0 < args.adaptive_dfs_prob_threshold < 1.0:
        raise ValueError("--adaptive-dfs-prob-threshold must be in (0, 1)")
    if args.adaptive_output_dir and not args.shared_eval_augmentations:
        raise ValueError(
            "--adaptive-output-dir requires --shared-eval-augmentations"
        )
    if args.opsd_min_train_pairs < 3:
        raise ValueError("--opsd-min-train-pairs must be at least 3")
    if args.opsd_color_permutations < 1:
        raise ValueError("--opsd-color-permutations must be positive")
    if args.opsd_max_updates < 1:
        raise ValueError("--opsd-max-updates must be positive")
    if not 0.0 <= args.opsd_cross_view_probability <= 1.0:
        raise ValueError("--opsd-cross-view-probability must be in [0, 1]")
    if args.repair_ttft_total_steps != 128:
        raise ValueError("--repair-ttft-total-steps must remain 128 for the controlled experiment")
    if not 0.0 <= args.repair_ttft_stage2_repair_fraction <= 1.0:
        raise ValueError("--repair-ttft-stage2-repair-fraction must be in [0, 1]")
    if args.scheduled_sampling_warmup_steps < 0:
        raise ValueError("--scheduled-sampling-warmup-steps must be non-negative")
    if args.scheduled_sampling_total_steps < 1:
        raise ValueError("--scheduled-sampling-total-steps must be positive")
    if (
        args.ttft_method == "one_pass_ss"
        and args.scheduled_sampling_warmup_steps >= args.scheduled_sampling_total_steps
    ):
        raise ValueError(
            "--scheduled-sampling-warmup-steps must be smaller than "
            "--scheduled-sampling-total-steps"
        )
    if not 0.0 <= args.scheduled_sampling_mix_probability <= 1.0:
        raise ValueError("--scheduled-sampling-mix-probability must be in [0, 1]")
    end_time = args.end_time if args.end_time is not None else time.time() + 12 * 3600
    os.environ["ARC_USE_SPECULATIVE_DFS"] = "1" if args.use_speculative_dfs else "0"
    os.environ["ARC_USE_UNSLOTH_MULTITOKEN_DFS"] = "1" if args.use_unsloth_multitoken_dfs else "0"
    os.environ["ARC_USE_UNSLOTH_STRUCTURED_ROWS"] = "1" if args.use_unsloth_structured_rows else "0"
    os.environ["ARC_UNSLOTH_MULTITOKEN_REPEAT_LEN"] = str(args.unsloth_multitoken_repeat_len)
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
    os.environ["ARC_SGLANG_PERSISTENT_INFER"] = "1" if (args.sglang_infer_from_manifest or args.sglang_stream_manifest) else "0"
    os.environ["ARC_SGLANG_CONSUME_ADAPTERS"] = "1" if args.sglang_consume_adapters else "0"
    os.environ["ARC_SGLANG_SPECULATIVE_REPEAT_LEN"] = str(args.sglang_speculative_repeat_len)
    os.environ["ARC_SGLANG_DYNAMIC_REPEAT"] = "1" if args.sglang_dynamic_repeat else "0"
    os.environ["ARC_DFS_PROB_THRESHOLD"] = str(args.dfs_prob_threshold)
    os.environ["ARC_EVAL_COLOR_PERMUTATIONS"] = str(args.eval_color_permutations)
    os.environ["ARC_SHARED_EVAL_AUGMENTATIONS"] = (
        "1" if args.shared_eval_augmentations else "0"
    )
    if args.adaptive_output_dir is not None:
        os.environ["ARC_ADAPTIVE_OUTPUT_DIR"] = args.adaptive_output_dir
    else:
        os.environ.pop("ARC_ADAPTIVE_OUTPUT_DIR", None)
    os.environ["ARC_ADAPTIVE_DFS_PROB_THRESHOLD"] = str(
        args.adaptive_dfs_prob_threshold
    )
    os.environ["ARC_ADAPTIVE_COLOR_PERMUTATIONS"] = str(
        args.adaptive_color_permutations
    )
    os.environ["ARC_ADAPTIVE_MIN_UNIQUE_CANDIDATES"] = str(
        args.adaptive_min_unique_candidates
    )
    os.environ["ARC_PROFILE_TIMINGS"] = "1" if args.profile_timings else "0"
    os.environ["ARC_TEST_PATH"] = args.test_path
    os.environ["ARC_MODEL_PATH"] = args.model_path
    os.environ["ARC_OUTPUT_DIR"] = args.output_dir
    os.environ["ARC_TTFT_METHOD"] = args.ttft_method
    if args.fixed_candidate_dir is not None:
        os.environ["ARC_FIXED_CANDIDATE_DIR"] = args.fixed_candidate_dir
    else:
        os.environ.pop("ARC_FIXED_CANDIDATE_DIR", None)
    if args.selected_augmentations_path is not None:
        os.environ["ARC_SELECTED_AUGMENTATIONS_PATH"] = args.selected_augmentations_path
    else:
        os.environ.pop("ARC_SELECTED_AUGMENTATIONS_PATH", None)
    os.environ["ARC_OPSD_LOG_DIR"] = args.opsd_log_dir
    os.environ["ARC_OPSD_MIN_TRAIN_PAIRS"] = str(args.opsd_min_train_pairs)
    os.environ["ARC_OPSD_COLOR_PERMUTATIONS"] = str(args.opsd_color_permutations)
    os.environ["ARC_OPSD_CROSS_VIEW_PROBABILITY"] = str(args.opsd_cross_view_probability)
    os.environ["ARC_OPSD_MAX_UPDATES"] = str(args.opsd_max_updates)
    os.environ["ARC_OPSD_LEARNING_RATE"] = str(args.opsd_learning_rate)
    os.environ["ARC_OPSD_TEMPERATURE"] = str(args.opsd_temperature)
    os.environ["ARC_OPSD_TOP_P"] = str(args.opsd_top_p)
    os.environ["ARC_OPSD_LAMBDA_CE"] = str(args.opsd_lambda_ce)
    os.environ["ARC_REPAIR_TTFT_LOG_DIR"] = args.repair_ttft_log_dir
    os.environ["ARC_REPAIR_TTFT_TOTAL_STEPS"] = str(args.repair_ttft_total_steps)
    os.environ["ARC_REPAIR_TTFT_LOO_STAGE1_STEPS"] = str(args.repair_ttft_loo_stage1_steps)
    os.environ["ARC_REPAIR_TTFT_WARM_STAGE1_STEPS"] = str(args.repair_ttft_warm_stage1_steps)
    os.environ["ARC_REPAIR_TTFT_STAGE2_REPAIR_FRACTION"] = str(
        args.repair_ttft_stage2_repair_fraction
    )
    os.environ["ARC_REPAIR_TTFT_LOO_HELDOUT_VIEWS"] = str(args.repair_ttft_loo_heldout_views)
    os.environ["ARC_REPAIR_TTFT_LOO_SEEN_VIEWS"] = str(args.repair_ttft_loo_seen_views)
    os.environ["ARC_REPAIR_TTFT_WARM_VIEWS_PER_PAIR"] = str(
        args.repair_ttft_warm_views_per_pair
    )
    os.environ["ARC_SCHEDULED_SAMPLING_WARMUP_STEPS"] = str(
        args.scheduled_sampling_warmup_steps
    )
    os.environ["ARC_SCHEDULED_SAMPLING_TOTAL_STEPS"] = str(
        args.scheduled_sampling_total_steps
    )
    os.environ["ARC_SCHEDULED_SAMPLING_MIX_PROBABILITY"] = str(
        args.scheduled_sampling_mix_probability
    )
    os.environ["ARC_SCHEDULED_SAMPLING_LOG_DIR"] = args.scheduled_sampling_log_dir
    if args.canon_ac_state is not None:
        os.environ["ARC_CANON_AC_STATE"] = args.canon_ac_state
    else:
        os.environ.pop("ARC_CANON_AC_STATE", None)
    print(
        "runtime flags:",
        f"speculative_dfs={os.environ['ARC_USE_SPECULATIVE_DFS']}",
        f"unsloth_multitoken_dfs={os.environ['ARC_USE_UNSLOTH_MULTITOKEN_DFS']}",
        f"unsloth_structured_rows={os.environ['ARC_USE_UNSLOTH_STRUCTURED_ROWS']}",
        f"unsloth_multitoken_repeat_len={os.environ['ARC_UNSLOTH_MULTITOKEN_REPEAT_LEN']}",
        f"use_sglang={os.environ['ARC_USE_SGLANG']}",
        f"sglang_tp_size={os.environ['ARC_SGLANG_TP_SIZE']}",
        f"sglang_mem_fraction_static={os.environ.get('ARC_SGLANG_MEM_FRACTION_STATIC')}",
        f"sglang_adapter_dir={os.environ['ARC_SGLANG_ADAPTER_DIR']}",
        f"sglang_adapter_manifest={os.environ['ARC_SGLANG_ADAPTER_MANIFEST']}",
        f"sglang_train_adapters_only={os.environ['ARC_SGLANG_TRAIN_ADAPTERS_ONLY']}",
        f"sglang_reuse_adapters={os.environ['ARC_SGLANG_REUSE_ADAPTERS']}",
        f"sglang_persistent_infer={os.environ['ARC_SGLANG_PERSISTENT_INFER']}",
        f"sglang_consume_adapters={os.environ['ARC_SGLANG_CONSUME_ADAPTERS']}",
        f"sglang_speculative_repeat_len={os.environ['ARC_SGLANG_SPECULATIVE_REPEAT_LEN']}",
        f"sglang_dynamic_repeat={os.environ['ARC_SGLANG_DYNAMIC_REPEAT']}",
        f"dfs_prob_threshold={os.environ['ARC_DFS_PROB_THRESHOLD']}",
        f"eval_color_permutations={os.environ['ARC_EVAL_COLOR_PERMUTATIONS']}",
        f"profile_timings={os.environ['ARC_PROFILE_TIMINGS']}",
        f"test_path={os.environ['ARC_TEST_PATH']}",
        f"model_path={os.environ['ARC_MODEL_PATH']}",
        f"output_dir={os.environ['ARC_OUTPUT_DIR']}",
        f"ttft_method={os.environ['ARC_TTFT_METHOD']}",
        f"fixed_candidate_dir={os.environ.get('ARC_FIXED_CANDIDATE_DIR')}",
        f"selected_augmentations_path={os.environ.get('ARC_SELECTED_AUGMENTATIONS_PATH')}",
        f"opsd_log_dir={os.environ['ARC_OPSD_LOG_DIR']}",
        f"opsd_min_train_pairs={os.environ['ARC_OPSD_MIN_TRAIN_PAIRS']}",
        f"opsd_color_permutations={os.environ['ARC_OPSD_COLOR_PERMUTATIONS']}",
        f"opsd_cross_view_probability={os.environ['ARC_OPSD_CROSS_VIEW_PROBABILITY']}",
        f"opsd_max_updates={os.environ['ARC_OPSD_MAX_UPDATES']}",
        f"opsd_learning_rate={os.environ['ARC_OPSD_LEARNING_RATE']}",
        f"opsd_temperature={os.environ['ARC_OPSD_TEMPERATURE']}",
        f"opsd_top_p={os.environ['ARC_OPSD_TOP_P']}",
        f"opsd_lambda_ce={os.environ['ARC_OPSD_LAMBDA_CE']}",
        f"scheduled_sampling_warmup_steps={os.environ['ARC_SCHEDULED_SAMPLING_WARMUP_STEPS']}",
        f"scheduled_sampling_total_steps={os.environ['ARC_SCHEDULED_SAMPLING_TOTAL_STEPS']}",
        f"scheduled_sampling_mix_probability={os.environ['ARC_SCHEDULED_SAMPLING_MIX_PROBABILITY']}",
        f"scheduled_sampling_log_dir={os.environ['ARC_SCHEDULED_SAMPLING_LOG_DIR']}",
        f"canon_ac_state={os.environ.get('ARC_CANON_AC_STATE')}",
    )

    if args.use_unsloth_multitoken_dfs:
        import importlib.util
        from pathlib import Path

        from patch_unsloth_qwen3_multitoken import patch_unsloth

        spec = importlib.util.find_spec("unsloth")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Cannot locate the Unsloth package for multi-token patching")
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        changed = patch_unsloth(package_dir)
        print("Unsloth multi-token patch ready:", package_dir, "changed=", [str(path) for path in changed])

    rerun_mode = True
    manager = mp.Manager()
    queue = manager.Queue()
    if args.sglang_stream_manifest:
        jobs = []
    elif args.sglang_infer_from_manifest:
        jobs = _load_manifest_jobs(args)
        for job in jobs:
            queue.put(job)
    else:
        with open(args.test_path, "r") as f:
            data = json.load(f)
        selected_keys = _load_selected_keys(args, data)
        if args.cheap_first:
            selected_keys.sort(key=lambda key: (_estimated_task_work(data[key]), key))
            print("cheap-first task order:", selected_keys, flush=True)
        for key in selected_keys:
            assert key in data, f"Unknown puzzle key: {key}"
            queue.put(key)
    sentinel_prefix = f"worker_{'infer' if (args.sglang_infer_from_manifest or args.sglang_stream_manifest) else 'train'}_"
    if args.sglang_stream_manifest:
        context = mp.spawn(
            local_worker,
            args=(queue, end_time, args.cuda_device_offset, sentinel_prefix),
            nprocs=effective_nprocs,
            join=False,
        )
        queued_keys = set()
        while time.time() <= end_time:
            new_jobs = _load_streaming_manifest_jobs(args, queued_keys)
            for job in new_jobs:
                queue.put(job)
                queued_keys.add(job["key"])
                print(f"[stream] queued ready adapter for {job['key']}", flush=True)
            if os.path.exists(args.sglang_producer_done) and not new_jobs:
                break
            time.sleep(1)
        for _ in range(effective_nprocs):
            queue.put(None)
        while not context.join():
            pass
        print(f"[stream] inference workers finished; queued_keys={len(queued_keys)}", flush=True)
    else:
        for _ in range(effective_nprocs):
            queue.put(None)
        mp.spawn(
            local_worker,
            args=(queue, end_time, args.cuda_device_offset, sentinel_prefix),
            nprocs=effective_nprocs,
        )
