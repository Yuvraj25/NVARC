import bz2
import fcntl
import gc
import hashlib
import io
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Union

import numpy as np
import torch
from datasets import Dataset
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

from arc_loader import ArcDataset, QwenFormatter
from arc_rescoring import FullPassRescorer
from arc_sglang import ArcSglangBackend, SglangConfig, SglangRescorer, inference_sglang_dfs, inference_sglang_speculative_dfs
from arc_search import ASSISTANT_TOKEN_ID, EOS_ID, USER_TOKEN_ID, default_max_score, inference_turbo_dfs

logging.disable(logging.WARNING)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def runtime_config():
    dfs_prob_threshold = float(os.environ.get("ARC_DFS_PROB_THRESHOLD", "0.2"))
    if not 0.0 < dfs_prob_threshold < 1.0:
        raise ValueError(f"ARC_DFS_PROB_THRESHOLD must be in (0, 1), got {dfs_prob_threshold}")
    sglang_mem_fraction = os.environ.get("ARC_SGLANG_MEM_FRACTION_STATIC")
    return {
        "use_speculative_dfs": _env_flag("ARC_USE_SPECULATIVE_DFS", default=False),
        "use_sglang": _env_flag("ARC_USE_SGLANG", default=False),
        "profile_timings": _env_flag("ARC_PROFILE_TIMINGS", default=False),
        "dfs_prob_threshold": dfs_prob_threshold,
        "model_path": os.environ.get("ARC_MODEL_PATH", "../input/qwen3_4b_grids15_sft139/"),
        "test_path": os.environ.get("ARC_TEST_PATH", "../input/arc-prize-2024/arc-agi_evaluation_challenges.json"),
        "output_dir": os.environ.get("ARC_OUTPUT_DIR", "../inference_outputs"),
        "sglang_tensor_parallel_size": int(os.environ.get("ARC_SGLANG_TP_SIZE", "1")),
        "sglang_mem_fraction_static": float(sglang_mem_fraction) if sglang_mem_fraction else None,
        "sglang_adapter_dir": os.environ.get("ARC_SGLANG_ADAPTER_DIR", "../sglang_adapters"),
        "sglang_adapter_manifest": os.environ.get("ARC_SGLANG_ADAPTER_MANIFEST"),
        "sglang_train_adapters_only": _env_flag("ARC_SGLANG_TRAIN_ADAPTERS_ONLY", default=False),
        "sglang_reuse_adapters": _env_flag("ARC_SGLANG_REUSE_ADAPTERS", default=False),
        "sglang_persistent_infer": _env_flag("ARC_SGLANG_PERSISTENT_INFER", default=False),
        "sglang_speculative_repeat_len": int(os.environ.get("ARC_SGLANG_SPECULATIVE_REPEAT_LEN", "5")),
        "sglang_dynamic_repeat": _env_flag("ARC_SGLANG_DYNAMIC_REPEAT", default=False),
    }


class UnslothFixedTrainer(UnslothTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


def stable_seed_from_key(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (1024**2)


def _safe_path_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _sglang_adapter_path(config, key: str) -> str:
    return os.path.join(config["sglang_adapter_dir"], _safe_path_key(key))


def _sglang_lora_name(key: str) -> str:
    return f"arc_{_safe_path_key(key)}"


def _default_sglang_manifest_path(config) -> str:
    return config["sglang_adapter_manifest"] or os.path.join(config["sglang_adapter_dir"], "adapter_manifest.json")


def _manifest_skeleton():
    return {"version": 1, "entries": []}


def _load_manifest_file(manifest_path: str):
    if not os.path.exists(manifest_path):
        return _manifest_skeleton()
    with open(manifest_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "entries" not in data or not isinstance(data["entries"], list):
        raise ValueError(f"Invalid adapter manifest at {manifest_path}")
    return data


def _upsert_manifest_entry(manifest_path: str, entry: dict[str, Any]):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    lock_path = f"{manifest_path}.lock"
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        data = _load_manifest_file(manifest_path)
        entries = data.get("entries", [])
        filtered = [existing for existing in entries if existing.get("key") != entry["key"]]
        filtered.append(entry)
        filtered.sort(key=lambda item: item.get("key", ""))
        data["entries"] = filtered
        with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(manifest_path), prefix=".manifest.", suffix=".tmp") as tmp_file:
            json.dump(data, tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
            tmp_path = tmp_file.name
        os.replace(tmp_path, manifest_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _path_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            total += os.path.getsize(os.path.join(root, file_name))
    return total


def _build_eval_batches(eval_ds):
    test_id_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        test_id_to_subkeys[test_id].append(subkey)

    batches = []
    for _test_id, subkeys in test_id_to_subkeys.items():
        batch = []
        for offset in [0, 4]:
            batch.extend(subkeys[offset : offset + 2])
        batches.append(batch)

        batch = []
        for offset in [2, 6]:
            batch.extend(subkeys[offset : offset + 2])
        batches.append(batch)

    for _test_id, subkeys in test_id_to_subkeys.items():
        batch = []
        for offset in [8, 12]:
            batch.extend(subkeys[offset : offset + 2])
        batches.append(batch)

        batch = []
        for offset in [10, 14]:
            batch.extend(subkeys[offset : offset + 2])
        batches.append(batch)
    return batches


def _prepare_eval_ds(puzzle_ds, formatter, max_seq_length: int, max_new_tokens: int):
    puzzle_ds_multi = puzzle_ds.split_multi_replies()
    eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)
    return puzzle_ds_multi, eval_ds


def _run_sglang_batches(
    rank,
    backend,
    tokenizer,
    formatter,
    puzzle_ds_multi,
    eval_ds,
    dir_outputs,
    max_seq_length,
    max_new_tokens,
    max_score,
    use_speculative_dfs,
    start_time,
    end_time,
    timing_stats,
    count_stats,
):
    batches = _build_eval_batches(eval_ds)
    known_scores = {}
    rescorers = {}

    for subkeys in batches:
        spend_time = time.time() - start_time
        if spend_time > 1200 or time.time() > end_time:
            print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle batch {subkeys[0].split('.')[0]}")
            break

        print(f"[Rank {rank}] decoding {subkeys}")
        count_stats["batches"] += 1

        tokenize_started_at = time.perf_counter()
        tokens = []
        for subkey in subkeys:
            data = eval_ds.get(subkey, formatter)
            tokens.append(tokenizer.encode(data["input"]))
        timing_stats["tokenize_inputs_s"] += time.perf_counter() - tokenize_started_at

        dfs_started_at = time.perf_counter()
        if use_speculative_dfs:
            dfs_result = inference_sglang_speculative_dfs(
                backend,
                tokens,
                max_new_tokens,
                max_score,
                end_time,
                count_stats=count_stats,
            )
        else:
            dfs_result = inference_sglang_dfs(
                backend,
                tokens,
                max_new_tokens,
                max_score,
                end_time,
                count_stats=count_stats,
            )
        timing_stats["dfs_s"] += time.perf_counter() - dfs_started_at
        count_stats["dfs_calls"] += 1

        for subkey_id, scored_beams in dfs_result:
            subkey = subkeys[subkey_id]
            bk = subkey.split(".")[0]
            decoded_result = []
            count_stats["subkeys_scored"] += 1

            if bk not in rescorers:
                rescorer_started_at = time.perf_counter()
                rescorers[bk] = SglangRescorer(
                    model=None,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    puzzle_ds_multi=puzzle_ds_multi,
                    base_key=bk,
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    seed=stable_seed_from_key(bk),
                    backend=backend,
                )
                timing_stats["rescorer_init_s"] += time.perf_counter() - rescorer_started_at
                count_stats["rescorers_created"] += 1

            for beam_score, beam_tokens in scored_beams:
                count_stats["beam_candidates_seen"] += 1
                array = formatter.convert_tokens_to_array(beam_tokens)
                if array is None:
                    count_stats["beam_candidates_invalid"] += 1
                    continue

                solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                grid_id = (bk, tuple(map(tuple, solution)))
                count_stats["beam_candidates_valid"] += 1

                if grid_id in known_scores:
                    augmented_scores = known_scores[grid_id]
                    count_stats["rescoring_cache_hits"] += 1
                else:
                    print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                    rescore_started_at = time.perf_counter()
                    augmented_scores = rescorers[bk].score_solution(solution)
                    timing_stats["rescoring_s"] += time.perf_counter() - rescore_started_at
                    known_scores[grid_id] = augmented_scores
                    count_stats["rescoring_cache_misses"] += 1

                decoded_result.append(
                    {
                        "beam_score": beam_score,
                        "score_aug": augmented_scores,
                        "solution": solution,
                    }
                )

            if len(decoded_result):
                write_started_at = time.perf_counter()
                with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                    pickle.dump(decoded_result, f)
                timing_stats["write_results_s"] += time.perf_counter() - write_started_at
                count_stats["subkeys_written"] += 1

    return rescorers


def _print_sglang_profile(rank, key, timing_stats, count_stats, backend, rescorers):
    backend_next_arc_s = backend.stats["next_arc_logprobs_time_s"]
    backend_next_arc_calls = int(backend.stats["next_arc_logprobs_calls"])
    backend_next_arc_prompts = int(backend.stats["next_arc_logprobs_prompts"])
    backend_next_arc_prompt_tokens = int(backend.stats["next_arc_logprobs_prompt_tokens"])
    backend_next_arc_max_batch = int(backend.stats["next_arc_logprobs_max_batch"])
    backend_draft_arc_s = backend.stats["draft_arc_logprobs_time_s"]
    backend_draft_arc_calls = int(backend.stats["draft_arc_logprobs_calls"])
    backend_draft_arc_prompts = int(backend.stats["draft_arc_logprobs_prompts"])
    backend_draft_arc_prompt_tokens = int(backend.stats["draft_arc_logprobs_prompt_tokens"])
    backend_draft_arc_tokens = int(backend.stats["draft_arc_logprobs_draft_tokens"])
    backend_draft_arc_max_batch = int(backend.stats["draft_arc_logprobs_max_batch"])
    backend_total_calls = backend_next_arc_calls + backend_draft_arc_calls
    backend_total_positions = backend_next_arc_prompts + backend_draft_arc_tokens
    backend_total_time_s = backend_next_arc_s + backend_draft_arc_s
    backend_max_batch = max(backend_next_arc_max_batch, backend_draft_arc_max_batch)
    backend_avg_positions_per_call = backend_total_positions / backend_total_calls if backend_total_calls else 0.0
    timing_stats["dfs_backend_next_arc_s"] = backend_total_time_s
    timing_stats["dfs_python_overhead_s"] = max(timing_stats["dfs_s"] - backend_total_time_s, 0.0)
    ordered_timings = [
        "engine_init_s",
        "training_s",
        "adapter_load_s",
        "eval_prep_s",
        "tokenize_inputs_s",
        "dfs_s",
        "dfs_backend_next_arc_s",
        "dfs_python_overhead_s",
        "rescorer_init_s",
        "rescoring_s",
        "write_results_s",
        "adapter_unload_s",
        "total_wall_s",
    ]
    timings_text = " ".join(f"{name}={timing_stats[name]:.3f}s" for name in ordered_timings)
    counts_text = " ".join(
        f"{name}={count_stats[name]}"
        for name in [
            "batches",
            "dfs_calls",
            "dfs_frames_expanded",
            "subkeys_scored",
            "subkeys_written",
            "beam_candidates_seen",
            "beam_candidates_valid",
            "beam_candidates_invalid",
            "rescoring_cache_hits",
            "rescoring_cache_misses",
            "rescorers_created",
            "spec_branches_started",
            "spec_branches_zero_extra",
            "spec_branches_one_extra",
            "spec_branches_two_extra",
            "spec_branches_three_extra",
            "spec_branches_four_extra",
            "spec_branches_five_plus_extra",
            "spec_extra_appends_total",
            "spec_side_frames_enqueued",
            "spec_stop_repeat_invalid",
            "spec_stop_threshold",
            "spec_stop_eos",
            "spec_stop_remaining_exhausted",
            "spec_stop_depth_limit",
            "spec_stop_end_time",
            "spec_dynamic_len_1",
            "spec_dynamic_len_5",
            "spec_dynamic_len_9",
            "spec_dynamic_len_max",
        ]
    )
    print(f"[Rank {rank}] timing summary for {key}: {timings_text}")
    print(f"[Rank {rank}] count summary for {key}: {counts_text}")
    print(
        f"[Rank {rank}] dfs backend summary for {key}: "
        f"next_arc_calls={backend_next_arc_calls} "
        f"draft_arc_calls={backend_draft_arc_calls} "
        f"calls_total={backend_total_calls} "
        f"prompts_evaluated={backend_total_positions} "
        f"next_prompt_tokens={backend_next_arc_prompt_tokens} "
        f"draft_prompt_tokens={backend_draft_arc_prompt_tokens} "
        f"draft_tokens_verified={backend_draft_arc_tokens} "
        f"avg_positions_per_call={backend_avg_positions_per_call:.2f} "
        f"max_batch_size={backend_max_batch}"
    )
    for base_key, rescorer in sorted(rescorers.items()):
        print(f"[Rank {rank}] rescorer summary for {base_key}: {rescorer.format_stats()}")


def worker_sglang(rank, queue, end_time, config):
    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192
    max_score = default_max_score(config["dfs_prob_threshold"])
    print(
        f"[Rank {rank}] config: use_sglang=True tp_size={config['sglang_tensor_parallel_size']} "
        f"mem_fraction_static={config['sglang_mem_fraction_static']} dfs_prob_threshold={config['dfs_prob_threshold']} "
        f"speculative_dfs={config['use_speculative_dfs']} sglang_speculative_repeat_len={config['sglang_speculative_repeat_len']} "
        f"dynamic_repeat={config['sglang_dynamic_repeat']} "
        f"train_adapters_only={config['sglang_train_adapters_only']} reuse_adapters={config['sglang_reuse_adapters']} "
        f"persistent_infer={config['sglang_persistent_infer']}"
    )

    arc_test_set = ArcDataset.from_file(config["test_path"])
    dir_outputs = config["output_dir"]
    os.makedirs(dir_outputs, exist_ok=True)
    os.makedirs(config["sglang_adapter_dir"], exist_ok=True)
    persistent_backend = None
    persistent_tokenizer = None
    persistent_formatter = None
    persistent_max_new_tokens = None
    if config["sglang_persistent_infer"]:
        persistent_tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            trust_remote_code=True,
            local_files_only=True,
        )
        persistent_formatter = QwenFormatter(tokenizer=persistent_tokenizer)
        persistent_max_new_tokens = persistent_formatter.max_new_tokens()
        persistent_backend = ArcSglangBackend(
            SglangConfig(
                model_path=config["model_path"],
                adapter_path=None,
                tensor_parallel_size=config["sglang_tensor_parallel_size"],
                mem_fraction_static=config["sglang_mem_fraction_static"],
                max_model_len=max_seq_length,
                max_loaded_loras=1,
                speculative_repeat_len=config["sglang_speculative_repeat_len"],
                dynamic_repeat_enabled=config["sglang_dynamic_repeat"],
            )
        )
        print(f"[Rank {rank}] persistent SGLang engine ready: engine_init_s={persistent_backend.engine_init_s:.3f}")

    try:
        while not queue.empty():
            if time.time() > end_time:
                print(f"[Rank {rank}] stop!")
                break

            job = queue.get()
            if job is None:
                break

            if isinstance(job, dict):
                key = job["key"]
                adapter_path = job["adapter_path"]
            else:
                key = job
                adapter_path = _sglang_adapter_path(config, key)

            start_time = time.time()
            puzzle_started_at = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            timing_stats = defaultdict(float)
            count_stats = defaultdict(int)
            timing_stats["engine_init_s"] = 0.0 if config["sglang_persistent_infer"] else 0.0
            puzzle_ds = arc_test_set.change_keys([key])
            model = None
            collator = None
            backend = persistent_backend
            tokenizer = persistent_tokenizer
            formatter = persistent_formatter
            max_new_tokens = persistent_max_new_tokens
            keep_adapter_dir = False

            try:
                if config["sglang_persistent_infer"]:
                    if not os.path.isdir(adapter_path):
                        raise FileNotFoundError(f"Saved adapter not found for {key}: {adapter_path}")
                    load_started_at = time.perf_counter()
                    backend.load_adapter(_sglang_lora_name(key), adapter_path)
                    timing_stats["adapter_load_s"] += time.perf_counter() - load_started_at
                    backend.reset_stats()
                    prep_started_at = time.perf_counter()
                    puzzle_ds_multi, eval_ds = _prepare_eval_ds(puzzle_ds, formatter, max_seq_length, max_new_tokens)
                    timing_stats["eval_prep_s"] += time.perf_counter() - prep_started_at
                    print(f"[Rank {rank}] persistent-infer adapter loaded for puzzle {key}: {adapter_path}")
                elif config["sglang_reuse_adapters"]:
                    if not os.path.isdir(adapter_path):
                        raise FileNotFoundError(f"Saved adapter not found for {key}: {adapter_path}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        config["model_path"],
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    formatter = QwenFormatter(tokenizer=tokenizer)
                    max_new_tokens = formatter.max_new_tokens()
                    prep_started_at = time.perf_counter()
                    puzzle_ds_multi, eval_ds = _prepare_eval_ds(puzzle_ds, formatter, max_seq_length, max_new_tokens)
                    timing_stats["eval_prep_s"] += time.perf_counter() - prep_started_at
                    print(f"[Rank {rank}] reusing adapter for puzzle {key}: {adapter_path}")
                    backend = ArcSglangBackend(
                        SglangConfig(
                            model_path=config["model_path"],
                            adapter_path=adapter_path,
                            tensor_parallel_size=config["sglang_tensor_parallel_size"],
                            mem_fraction_static=config["sglang_mem_fraction_static"],
                            max_model_len=max_seq_length,
                            speculative_repeat_len=config["sglang_speculative_repeat_len"],
                            dynamic_repeat_enabled=config["sglang_dynamic_repeat"],
                        )
                    )
                    timing_stats["engine_init_s"] = backend.engine_init_s
                else:
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=config["model_path"],
                        full_finetuning=False,
                        load_in_4bit=False,
                        local_files_only=True,
                        use_gradient_checkpointing=False,
                        max_seq_length=max_seq_length,
                    )
                    model = FastLanguageModel.get_peft_model(model, **peft_params)
                    for _name, param in model.named_parameters():
                        if param.dtype == torch.float32:
                            param.data = param.data.to(torch.bfloat16)

                    collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
                    formatter = QwenFormatter(tokenizer=tokenizer)
                    max_new_tokens = formatter.max_new_tokens()
                    model = FastLanguageModel.for_training(model)
                    train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
                    train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

                    training_started_at = time.perf_counter()
                    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                        trainer = UnslothFixedTrainer(
                            model=model,
                            tokenizer=tokenizer,
                            data_collator=collator,
                            train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                            dataset_text_field="text",
                            max_seq_length=max_seq_length,
                            args=UnslothTrainingArguments(**train_args),
                        )
                        stats = trainer.train()
                        model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                        del trainer
                    timing_stats["training_s"] += time.perf_counter() - training_started_at

                    prep_started_at = time.perf_counter()
                    model.save_pretrained(adapter_path)
                    keep_adapter_dir = config["sglang_train_adapters_only"] or os.environ.get("ARC_KEEP_SGLANG_ADAPTERS") == "1"
                    memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
                    print(f"[Rank {rank}] allocated {memory_allocated}MB for training")
                    torch.cuda.reset_peak_memory_stats()
                    print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

                    puzzle_ds_multi, eval_ds = _prepare_eval_ds(puzzle_ds, formatter, max_seq_length, max_new_tokens)
                    timing_stats["eval_prep_s"] += time.perf_counter() - prep_started_at

                    del model
                    del collator
                    model = None
                    collator = None
                    gc.collect()
                    torch.cuda.empty_cache()

                    if config["sglang_train_adapters_only"]:
                        manifest_path = _default_sglang_manifest_path(config)
                        _upsert_manifest_entry(
                            manifest_path,
                            {
                                "key": key,
                                "adapter_path": adapter_path,
                                "size_bytes": _path_size_bytes(adapter_path),
                                "status": "ready",
                                "updated_at": time.time(),
                            },
                        )
                        spend_time = time.time() - start_time
                        print(f"[Rank {rank}] saved adapter for puzzle {key} to {adapter_path}")
                        print(f"[Rank {rank}] adapter manifest updated: {manifest_path}")
                        print(f"[Rank {rank}] finished adapter-only pass for {key} in {spend_time:.1f}s")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    backend = ArcSglangBackend(
                        SglangConfig(
                            model_path=config["model_path"],
                            adapter_path=adapter_path,
                            tensor_parallel_size=config["sglang_tensor_parallel_size"],
                            mem_fraction_static=config["sglang_mem_fraction_static"],
                            max_model_len=max_seq_length,
                            speculative_repeat_len=config["sglang_speculative_repeat_len"],
                            dynamic_repeat_enabled=config["sglang_dynamic_repeat"],
                        )
                    )
                    timing_stats["engine_init_s"] = backend.engine_init_s

                rescorers = _run_sglang_batches(
                    rank=rank,
                    backend=backend,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    puzzle_ds_multi=puzzle_ds_multi,
                    eval_ds=eval_ds,
                    dir_outputs=dir_outputs,
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    max_score=max_score,
                    use_speculative_dfs=config["use_speculative_dfs"],
                    start_time=start_time,
                    end_time=end_time,
                    timing_stats=timing_stats,
                    count_stats=count_stats,
                )
                memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
                print(f"[Rank {rank}] allocated {memory_allocated}MB for sglang inference")
            finally:
                if config["sglang_persistent_infer"] and backend is not None:
                    unload_started_at = time.perf_counter()
                    backend.unload_adapter()
                    timing_stats["adapter_unload_s"] += time.perf_counter() - unload_started_at
                elif backend is not None:
                    backend.close()
                if (
                    not config["sglang_persistent_infer"]
                    and not config["sglang_reuse_adapters"]
                    and not keep_adapter_dir
                ):
                    shutil.rmtree(adapter_path, ignore_errors=True)
                if not config["sglang_persistent_infer"]:
                    if tokenizer is not None:
                        del tokenizer
                    if formatter is not None:
                        del formatter
                    gc.collect()
                    torch.cuda.empty_cache()

            spend_time = time.time() - start_time
            timing_stats["total_wall_s"] = time.perf_counter() - puzzle_started_at
            print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
            if config["profile_timings"]:
                _print_sglang_profile(rank, key, timing_stats, count_stats, backend, rescorers)
    finally:
        if persistent_backend is not None:
            persistent_backend.close()
        if persistent_tokenizer is not None:
            del persistent_tokenizer
        if persistent_formatter is not None:
            del persistent_formatter
        gc.collect()
        torch.cuda.empty_cache()


def worker(rank, queue, end_time):
    config = runtime_config()
    if config["use_sglang"]:
        return worker_sglang(rank, queue, end_time, config)

    if config["use_speculative_dfs"]:
        raise NotImplementedError("Speculative DFS is feature-flagged but not implemented yet")

    rerun_mode = True

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_path"],
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)
    for _name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    max_score = default_max_score(config["dfs_prob_threshold"])
    print(
        f"[Rank {rank}] config: speculative_dfs={config['use_speculative_dfs']} "
        f"dfs_prob_threshold={config['dfs_prob_threshold']}"
    )

    arc_test_set = ArcDataset.from_file(config["test_path"])
    dir_outputs = config["output_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    while not queue.empty():
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()
        puzzle_started_at = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        timing_stats = defaultdict(float)
        count_stats = defaultdict(int)

        set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)
        puzzle_ds = arc_test_set.change_keys([key])
        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        training_started_at = time.perf_counter()
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args),
            )

            stats = trainer.train()
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            del trainer
        timing_stats["training_s"] += time.perf_counter() - training_started_at

        prep_started_at = time.perf_counter()
        model = FastLanguageModel.for_inference(model)
        gc.collect()
        torch.cuda.empty_cache()

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")
        torch.cuda.reset_peak_memory_stats()
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)
        timing_stats["eval_prep_s"] += time.perf_counter() - prep_started_at

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset : offset + 2])
            batches.append(batch)

            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset : offset + 2])
            batches.append(batch)

        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset : offset + 2])
            batches.append(batch)

            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset : offset + 2])
            batches.append(batch)

        with torch.inference_mode():
            known_scores = {}
            rescorers = {}

            for subkeys in batches:
                spend_time = time.time() - start_time
                if spend_time > 1200 or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                    break

                print(f"[Rank {rank}] decoding {subkeys}")
                count_stats["batches"] += 1

                tokenize_started_at = time.perf_counter()
                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))
                timing_stats["tokenize_inputs_s"] += time.perf_counter() - tokenize_started_at

                dfs_started_at = time.perf_counter()
                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)
                timing_stats["dfs_s"] += time.perf_counter() - dfs_started_at
                count_stats["dfs_calls"] += 1

                for subkey_id, scored_beams in dfs_result:
                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []
                    count_stats["subkeys_scored"] += 1

                    if bk not in rescorers:
                        rescorer_started_at = time.perf_counter()
                        rescorers[bk] = FullPassRescorer(
                            model=model,
                            tokenizer=tokenizer,
                            formatter=formatter,
                            puzzle_ds_multi=puzzle_ds_multi,
                            base_key=bk,
                            max_seq_length=max_seq_length,
                            max_new_tokens=max_new_tokens,
                            seed=stable_seed_from_key(bk),
                        )
                        timing_stats["rescorer_init_s"] += time.perf_counter() - rescorer_started_at
                        count_stats["rescorers_created"] += 1

                    for beam_score, beam_tokens in scored_beams:
                        count_stats["beam_candidates_seen"] += 1
                        array = formatter.convert_tokens_to_array(beam_tokens)
                        if array is None:
                            count_stats["beam_candidates_invalid"] += 1
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        grid_id = (bk, tuple(map(tuple, solution)))
                        count_stats["beam_candidates_valid"] += 1

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                            count_stats["rescoring_cache_hits"] += 1
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            rescore_started_at = time.perf_counter()
                            augmented_scores = rescorers[bk].score_solution(solution)
                            timing_stats["rescoring_s"] += time.perf_counter() - rescore_started_at
                            known_scores[grid_id] = augmented_scores
                            count_stats["rescoring_cache_misses"] += 1

                        decoded_result.append(
                            {
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            }
                        )

                    if len(decoded_result):
                        write_started_at = time.perf_counter()
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)
                        timing_stats["write_results_s"] += time.perf_counter() - write_started_at
                        count_stats["subkeys_written"] += 1

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
        if config["profile_timings"]:
            timing_stats["total_wall_s"] = time.perf_counter() - puzzle_started_at
            ordered_timings = [
                "training_s",
                "eval_prep_s",
                "tokenize_inputs_s",
                "dfs_s",
                "rescorer_init_s",
                "rescoring_s",
                "write_results_s",
                "total_wall_s",
            ]
            timings_text = " ".join(f"{name}={timing_stats[name]:.3f}s" for name in ordered_timings)
            counts_text = " ".join(
                f"{name}={count_stats[name]}"
                for name in [
                    "batches",
                    "dfs_calls",
                    "subkeys_scored",
                    "subkeys_written",
                    "beam_candidates_seen",
                    "beam_candidates_valid",
                    "beam_candidates_invalid",
                    "rescoring_cache_hits",
                    "rescoring_cache_misses",
                    "rescorers_created",
                ]
            )
            print(f"[Rank {rank}] timing summary for {key}: {timings_text}")
            print(f"[Rank {rank}] count summary for {key}: {counts_text}")
            for base_key, rescorer in sorted(rescorers.items()):
                print(f"[Rank {rank}] rescorer summary for {base_key}: {rescorer.format_stats()}")
