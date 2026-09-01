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

from arc_loader import ArcDataset, QwenFormatter
from arc_opsd import (
    TEACHER_ADAPTER_NAME,
    activate_adapter,
    build_opsd_examples,
    clone_frozen_teacher_adapter,
    remove_teacher_adapter,
    run_opsd_correction,
    split_puzzle_for_opsd,
)
from arc_rescoring import FullPassRescorer
from arc_repair_ttft import (
    REPAIR_TTFT_METHODS,
    build_stage_two_mixture,
    deterministic_rows,
    mine_repair_examples,
    mixed_completion_collator,
    tokenize_ordinary_text,
)
from arc_selected_augmentations import load_selected_augmentations, prepare_selected_eval_ds
from arc_sglang import ArcSglangBackend, SglangConfig, SglangRescorer, inference_sglang_dfs, inference_sglang_speculative_dfs
from arc_search import ASSISTANT_TOKEN_ID, EOS_ID, USER_TOKEN_ID, default_max_score, inference_turbo_dfs
from arc_search_multitoken import (
    inference_turbo_dfs_multitoken,
    resume_turbo_dfs_multitoken,
)
from arc_scheduled_sampling import make_one_pass_scheduled_sampling_trainer_class

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
    ttft_method = os.environ.get("ARC_TTFT_METHOD", "full_sft")
    valid_ttft_methods = {
        "full_sft",
        "one_pass_ss",
        "reduced_sft",
        "reduced_plus_sft_c",
        "reduced_plus_opsd",
        *REPAIR_TTFT_METHODS,
    }
    if ttft_method not in valid_ttft_methods:
        raise ValueError(f"ARC_TTFT_METHOD must be one of {sorted(valid_ttft_methods)}, got {ttft_method!r}")
    cross_view_probability = float(os.environ.get("ARC_OPSD_CROSS_VIEW_PROBABILITY", "0.2"))
    if not 0.0 <= cross_view_probability <= 1.0:
        raise ValueError(
            f"ARC_OPSD_CROSS_VIEW_PROBABILITY must be in [0, 1], got {cross_view_probability}"
        )
    eval_color_permutations = int(os.environ.get("ARC_EVAL_COLOR_PERMUTATIONS", "2"))
    if eval_color_permutations < 1:
        raise ValueError(
            "ARC_EVAL_COLOR_PERMUTATIONS must be positive, "
            f"got {eval_color_permutations}"
        )
    adaptive_threshold = float(os.environ.get("ARC_ADAPTIVE_DFS_PROB_THRESHOLD", "0.1"))
    if not 0.0 < adaptive_threshold < 1.0:
        raise ValueError(
            "ARC_ADAPTIVE_DFS_PROB_THRESHOLD must be in (0, 1), "
            f"got {adaptive_threshold}"
        )
    return {
        "use_speculative_dfs": _env_flag("ARC_USE_SPECULATIVE_DFS", default=False),
        "use_unsloth_multitoken_dfs": _env_flag("ARC_USE_UNSLOTH_MULTITOKEN_DFS", default=False),
        "use_unsloth_structured_rows": _env_flag("ARC_USE_UNSLOTH_STRUCTURED_ROWS", default=False),
        "unsloth_multitoken_repeat_len": int(os.environ.get("ARC_UNSLOTH_MULTITOKEN_REPEAT_LEN", "9")),
        "use_sglang": _env_flag("ARC_USE_SGLANG", default=False),
        "profile_timings": _env_flag("ARC_PROFILE_TIMINGS", default=False),
        "dfs_prob_threshold": dfs_prob_threshold,
        "eval_color_permutations": eval_color_permutations,
        "shared_eval_augmentations": _env_flag(
            "ARC_SHARED_EVAL_AUGMENTATIONS", default=False
        ),
        "adaptive_output_dir": os.environ.get("ARC_ADAPTIVE_OUTPUT_DIR"),
        "adaptive_resume_frontier": _env_flag(
            "ARC_ADAPTIVE_RESUME_FRONTIER", default=False
        ),
        "adaptive_dfs_prob_threshold": adaptive_threshold,
        "adaptive_color_permutations": int(
            os.environ.get("ARC_ADAPTIVE_COLOR_PERMUTATIONS", "3")
        ),
        "adaptive_min_unique_candidates": int(
            os.environ.get("ARC_ADAPTIVE_MIN_UNIQUE_CANDIDATES", "2")
        ),
        "compare_structured_output_dir": os.environ.get(
            "ARC_COMPARE_STRUCTURED_OUTPUT_DIR"
        ),
        "compare_fresh_adaptive_output_dir": os.environ.get(
            "ARC_COMPARE_FRESH_ADAPTIVE_OUTPUT_DIR"
        ),
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
        "sglang_consume_adapters": _env_flag("ARC_SGLANG_CONSUME_ADAPTERS", default=False),
        "sglang_speculative_repeat_len": int(os.environ.get("ARC_SGLANG_SPECULATIVE_REPEAT_LEN", "5")),
        "sglang_dynamic_repeat": _env_flag("ARC_SGLANG_DYNAMIC_REPEAT", default=False),
        "ttft_method": ttft_method,
        "opsd_min_train_pairs": int(os.environ.get("ARC_OPSD_MIN_TRAIN_PAIRS", "3")),
        "opsd_color_permutations": int(os.environ.get("ARC_OPSD_COLOR_PERMUTATIONS", "2")),
        "opsd_cross_view_probability": cross_view_probability,
        "opsd_max_updates": int(os.environ.get("ARC_OPSD_MAX_UPDATES", "16")),
        "opsd_learning_rate": float(os.environ.get("ARC_OPSD_LEARNING_RATE", "5e-5")),
        "opsd_temperature": float(os.environ.get("ARC_OPSD_TEMPERATURE", "1.0")),
        "opsd_top_p": float(os.environ.get("ARC_OPSD_TOP_P", "1.0")),
        "opsd_lambda_ce": float(os.environ.get("ARC_OPSD_LAMBDA_CE", "0.0")),
        "opsd_log_dir": os.environ.get("ARC_OPSD_LOG_DIR", "../opsd_logs"),
        "repair_ttft_log_dir": os.environ.get("ARC_REPAIR_TTFT_LOG_DIR", "../repair_ttft_logs"),
        "repair_ttft_total_steps": int(os.environ.get("ARC_REPAIR_TTFT_TOTAL_STEPS", "128")),
        "repair_ttft_loo_stage1_steps": int(os.environ.get("ARC_REPAIR_TTFT_LOO_STAGE1_STEPS", "64")),
        "repair_ttft_warm_stage1_steps": int(os.environ.get("ARC_REPAIR_TTFT_WARM_STAGE1_STEPS", "32")),
        "repair_ttft_stage2_repair_fraction": float(
            os.environ.get("ARC_REPAIR_TTFT_STAGE2_REPAIR_FRACTION", "0.5")
        ),
        "repair_ttft_loo_heldout_views": int(
            os.environ.get("ARC_REPAIR_TTFT_LOO_HELDOUT_VIEWS", "16")
        ),
        "repair_ttft_loo_seen_views": int(
            os.environ.get("ARC_REPAIR_TTFT_LOO_SEEN_VIEWS", "4")
        ),
        "repair_ttft_warm_views_per_pair": int(
            os.environ.get("ARC_REPAIR_TTFT_WARM_VIEWS_PER_PAIR", "8")
        ),
        "scheduled_sampling_warmup_steps": int(
            os.environ.get("ARC_SCHEDULED_SAMPLING_WARMUP_STEPS", "32")
        ),
        "scheduled_sampling_total_steps": int(
            os.environ.get("ARC_SCHEDULED_SAMPLING_TOTAL_STEPS", "128")
        ),
        "scheduled_sampling_mix_probability": float(
            os.environ.get("ARC_SCHEDULED_SAMPLING_MIX_PROBABILITY", "0.5")
        ),
        "scheduled_sampling_log_dir": os.environ.get(
            "ARC_SCHEDULED_SAMPLING_LOG_DIR", "../scheduled_sampling_logs"
        ),
        "fixed_candidate_dir": os.environ.get("ARC_FIXED_CANDIDATE_DIR"),
        "selected_augmentations_path": os.environ.get("ARC_SELECTED_AUGMENTATIONS_PATH"),
        "canon_ac_state": os.environ.get("ARC_CANON_AC_STATE"),
    }


def _get_auto_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer


def _make_unsloth_fixed_trainer_class(UnslothTrainer):
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
            return (loss, outputs) if return_outputs else loss

    return UnslothFixedTrainer


def _make_qwen_data_collator_class(DataCollatorForLanguageModeling):
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
                if not batch["labels"][i].ne(-100).any():
                    raise RuntimeError(
                        "Completion-only collator masked every label; "
                        f"user_positions={user_start_idx} assistant_positions={assistant_start_idx} "
                        f"eos_positions={end_idx.tolist()}"
                    )
            return batch

    return QwenDataCollatorForCompletionOnlyLM


def _restore_qwen_role_tokens(model, tokenizer):
    """Restore the ordinary role tokens hidden by Transformers 5 token loading."""
    from transformers import AddedToken

    tokenizer.add_tokens(
        [
            AddedToken("user", special=False, normalized=False),
            AddedToken("assistant", special=False, normalized=False),
        ],
        special_tokens=False,
    )
    observed = {
        "user": tokenizer.encode("user", add_special_tokens=False),
        "assistant": tokenizer.encode("assistant", add_special_tokens=False),
    }
    expected = {"user": [USER_TOKEN_ID], "assistant": [ASSISTANT_TOKEN_ID]}
    if observed != expected:
        raise RuntimeError(f"Qwen role-token restoration failed: expected={expected} observed={observed}")

    tokenizer_vocab = len(tokenizer)
    input_vocab = model.get_input_embeddings().weight.shape[0]
    output_vocab = model.get_output_embeddings().weight.shape[0]
    if tokenizer_vocab != input_vocab or tokenizer_vocab != output_vocab:
        raise RuntimeError(
            "Tokenizer/model vocabulary mismatch after role-token restoration: "
            f"tokenizer={tokenizer_vocab} input={input_vocab} output={output_vocab}"
        )


def _get_unsloth_training_stack():
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    return {
        "FastLanguageModel": FastLanguageModel,
        "UnslothTrainingArguments": UnslothTrainingArguments,
        "UnslothFixedTrainer": _make_unsloth_fixed_trainer_class(UnslothTrainer),
        "QwenDataCollatorForCompletionOnlyLM": _make_qwen_data_collator_class(DataCollatorForLanguageModeling),
        "get_peft_model_state_dict": get_peft_model_state_dict,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "AutoTokenizer": AutoTokenizer,
    }


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
    manifest_dir = os.path.dirname(manifest_path) or "."
    os.makedirs(manifest_dir, exist_ok=True)
    lock_path = f"{manifest_path}.lock"
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        data = _load_manifest_file(manifest_path)
        entries = data.get("entries", [])
        filtered = [existing for existing in entries if existing.get("key") != entry["key"]]
        filtered.append(entry)
        filtered.sort(key=lambda item: item.get("key", ""))
        data["entries"] = filtered
        with tempfile.NamedTemporaryFile("w", delete=False, dir=manifest_dir, prefix=".manifest.", suffix=".tmp") as tmp_file:
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


def _build_eval_batches(eval_ds, tokenizer=None, formatter=None):
    test_id_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        test_id_to_subkeys[test_id].append(subkey)

    if any(len(subkeys) != 16 for subkeys in test_id_to_subkeys.values()):
        if tokenizer is None or formatter is None:
            raise ValueError(
                "Non-16-view inference requires tokenizer-aware batching so "
                "each DFS prefill batch has equal-length prompts"
            )
        batches = []
        for _test_id, subkeys in test_id_to_subkeys.items():
            length_to_subkeys = defaultdict(list)
            for subkey in subkeys:
                data = eval_ds.get(subkey, formatter)
                token_length = len(tokenizer.encode(data["input"]))
                length_to_subkeys[token_length].append(subkey)
            for token_length in sorted(length_to_subkeys):
                matching_subkeys = length_to_subkeys[token_length]
                batches.extend(
                    matching_subkeys[offset : offset + 4]
                    for offset in range(0, len(matching_subkeys), 4)
                )
        return batches

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
    eval_color_permutations = int(os.environ.get("ARC_EVAL_COLOR_PERMUTATIONS", "2"))
    eval_ds = puzzle_ds_multi.augment(n=eval_color_permutations, seed=2)
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)
    return puzzle_ds_multi, eval_ds


def _prepare_shared_eval_ds(
    puzzle_ds,
    formatter,
    max_seq_length: int,
    max_new_tokens: int,
    color_permutations: int,
    seed: int,
):
    puzzle_ds_multi = puzzle_ds.split_multi_replies()
    eval_ds = puzzle_ds.augment(n=color_permutations, seed=seed)
    eval_ds = eval_ds.split_multi_replies_shared_views()
    eval_ds = eval_ds.cut_to_len(
        formatter=formatter,
        name="input",
        max_len=max_seq_length - max_new_tokens,
    )
    return puzzle_ds_multi, eval_ds


def _rescore_fixed_candidate_pool(
    model,
    tokenizer,
    formatter,
    puzzle_ds_multi,
    puzzle_key: str,
    candidate_dir: str,
    output_dir: str,
    max_seq_length: int,
    max_new_tokens: int,
    timing_stats,
    count_stats,
):
    if not os.path.isdir(candidate_dir):
        raise FileNotFoundError(f"Fixed candidate directory does not exist: {candidate_dir}")
    source_names = sorted(name for name in os.listdir(candidate_dir) if name.startswith(f"{puzzle_key}_"))
    if not source_names:
        raise FileNotFoundError(f"No fixed-pool candidates found for puzzle {puzzle_key} in {candidate_dir}")

    rescorers = {}
    known_scores = {}
    for source_name in source_names:
        source_path = os.path.join(candidate_dir, source_name)
        if not os.path.isfile(source_path):
            continue
        with bz2.BZ2File(source_path, "rb") as source_file:
            decoded_result = pickle.load(source_file)
        base_key = source_name.split(".", 1)[0]
        if base_key not in puzzle_ds_multi.queries:
            raise KeyError(f"Candidate file {source_name} refers to unknown test output {base_key}")
        if base_key not in rescorers:
            started_at = time.perf_counter()
            rescorers[base_key] = FullPassRescorer(
                model=model,
                tokenizer=tokenizer,
                formatter=formatter,
                puzzle_ds_multi=puzzle_ds_multi,
                base_key=base_key,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                seed=stable_seed_from_key(base_key),
            )
            timing_stats["rescorer_init_s"] += time.perf_counter() - started_at
            count_stats["rescorers_created"] += 1

        for candidate in decoded_result:
            solution = np.asarray(candidate["solution"])
            grid_id = (base_key, tuple(map(tuple, solution.tolist())))
            if grid_id not in known_scores:
                started_at = time.perf_counter()
                known_scores[grid_id] = rescorers[base_key].score_solution(solution)
                timing_stats["rescoring_s"] += time.perf_counter() - started_at
                count_stats["rescoring_cache_misses"] += 1
            else:
                count_stats["rescoring_cache_hits"] += 1
            candidate["score_aug"] = known_scores[grid_id]
            count_stats["beam_candidates_valid"] += 1

        target_path = os.path.join(output_dir, source_name)
        with bz2.BZ2File(target_path, "wb") as target_file:
            pickle.dump(decoded_result, target_file)
        count_stats["subkeys_written"] += 1

    count_stats["fixed_pool_source_files"] += len(source_names)
    count_stats["fixed_pool_unique_candidates"] += len(known_scores)
    for rescorer in rescorers.values():
        print(rescorer.format_stats())


def _consume_hf_dfs_result(
    *,
    rank,
    subkeys,
    dfs_result,
    model,
    tokenizer,
    formatter,
    puzzle_ds_multi,
    output_dir,
    max_seq_length,
    max_new_tokens,
    timing_stats,
    count_stats,
    timing_prefix,
    known_scores,
    rescorers,
):
    prefix = f"{timing_prefix}_" if timing_prefix else ""
    for subkey_id, scored_beams in dfs_result:
        subkey = subkeys[subkey_id]
        base_key = subkey.split(".")[0]
        decoded_result = []
        count_stats[f"{prefix}subkeys_scored"] += 1

        if base_key not in rescorers:
            rescorer_started_at = time.perf_counter()
            rescorers[base_key] = FullPassRescorer(
                model=model,
                tokenizer=tokenizer,
                formatter=formatter,
                puzzle_ds_multi=puzzle_ds_multi,
                base_key=base_key,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                seed=stable_seed_from_key(base_key),
            )
            timing_stats[f"{prefix}rescorer_init_s"] += (
                time.perf_counter() - rescorer_started_at
            )
            count_stats[f"{prefix}rescorers_created"] += 1

        for beam_score, beam_tokens in scored_beams:
            count_stats[f"{prefix}beam_candidates_seen"] += 1
            array = formatter.convert_tokens_to_array(beam_tokens)
            if array is None:
                count_stats[f"{prefix}beam_candidates_invalid"] += 1
                continue

            solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
            grid_id = (base_key, tuple(map(tuple, solution)))
            count_stats[f"{prefix}beam_candidates_valid"] += 1
            if grid_id in known_scores:
                augmented_scores = known_scores[grid_id]
                count_stats[f"{prefix}rescoring_cache_hits"] += 1
            else:
                print(
                    f"[Rank {rank}] {timing_prefix or 'primary'} scoring "
                    f"{subkey} #{len(decoded_result)}"
                )
                rescore_started_at = time.perf_counter()
                augmented_scores = rescorers[base_key].score_solution(solution)
                timing_stats[f"{prefix}rescoring_s"] += (
                    time.perf_counter() - rescore_started_at
                )
                known_scores[grid_id] = augmented_scores
                count_stats[f"{prefix}rescoring_cache_misses"] += 1

            decoded_result.append(
                {
                    "beam_score": beam_score,
                    "score_aug": augmented_scores,
                    "solution": solution,
                }
            )

        if decoded_result:
            write_started_at = time.perf_counter()
            with bz2.BZ2File(os.path.join(output_dir, subkey), "w") as output_file:
                pickle.dump(decoded_result, output_file)
            timing_stats[f"{prefix}write_results_s"] += (
                time.perf_counter() - write_started_at
            )
            count_stats[f"{prefix}subkeys_written"] += 1


def _run_hf_eval_batches(
    *,
    rank,
    key,
    model,
    tokenizer,
    formatter,
    puzzle_ds_multi,
    eval_ds,
    output_dir,
    max_seq_length,
    max_new_tokens,
    max_score,
    use_multitoken,
    use_structured_rows,
    repeat_len,
    start_time,
    end_time,
    timing_stats,
    count_stats,
    timing_prefix="",
    known_scores=None,
    rescorers=None,
    frontier_max_score=None,
    frontier_batches=None,
):
    """Decode one primary or adaptive view set using the live TTFT model."""
    os.makedirs(output_dir, exist_ok=True)
    batches = _build_eval_batches(eval_ds, tokenizer=tokenizer, formatter=formatter)
    known_scores = {} if known_scores is None else known_scores
    rescorers = {} if rescorers is None else rescorers
    if frontier_max_score is not None and frontier_batches is None:
        frontier_batches = []
    prefix = f"{timing_prefix}_" if timing_prefix else ""

    for subkeys in batches:
        spend_time = time.time() - start_time
        if spend_time > 1200 or time.time() > end_time:
            print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
            break

        print(f"[Rank {rank}] {timing_prefix or 'primary'} decoding {subkeys}")
        count_stats[f"{prefix}batches"] += 1

        tokenize_started_at = time.perf_counter()
        tokens = [
            tokenizer.encode(eval_ds.get(subkey, formatter)["input"])
            for subkey in subkeys
        ]
        timing_stats[f"{prefix}tokenize_inputs_s"] += (
            time.perf_counter() - tokenize_started_at
        )

        dfs_started_at = time.perf_counter()
        if use_multitoken:
            multitoken_stats = {}
            search_result = inference_turbo_dfs_multitoken(
                model,
                tokens,
                max_new_tokens,
                max_score,
                end_time,
                repeat_len=repeat_len,
                stats=multitoken_stats,
                structured_rows=use_structured_rows,
                frontier_max_score=frontier_max_score,
                return_frontier=frontier_max_score is not None,
            )
            if frontier_max_score is not None:
                dfs_result, batch_frontier = search_result
                frontier_batches.append(
                    {
                        "subkeys": list(subkeys),
                        "tokens": tokens,
                        "frontier": batch_frontier,
                    }
                )
            else:
                dfs_result = search_result
            for name, value in multitoken_stats.items():
                if name == "model_time_s":
                    timing_stats[f"{prefix}dfs_multitoken_model_s"] += value
                else:
                    count_stats[f"{prefix}dfs_multitoken_{name}"] += value
        else:
            dfs_result = inference_turbo_dfs(
                model, tokens, max_new_tokens, max_score, end_time
            )
        timing_stats[f"{prefix}dfs_s"] += time.perf_counter() - dfs_started_at
        count_stats[f"{prefix}dfs_calls"] += 1

        _consume_hf_dfs_result(
            rank=rank,
            subkeys=subkeys,
            dfs_result=dfs_result,
            model=model,
            tokenizer=tokenizer,
            formatter=formatter,
            puzzle_ds_multi=puzzle_ds_multi,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            timing_stats=timing_stats,
            count_stats=count_stats,
            timing_prefix=timing_prefix,
            known_scores=known_scores,
            rescorers=rescorers,
        )

    return known_scores, rescorers


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
    batches = _build_eval_batches(eval_ds, tokenizer=tokenizer, formatter=formatter)
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
    backend_next_arc_cached_tokens = int(backend.stats["next_arc_logprobs_cached_tokens"])
    backend_next_arc_max_batch = int(backend.stats["next_arc_logprobs_max_batch"])
    backend_draft_arc_s = backend.stats["draft_arc_logprobs_time_s"]
    backend_draft_arc_calls = int(backend.stats["draft_arc_logprobs_calls"])
    backend_draft_arc_prompts = int(backend.stats["draft_arc_logprobs_prompts"])
    backend_draft_arc_prompt_tokens = int(backend.stats["draft_arc_logprobs_prompt_tokens"])
    backend_draft_arc_tokens = int(backend.stats["draft_arc_logprobs_draft_tokens"])
    backend_draft_arc_cached_tokens = int(backend.stats["draft_arc_logprobs_cached_tokens"])
    backend_draft_arc_max_batch = int(backend.stats["draft_arc_logprobs_max_batch"])
    backend_total_calls = backend_next_arc_calls + backend_draft_arc_calls
    backend_total_positions = backend_next_arc_prompts + backend_draft_arc_tokens
    backend_total_prompt_tokens = (
        backend_next_arc_prompt_tokens + backend_draft_arc_prompt_tokens + backend_draft_arc_tokens
    )
    backend_total_cached_tokens = backend_next_arc_cached_tokens + backend_draft_arc_cached_tokens
    backend_total_time_s = backend_next_arc_s + backend_draft_arc_s
    backend_max_batch = max(backend_next_arc_max_batch, backend_draft_arc_max_batch)
    backend_avg_positions_per_call = backend_total_positions / backend_total_calls if backend_total_calls else 0.0
    backend_cache_fraction = (
        backend_total_cached_tokens / backend_total_prompt_tokens if backend_total_prompt_tokens else 0.0
    )
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
        f"next_cached_tokens={backend_next_arc_cached_tokens} "
        f"draft_prompt_tokens={backend_draft_arc_prompt_tokens} "
        f"draft_tokens_verified={backend_draft_arc_tokens} "
        f"draft_cached_tokens={backend_draft_arc_cached_tokens} "
        f"cache_fraction={backend_cache_fraction:.6f} "
        f"avg_positions_per_call={backend_avg_positions_per_call:.2f} "
        f"max_batch_size={backend_max_batch}"
    )
    backend_score_calls = int(backend.stats["score_arc_logprobs_calls"])
    backend_score_prompts = int(backend.stats["score_arc_logprobs_prompts"])
    backend_score_prompt_tokens = int(backend.stats["score_arc_logprobs_prompt_tokens"])
    backend_score_answer_tokens = int(backend.stats["score_arc_logprobs_answer_tokens"])
    backend_score_cached_tokens = int(backend.stats["score_arc_logprobs_cached_tokens"])
    backend_score_input_tokens = backend_score_prompt_tokens + backend_score_answer_tokens
    backend_score_cache_fraction = (
        backend_score_cached_tokens / backend_score_input_tokens if backend_score_input_tokens else 0.0
    )
    print(
        f"[Rank {rank}] rescore backend summary for {key}: "
        f"calls={backend_score_calls} "
        f"prompts={backend_score_prompts} "
        f"prompt_tokens={backend_score_prompt_tokens} "
        f"answer_tokens={backend_score_answer_tokens} "
        f"cached_tokens={backend_score_cached_tokens} "
        f"cache_fraction={backend_score_cache_fraction:.6f}"
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
        use_gradient_checkpointing=True,
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
        gradient_checkpointing=True,
    )

    max_seq_length = 8192
    max_score = default_max_score(config["dfs_prob_threshold"])
    print(
        f"[Rank {rank}] config: use_sglang=True tp_size={config['sglang_tensor_parallel_size']} "
        f"mem_fraction_static={config['sglang_mem_fraction_static']} dfs_prob_threshold={config['dfs_prob_threshold']} "
        f"speculative_dfs={config['use_speculative_dfs']} sglang_speculative_repeat_len={config['sglang_speculative_repeat_len']} "
        f"dynamic_repeat={config['sglang_dynamic_repeat']} "
        f"train_adapters_only={config['sglang_train_adapters_only']} reuse_adapters={config['sglang_reuse_adapters']} "
        f"persistent_infer={config['sglang_persistent_infer']} consume_adapters={config['sglang_consume_adapters']}"
    )

    arc_test_set = ArcDataset.from_file(config["test_path"])
    dir_outputs = config["output_dir"]
    os.makedirs(dir_outputs, exist_ok=True)
    if config["sglang_train_adapters_only"]:
        os.makedirs(config["sglang_adapter_dir"], exist_ok=True)
    os.makedirs(config["sglang_adapter_dir"], exist_ok=True)
    persistent_backend = None
    persistent_tokenizer = None
    persistent_formatter = None
    persistent_max_new_tokens = None
    if config["sglang_persistent_infer"]:
        AutoTokenizer = _get_auto_tokenizer()
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
        while True:
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
                    if config["selected_augmentations_path"]:
                        descriptors = load_selected_augmentations(
                            config["selected_augmentations_path"], key
                        )
                        eval_ds = prepare_selected_eval_ds(
                            puzzle_ds_multi,
                            descriptors,
                            formatter,
                            max_seq_length,
                            max_new_tokens,
                        )
                        print(
                            f"[Rank {rank}] selected augmentation replay for {key}: "
                            f"descriptors={len(descriptors)} outputs={len(puzzle_ds_multi.keys)}"
                        )
                    timing_stats["eval_prep_s"] += time.perf_counter() - prep_started_at
                    print(f"[Rank {rank}] persistent-infer adapter loaded for puzzle {key}: {adapter_path}")
                elif config["sglang_reuse_adapters"]:
                    if not os.path.isdir(adapter_path):
                        raise FileNotFoundError(f"Saved adapter not found for {key}: {adapter_path}")
                    AutoTokenizer = _get_auto_tokenizer()
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
                    training_stack = _get_unsloth_training_stack()
                    FastLanguageModel = training_stack["FastLanguageModel"]
                    UnslothTrainingArguments = training_stack["UnslothTrainingArguments"]
                    UnslothFixedTrainer = training_stack["UnslothFixedTrainer"]
                    QwenDataCollatorForCompletionOnlyLM = training_stack["QwenDataCollatorForCompletionOnlyLM"]
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=config["model_path"],
                        full_finetuning=False,
                        load_in_4bit=False,
                        local_files_only=True,
                        use_gradient_checkpointing=True,
                        max_seq_length=max_seq_length,
                    )
                    _restore_qwen_role_tokens(model, tokenizer)
                    model = FastLanguageModel.get_peft_model(model, **peft_params)
                    for _name, param in model.named_parameters():
                        if param.dtype == torch.float32:
                            param.data = param.data.to(torch.bfloat16)

                    collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
                    formatter = QwenFormatter(tokenizer=tokenizer)
                    max_new_tokens = formatter.max_new_tokens()
                    model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
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
                    if config["sglang_consume_adapters"]:
                        shutil.rmtree(adapter_path, ignore_errors=True)
                        _upsert_manifest_entry(
                            _default_sglang_manifest_path(config),
                            {
                                "key": key,
                                "adapter_path": adapter_path,
                                "size_bytes": 0,
                                "status": "consumed",
                                "updated_at": time.time(),
                            },
                        )
                        print(f"[Rank {rank}] consumed adapter for puzzle {key}: {adapter_path}")
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
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
        gradient_checkpointing=True,
    )

    max_seq_length = 8192
    training_stack = _get_unsloth_training_stack()
    FastLanguageModel = training_stack["FastLanguageModel"]
    UnslothTrainingArguments = training_stack["UnslothTrainingArguments"]
    UnslothFixedTrainer = training_stack["UnslothFixedTrainer"]
    OnePassScheduledSamplingTrainer = make_one_pass_scheduled_sampling_trainer_class(
        UnslothFixedTrainer
    )
    QwenDataCollatorForCompletionOnlyLM = training_stack["QwenDataCollatorForCompletionOnlyLM"]
    get_peft_model_state_dict = training_stack["get_peft_model_state_dict"]
    set_peft_model_state_dict = training_stack["set_peft_model_state_dict"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_path"],
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=True,
        max_seq_length=max_seq_length,
    )
    _restore_qwen_role_tokens(model, tokenizer)

    canon_hooks = None
    default_canon_weights = None
    if config["canon_ac_state"]:
        from arc_canon import (
            add_canon_ac_modules,
            canon_parameters,
            canon_state_dict,
            install_canon_ac_training_hooks,
            load_canon_state,
            load_canon_state_dict,
        )

        canon_path = os.path.abspath(config["canon_ac_state"])
        if not os.path.isfile(canon_path):
            raise FileNotFoundError(canon_path)
        add_canon_ac_modules(model, kernel_size=4, zero_init=True)
        load_canon_state(model, canon_path)
        canon_hooks = install_canon_ac_training_hooks(model)
        print(f"[Rank {rank}] loaded residual Canon-AC from {canon_path}", flush=True)

    model = FastLanguageModel.get_peft_model(model, **peft_params)
    if config["canon_ac_state"]:
        # PEFT freezes every base-model parameter when it installs the fresh
        # per-task adapter. Canon is deliberately trainable during TTFT, so it
        # must be re-enabled explicitly after get_peft_model.
        for parameter in canon_parameters(model):
            parameter.requires_grad = True
        default_canon_weights = {
            key: value.clone().detach()
            for key, value in canon_state_dict(model).items()
        }
        print(
            f"[Rank {rank}] Canon TTFT trainable parameters="
            f"{sum(parameter.numel() for parameter in canon_parameters(model))}",
            flush=True,
        )
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
        f"dfs_prob_threshold={config['dfs_prob_threshold']} "
        f"multitoken_dfs={config['use_unsloth_multitoken_dfs']} "
        f"structured_rows={config['use_unsloth_structured_rows']} "
        f"ttft_method={config['ttft_method']} fixed_candidate_dir={config['fixed_candidate_dir']} "
        f"shared_eval_augmentations={config['shared_eval_augmentations']} "
        f"adaptive_output_dir={config['adaptive_output_dir']} "
        f"canon_ac={config['canon_ac_state'] is not None}"
    )
    if config["ttft_method"] != "full_sft" and config["use_sglang"]:
        raise ValueError("Reduced-pair and OPSD TTFT modes require the gradient-capable Unsloth/HF worker")

    arc_test_set = ArcDataset.from_file(config["test_path"])
    dir_outputs = config["output_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    while True:
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
        if default_canon_weights is not None:
            load_canon_state_dict(model, default_canon_weights)

        model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
        activate_adapter(model, "default", trainable=True)
        if default_canon_weights is not None:
            for parameter in canon_parameters(model):
                parameter.requires_grad = True
        puzzle_ds = arc_test_set.change_keys([key])
        num_train_pairs = len(puzzle_ds.queries[key]["train"])
        effective_ttft_method = config["ttft_method"]
        opsd_split = None
        if effective_ttft_method in {
            "reduced_sft",
            "reduced_plus_sft_c",
            "reduced_plus_opsd",
            "loo_repair_mix",
        } and num_train_pairs < config["opsd_min_train_pairs"]:
            print(
                f"[Rank {rank}] {key}: falling back to full_sft because it has "
                f"{num_train_pairs} training pairs (< {config['opsd_min_train_pairs']})"
            )
            effective_ttft_method = "full_sft"
        if effective_ttft_method in {
            "reduced_sft",
            "reduced_plus_sft_c",
            "reduced_plus_opsd",
            "loo_repair_mix",
        }:
            opsd_split = split_puzzle_for_opsd(puzzle_ds, key)
            initial_sft_ds = opsd_split.reduced_dataset
            print(
                f"[Rank {rank}] {key}: reserved C pair index={opsd_split.reserved_pair_index}; "
                f"initial SFT pairs={opsd_split.sft_pair_indices}"
            )
        else:
            initial_sft_ds = puzzle_ds

        train_ds = initial_sft_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)
        train_rows = train_ds.as_list(formatter)
        repair_ttft_stats = None
        if effective_ttft_method in REPAIR_TTFT_METHODS:
            if config["repair_ttft_total_steps"] != 128:
                raise ValueError("The controlled repair-TTFT experiment requires exactly 128 total steps")
            if effective_ttft_method == "loo_repair_mix":
                stage1_steps = config["repair_ttft_loo_stage1_steps"]
            else:
                stage1_steps = config["repair_ttft_warm_stage1_steps"]
            if not 0 < stage1_steps < config["repair_ttft_total_steps"]:
                raise ValueError(
                    f"Invalid repair TTFT stage-1 budget {stage1_steps}/"
                    f"{config['repair_ttft_total_steps']}"
                )
            train_rows = deterministic_rows(
                train_rows,
                stage1_steps,
                stable_seed_from_key(f"{key}:stage1:{effective_ttft_method}"),
            )
            repair_ttft_stats = {
                "puzzle_key": key,
                "requested_method": config["ttft_method"],
                "effective_method": effective_ttft_method,
                "num_train_pairs": num_train_pairs,
                "reserved_pair_index": (
                    opsd_split.reserved_pair_index if opsd_split is not None else None
                ),
                "stage1_steps": stage1_steps,
                "total_steps": config["repair_ttft_total_steps"],
                "timing_s": {},
            }

        training_started_at = time.perf_counter()
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer_class = (
                OnePassScheduledSamplingTrainer
                if effective_ttft_method == "one_pass_ss"
                else UnslothFixedTrainer
            )
            trainer_kwargs = {}
            if effective_ttft_method == "one_pass_ss":
                if not (
                    0 <= config["scheduled_sampling_warmup_steps"]
                    < config["scheduled_sampling_total_steps"]
                    <= len(train_rows)
                ):
                    raise ValueError(
                        "One-pass scheduled-sampling steps must satisfy "
                        "0 <= warmup < total <= available training rows; got "
                        f"warmup={config['scheduled_sampling_warmup_steps']}, "
                        f"total={config['scheduled_sampling_total_steps']}, "
                        f"rows={len(train_rows)}"
                    )
                trainer_kwargs = {
                    "scheduled_sampling_warmup_steps": config["scheduled_sampling_warmup_steps"],
                    "scheduled_sampling_mix_probability": config[
                        "scheduled_sampling_mix_probability"
                    ],
                    "scheduled_sampling_seed": stable_seed_from_key(f"{key}:one-pass-ss"),
                }
            initial_train_args = dict(train_args)
            if effective_ttft_method == "one_pass_ss":
                initial_train_args["max_steps"] = config["scheduled_sampling_total_steps"]
            trainer = trainer_class(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_rows),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**initial_train_args),
                **trainer_kwargs,
            )

            stats = trainer.train()
            scheduled_sampling_stats = (
                trainer.scheduled_sampling_summary()
                if effective_ttft_method == "one_pass_ss"
                else None
            )
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            del trainer
        timing_stats["training_s"] += time.perf_counter() - training_started_at

        if default_canon_weights is not None:
            adapted_canon_weights = canon_state_dict(model)
            canon_delta_l2 = sum(
                (adapted_canon_weights[name].float() - default_canon_weights[name].float())
                .square()
                .sum()
                .item()
                for name in default_canon_weights
            ) ** 0.5
            print(
                f"[Rank {rank}] {key}: Canon TTFT delta_l2={canon_delta_l2:.8f}",
                flush=True,
            )

        if scheduled_sampling_stats is not None:
            scheduled_sampling_stats.update(
                {
                    "puzzle_key": key,
                    "num_train_pairs": num_train_pairs,
                    "training_s": timing_stats["training_s"],
                    "trainer_global_step": int(stats.global_step),
                    "trainer_training_loss": float(stats.training_loss),
                }
            )
            os.makedirs(config["scheduled_sampling_log_dir"], exist_ok=True)
            with open(
                os.path.join(config["scheduled_sampling_log_dir"], f"{_safe_path_key(key)}.json"),
                "w",
            ) as log_file:
                json.dump(scheduled_sampling_stats, log_file, indent=2, sort_keys=True)
                log_file.write("\n")
            print(
                f"[Rank {rank}] {key}: one-pass SS "
                f"steps={scheduled_sampling_stats['scheduled_sampling_steps']} "
                f"digit_accuracy={scheduled_sampling_stats['teacher_forced_digit_accuracy']} "
                f"changed_fraction={scheduled_sampling_stats['realized_change_fraction']}"
            )

        if effective_ttft_method in REPAIR_TTFT_METHODS:
            mining_started_at = time.perf_counter()
            model = FastLanguageModel.for_inference(model)
            if effective_ttft_method == "loo_repair_mix":
                pair_view_counts = {
                    pair_index: (
                        config["repair_ttft_loo_heldout_views"]
                        if pair_index == opsd_split.reserved_pair_index
                        else config["repair_ttft_loo_seen_views"]
                    )
                    for pair_index in range(num_train_pairs)
                }
            else:
                pair_view_counts = {
                    pair_index: config["repair_ttft_warm_views_per_pair"]
                    for pair_index in range(num_train_pairs)
                }
            repair_rows, mining_stats = mine_repair_examples(
                model=model,
                tokenizer=tokenizer,
                formatter=formatter,
                puzzle_ds=puzzle_ds,
                puzzle_key=key,
                pair_view_counts=pair_view_counts,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                seed=stable_seed_from_key(f"{key}:repair-mine:{effective_ttft_method}"),
            )
            mining_elapsed = time.perf_counter() - mining_started_at
            timing_stats["repair_mining_s"] += mining_elapsed
            repair_ttft_stats["timing_s"]["repair_mining_s"] = mining_elapsed
            repair_ttft_stats["mining"] = mining_stats

            ordinary_source = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
            ordinary_source = ordinary_source.cut_to_len(
                formatter=formatter,
                name="text",
                max_len=max_seq_length,
            )
            ordinary_rows = []
            for row in ordinary_source.as_list(formatter):
                tokenized = tokenize_ordinary_text(row["text"], tokenizer, max_seq_length)
                if tokenized is not None:
                    ordinary_rows.append(tokenized)
            stage2_steps = config["repair_ttft_total_steps"] - repair_ttft_stats["stage1_steps"]
            mixed_rows, mixture_stats = build_stage_two_mixture(
                ordinary_rows=ordinary_rows,
                repair_rows=repair_rows,
                total_steps=stage2_steps,
                repair_fraction=config["repair_ttft_stage2_repair_fraction"],
                seed=stable_seed_from_key(f"{key}:stage2:{effective_ttft_method}"),
            )
            repair_ttft_stats["stage2_steps"] = stage2_steps
            repair_ttft_stats["stage2_mixture"] = mixture_stats

            correction_started_at = time.perf_counter()
            activate_adapter(model, "default", trainable=True)
            model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                correction_trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=mixed_completion_collator(tokenizer),
                    train_dataset=Dataset.from_list(mixed_rows),
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )
                correction_stats = correction_trainer.train()
                model = correction_trainer.accelerator.unwrap_model(
                    model, keep_fp32_wrapper=False
                )
                del correction_trainer
            correction_elapsed = time.perf_counter() - correction_started_at
            timing_stats["repair_stage2_training_s"] += correction_elapsed
            repair_ttft_stats["timing_s"]["stage2_training_s"] = correction_elapsed
            repair_ttft_stats["stage2_train_stats"] = {
                "global_step": int(correction_stats.global_step),
                "training_loss": float(correction_stats.training_loss),
            }
            os.makedirs(config["repair_ttft_log_dir"], exist_ok=True)
            with open(
                os.path.join(config["repair_ttft_log_dir"], f"{_safe_path_key(key)}.json"),
                "w",
            ) as log_file:
                json.dump(repair_ttft_stats, log_file, indent=2, sort_keys=True)
                log_file.write("\n")
            print(
                f"[Rank {rank}] {key}: repair TTFT method={effective_ttft_method} "
                f"stage1={repair_ttft_stats['stage1_steps']} "
                f"stage2={stage2_steps} repairs={len(repair_rows)} "
                f"mixture={mixture_stats}"
            )

        opsd_stats = None
        if effective_ttft_method == "reduced_plus_sft_c":
            correction_started_at = time.perf_counter()
            correction_examples = build_opsd_examples(
                opsd_split,
                formatter=formatter,
                color_permutations=config["opsd_color_permutations"],
                cross_view_probability=0.0,
                seed=stable_seed_from_key(key),
            )
            correction_rows = [
                {"text": example.student_prompt + example.gold_reply}
                for example in correction_examples
                if len(tokenizer.encode(example.student_prompt + example.gold_reply)) <= max_seq_length
            ]
            if not correction_rows:
                raise RuntimeError(f"No augmented-SFT C examples fit max_seq_length for {key}")
            activate_adapter(model, "default", trainable=True)
            model = FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                correction_trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=Dataset.from_list(correction_rows),
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )
                correction_stats = correction_trainer.train()
                model = correction_trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                del correction_trainer
            timing_stats["sft_c_correction_s"] += time.perf_counter() - correction_started_at
            print(
                f"[Rank {rank}] {key}: augmented-SFT control trained on "
                f"{len(correction_rows)} transformed C examples; stats={correction_stats}"
            )
        if effective_ttft_method == "reduced_plus_opsd":
            correction_started_at = time.perf_counter()
            examples = build_opsd_examples(
                opsd_split,
                formatter=formatter,
                color_permutations=config["opsd_color_permutations"],
                cross_view_probability=config["opsd_cross_view_probability"],
                seed=stable_seed_from_key(key),
            )
            clone_frozen_teacher_adapter(
                model,
                get_peft_model_state_dict=get_peft_model_state_dict,
                set_peft_model_state_dict=set_peft_model_state_dict,
            )
            try:
                opsd_stats = run_opsd_correction(
                    model=model,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    examples=examples,
                    max_seq_length=max_seq_length,
                    max_updates=config["opsd_max_updates"],
                    learning_rate=config["opsd_learning_rate"],
                    temperature=config["opsd_temperature"],
                    top_p=config["opsd_top_p"],
                    lambda_ce=config["opsd_lambda_ce"],
                    seed=stable_seed_from_key(key),
                )
            finally:
                remove_teacher_adapter(model, TEACHER_ADAPTER_NAME)
                model.set_adapter("default")
                for parameter in model.parameters():
                    parameter.requires_grad = False
            timing_stats["opsd_correction_s"] += time.perf_counter() - correction_started_at
            os.makedirs(config["opsd_log_dir"], exist_ok=True)
            opsd_log = {
                "puzzle_key": key,
                "requested_ttft_method": config["ttft_method"],
                "effective_ttft_method": effective_ttft_method,
                "num_train_pairs": num_train_pairs,
                "reserved_pair_index": opsd_split.reserved_pair_index,
                "sft_pair_indices": opsd_split.sft_pair_indices,
                "color_permutations": config["opsd_color_permutations"],
                "cross_view_probability": config["opsd_cross_view_probability"],
                "learning_rate": config["opsd_learning_rate"],
                "lambda_ce": config["opsd_lambda_ce"],
                "stats": opsd_stats,
            }
            with open(os.path.join(config["opsd_log_dir"], f"{_safe_path_key(key)}.json"), "w") as log_file:
                json.dump(opsd_log, log_file, indent=2, sort_keys=True)
                log_file.write("\n")
            print(
                f"[Rank {rank}] {key}: OPSD accepted "
                f"{opsd_stats['accepted_updates']}/{opsd_stats['attempted_examples']} attempted examples"
            )

        if config["sglang_train_adapters_only"]:
            adapter_path = _sglang_adapter_path(config, key)
            model.save_pretrained(adapter_path)
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
            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for training")
            torch.cuda.reset_peak_memory_stats()
            print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")
            spend_time = time.time() - start_time
            print(f"[Rank {rank}] saved adapter for puzzle {key} to {adapter_path}")
            print(f"[Rank {rank}] adapter manifest updated: {manifest_path}")
            print(f"[Rank {rank}] finished adapter-only pass for {key} in {spend_time:.1f}s")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        prep_started_at = time.perf_counter()
        model = FastLanguageModel.for_inference(model)
        if config["canon_ac_state"]:
            from arc_canon import prepare_canon_inference

            prepare_canon_inference(model)
        gc.collect()
        torch.cuda.empty_cache()

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")
        torch.cuda.reset_peak_memory_stats()
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        if config["shared_eval_augmentations"]:
            puzzle_ds_multi, eval_ds = _prepare_shared_eval_ds(
                puzzle_ds,
                formatter,
                max_seq_length,
                max_new_tokens,
                color_permutations=config["eval_color_permutations"],
                seed=2,
            )
        else:
            puzzle_ds_multi, eval_ds = _prepare_eval_ds(
                puzzle_ds, formatter, max_seq_length, max_new_tokens
            )
        timing_stats["eval_prep_s"] += time.perf_counter() - prep_started_at

        if config["fixed_candidate_dir"]:
            print(f"[Rank {rank}] rescoring fixed candidate pool for {key}")
            with torch.inference_mode():
                _rescore_fixed_candidate_pool(
                    model=model,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    puzzle_ds_multi=puzzle_ds_multi,
                    puzzle_key=key,
                    candidate_dir=config["fixed_candidate_dir"],
                    output_dir=dir_outputs,
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    timing_stats=timing_stats,
                    count_stats=count_stats,
                )
            # Phase 1 deliberately freezes the candidate pool; do not run DFS.
            eval_ds = eval_ds.change_keys([])

        with torch.inference_mode():
            frontier_batches = []
            known_scores, rescorers = _run_hf_eval_batches(
                rank=rank,
                key=key,
                model=model,
                tokenizer=tokenizer,
                formatter=formatter,
                puzzle_ds_multi=puzzle_ds_multi,
                eval_ds=eval_ds,
                output_dir=dir_outputs,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                max_score=max_score,
                use_multitoken=config["use_unsloth_multitoken_dfs"],
                use_structured_rows=config["use_unsloth_structured_rows"],
                repeat_len=config["unsloth_multitoken_repeat_len"],
                start_time=start_time,
                end_time=end_time,
                timing_stats=timing_stats,
                count_stats=count_stats,
                frontier_max_score=(
                    default_max_score(config["adaptive_dfs_prob_threshold"])
                    if config["adaptive_resume_frontier"]
                    else None
                ),
                frontier_batches=frontier_batches,
            )

            adaptive_output_dir = config["adaptive_output_dir"]
            starved = set()
            if adaptive_output_dir:
                unique_counts = {
                    base_key: sum(1 for candidate_key in known_scores if candidate_key[0] == base_key)
                    for base_key in puzzle_ds_multi.keys
                }
                starved = {
                    base_key
                    for base_key, unique_count in unique_counts.items()
                    if unique_count < config["adaptive_min_unique_candidates"]
                }
                print(
                    f"[Rank {rank}] primary unique candidates for {key}: "
                    f"{unique_counts}; adaptive_outputs={sorted(starved)}"
                )
                if starved and time.time() < end_time and time.time() - start_time <= 1200:
                    if config["adaptive_resume_frontier"]:
                        for frontier_batch in frontier_batches:
                            filtered_frontier = [
                                entry
                                for entry in frontier_batch["frontier"]
                                if frontier_batch["subkeys"][entry["lane"]].split(".")[0]
                                in starved
                            ]
                            if not filtered_frontier:
                                continue
                            resume_started_at = time.perf_counter()
                            resume_stats = {}
                            resumed = resume_turbo_dfs_multitoken(
                                model=model,
                                prefix_tokens=frontier_batch["tokens"],
                                frontier=filtered_frontier,
                                max_new_tokens=max_new_tokens,
                                max_score=default_max_score(
                                    config["adaptive_dfs_prob_threshold"]
                                ),
                                end_time=end_time,
                                repeat_len=config["unsloth_multitoken_repeat_len"],
                                stats=resume_stats,
                            )
                            timing_stats["adaptive_dfs_s"] += (
                                time.perf_counter() - resume_started_at
                            )
                            for name, value in resume_stats.items():
                                if name.endswith("_time_s"):
                                    timing_stats[f"adaptive_{name}"] += value
                                else:
                                    count_stats[f"adaptive_{name}"] += value
                            _consume_hf_dfs_result(
                                rank=rank,
                                subkeys=frontier_batch["subkeys"],
                                dfs_result=resumed,
                                model=model,
                                tokenizer=tokenizer,
                                formatter=formatter,
                                puzzle_ds_multi=puzzle_ds_multi,
                                output_dir=adaptive_output_dir,
                                max_seq_length=max_seq_length,
                                max_new_tokens=max_new_tokens,
                                timing_stats=timing_stats,
                                count_stats=count_stats,
                                timing_prefix="adaptive",
                                known_scores=known_scores,
                                rescorers=rescorers,
                            )
                    else:
                        adaptive_prep_started_at = time.perf_counter()
                        _, adaptive_eval_ds = _prepare_shared_eval_ds(
                            puzzle_ds,
                            formatter,
                            max_seq_length,
                            max_new_tokens,
                            color_permutations=config["adaptive_color_permutations"],
                            seed=3,
                        )
                        adaptive_eval_ds = adaptive_eval_ds.change_keys(
                            [
                                subkey
                                for subkey in adaptive_eval_ds.keys
                                if subkey.split(".")[0] in starved
                            ]
                        )
                        timing_stats["adaptive_eval_prep_s"] += (
                            time.perf_counter() - adaptive_prep_started_at
                        )
                        _run_hf_eval_batches(
                            rank=rank,
                            key=key,
                            model=model,
                            tokenizer=tokenizer,
                            formatter=formatter,
                            puzzle_ds_multi=puzzle_ds_multi,
                            eval_ds=adaptive_eval_ds,
                            output_dir=adaptive_output_dir,
                            max_seq_length=max_seq_length,
                            max_new_tokens=max_new_tokens,
                            max_score=default_max_score(
                                config["adaptive_dfs_prob_threshold"]
                            ),
                            use_multitoken=config["use_unsloth_multitoken_dfs"],
                            use_structured_rows=config["use_unsloth_structured_rows"],
                            repeat_len=config["unsloth_multitoken_repeat_len"],
                            start_time=start_time,
                            end_time=end_time,
                            timing_stats=timing_stats,
                            count_stats=count_stats,
                            timing_prefix="adaptive",
                            known_scores=known_scores,
                            rescorers=rescorers,
                        )

            compare_fresh_dir = config["compare_fresh_adaptive_output_dir"]
            if compare_fresh_dir and starved and time.time() < end_time:
                fresh_eval_ds = eval_ds.change_keys(
                    [
                        subkey
                        for subkey in eval_ds.keys
                        if subkey.split(".")[0] in starved
                    ]
                )
                fresh_timing_stats = defaultdict(float)
                fresh_count_stats = defaultdict(int)
                _run_hf_eval_batches(
                    rank=rank,
                    key=key,
                    model=model,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    puzzle_ds_multi=puzzle_ds_multi,
                    eval_ds=fresh_eval_ds,
                    output_dir=compare_fresh_dir,
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    max_score=default_max_score(
                        config["adaptive_dfs_prob_threshold"]
                    ),
                    use_multitoken=config["use_unsloth_multitoken_dfs"],
                    use_structured_rows=False,
                    repeat_len=config["unsloth_multitoken_repeat_len"],
                    start_time=time.time(),
                    end_time=end_time,
                    timing_stats=fresh_timing_stats,
                    count_stats=fresh_count_stats,
                    timing_prefix="fresh_compare",
                    known_scores={},
                    rescorers={},
                )
                diagnostics_dir = f"{compare_fresh_dir}_diagnostics"
                os.makedirs(diagnostics_dir, exist_ok=True)
                with open(os.path.join(diagnostics_dir, f"{key}.json"), "w") as f:
                    json.dump(
                        {
                            "timing": dict(fresh_timing_stats),
                            "counts": dict(fresh_count_stats),
                            "starved_outputs": sorted(starved),
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )

            compare_structured_dir = config["compare_structured_output_dir"]
            if compare_structured_dir and time.time() < end_time:
                structured_timing_stats = defaultdict(float)
                structured_count_stats = defaultdict(int)
                _run_hf_eval_batches(
                    rank=rank,
                    key=key,
                    model=model,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    puzzle_ds_multi=puzzle_ds_multi,
                    eval_ds=eval_ds,
                    output_dir=compare_structured_dir,
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    max_score=max_score,
                    use_multitoken=True,
                    use_structured_rows=True,
                    repeat_len=config["unsloth_multitoken_repeat_len"],
                    start_time=time.time(),
                    end_time=end_time,
                    timing_stats=structured_timing_stats,
                    count_stats=structured_count_stats,
                    timing_prefix="structured_compare",
                    known_scores={},
                    rescorers={},
                )
                diagnostics_dir = f"{compare_structured_dir}_diagnostics"
                os.makedirs(diagnostics_dir, exist_ok=True)
                with open(os.path.join(diagnostics_dir, f"{key}.json"), "w") as f:
                    json.dump(
                        {
                            "control_timing": dict(timing_stats),
                            "control_counts": dict(count_stats),
                            "structured_timing": dict(structured_timing_stats),
                            "structured_counts": dict(structured_count_stats),
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
        if config["profile_timings"]:
            timing_stats["total_wall_s"] = time.perf_counter() - puzzle_started_at
            ordered_timings = [
                "training_s",
                "opsd_correction_s",
                "sft_c_correction_s",
                "eval_prep_s",
                "tokenize_inputs_s",
                "dfs_s",
                "dfs_multitoken_model_s",
                "rescorer_init_s",
                "rescoring_s",
                "write_results_s",
                "adaptive_eval_prep_s",
                "adaptive_tokenize_inputs_s",
                "adaptive_dfs_s",
                "adaptive_dfs_multitoken_model_s",
                "adaptive_rescorer_init_s",
                "adaptive_rescoring_s",
                "adaptive_write_results_s",
                "total_wall_s",
            ]
            timings_text = " ".join(f"{name}={timing_stats[name]:.3f}s" for name in ordered_timings)
            counts_text = " ".join(
                f"{name}={count_stats[name]}"
                for name in [
                    "batches",
                    "dfs_calls",
                    "dfs_multitoken_block_calls",
                    "dfs_multitoken_draft_tokens",
                    "dfs_multitoken_accepted_extra_tokens",
                    "dfs_multitoken_zero_extra_blocks",
                    "subkeys_scored",
                    "subkeys_written",
                    "beam_candidates_seen",
                    "beam_candidates_valid",
                    "beam_candidates_invalid",
                    "rescoring_cache_hits",
                    "rescoring_cache_misses",
                    "rescorers_created",
                    "adaptive_batches",
                    "adaptive_dfs_calls",
                    "adaptive_subkeys_scored",
                    "adaptive_subkeys_written",
                    "adaptive_beam_candidates_seen",
                    "adaptive_beam_candidates_valid",
                    "adaptive_beam_candidates_invalid",
                    "adaptive_rescoring_cache_hits",
                    "adaptive_rescoring_cache_misses",
                    "fixed_pool_source_files",
                    "fixed_pool_unique_candidates",
                ]
            )
            print(f"[Rank {rank}] timing summary for {key}: {timings_text}")
            print(f"[Rank {rank}] count summary for {key}: {counts_text}")
            for base_key, rescorer in sorted(rescorers.items()):
                print(f"[Rank {rank}] rescorer summary for {base_key}: {rescorer.format_stats()}")
