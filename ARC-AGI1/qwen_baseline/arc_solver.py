import bz2
import gc
import hashlib
import io
import logging
import os
import pickle
import time
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Union

import numpy as np
import torch
from datasets import Dataset
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import DataCollatorForLanguageModeling
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

from arc_loader import ArcDataset, QwenFormatter
from arc_rescoring import FullPassRescorer, PrefixCachedRescorer
from arc_search import ASSISTANT_TOKEN_ID, EOS_ID, USER_TOKEN_ID, default_max_score, inference_turbo_dfs
from arc_vllm import ArcVllmBackend, VllmConfig, VllmRescorer, inference_vllm_dfs

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
    return {
        "use_prefix_cached_rescoring": _env_flag("ARC_USE_PREFIX_CACHED_RESCORING", default=False),
        "use_vllm": _env_flag("ARC_USE_VLLM", default=False),
        "vllm_adapter_dir": os.environ.get("ARC_VLLM_ADAPTER_DIR"),
        "vllm_tensor_parallel_size": int(os.environ.get("ARC_VLLM_TENSOR_PARALLEL_SIZE", "1")),
        "vllm_gpu_memory_utilization": float(os.environ.get("ARC_VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        "vllm_enable_prefix_caching": _env_flag("ARC_VLLM_ENABLE_PREFIX_CACHING", default=True),
        "use_speculative_dfs": _env_flag("ARC_USE_SPECULATIVE_DFS", default=False),
        "profile_timings": _env_flag("ARC_PROFILE_TIMINGS", default=False),
        "dfs_prob_threshold": dfs_prob_threshold,
        "model_path": os.environ.get("ARC_MODEL_PATH", "../input/qwen3_4b_grids15_sft139/"),
        "test_path": os.environ.get("ARC_TEST_PATH", "../input/arc-prize-2024/arc-agi_evaluation_challenges.json"),
        "output_dir": os.environ.get("ARC_OUTPUT_DIR", "../inference_outputs"),
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


def worker(rank, queue, end_time):
    config = runtime_config()
    if config["use_speculative_dfs"]:
        raise NotImplementedError("Speculative DFS is feature-flagged but not implemented yet")

    rerun_mode = True

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
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

    def load_training_model():
        loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_path"],
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=False,
            max_seq_length=max_seq_length,
        )

        loaded_model = FastLanguageModel.get_peft_model(loaded_model, **peft_params)
        for _name, param in loaded_model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

        weights = get_peft_model_state_dict(loaded_model, adapter_name="default")
        weights = {k: v.detach().cpu().clone() for k, v in weights.items()}

        loaded_collator = QwenDataCollatorForCompletionOnlyLM(
            tokenizer=loaded_tokenizer,
            mlm=False,
        )
        loaded_formatter = QwenFormatter(tokenizer=loaded_tokenizer)
        return loaded_model, loaded_tokenizer, weights, loaded_collator, loaded_formatter, loaded_formatter.max_new_tokens()

    if config["use_vllm"]:
        model = tokenizer = default_weights = collator = formatter = max_new_tokens = None
    else:
        model, tokenizer, default_weights, collator, formatter, max_new_tokens = load_training_model()

    max_score = default_max_score(config["dfs_prob_threshold"])
    rescoring_cls = PrefixCachedRescorer if config["use_prefix_cached_rescoring"] else FullPassRescorer
    print(
        f"[Rank {rank}] config: prefix_cached_rescoring={config['use_prefix_cached_rescoring']} "
        f"use_vllm={config['use_vllm']} speculative_dfs={config['use_speculative_dfs']} "
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

        if config["use_vllm"]:
            model, tokenizer, default_weights, collator, formatter, max_new_tokens = load_training_model()

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
        backend = None
        adapter_path = None

        if config["use_vllm"]:
            adapter_root = config["vllm_adapter_dir"] or os.path.join(config["output_dir"], "vllm_adapters")
            adapter_path = os.path.join(adapter_root, key)
            os.makedirs(adapter_path, exist_ok=True)
            model.save_pretrained(adapter_path)
            timing_stats["adapter_save_s"] += time.perf_counter() - prep_started_at
            del model
            del default_weights
            del collator
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        else:
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
            if config["use_vllm"]:
                backend = ArcVllmBackend(
                    VllmConfig(
                        model_path=config["model_path"],
                        adapter_path=adapter_path,
                        tensor_parallel_size=config["vllm_tensor_parallel_size"],
                        gpu_memory_utilization=config["vllm_gpu_memory_utilization"],
                        enable_prefix_caching=config["vllm_enable_prefix_caching"],
                        max_model_len=max_seq_length,
                    )
                )

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
                if config["use_vllm"]:
                    dfs_result = inference_vllm_dfs(backend, tokens, max_new_tokens, max_score, end_time)
                else:
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
                        rescorer_kwargs = dict(
                            tokenizer=tokenizer,
                            formatter=formatter,
                            puzzle_ds_multi=puzzle_ds_multi,
                            base_key=bk,
                            max_seq_length=max_seq_length,
                            max_new_tokens=max_new_tokens,
                            seed=stable_seed_from_key(bk),
                        )
                        if config["use_vllm"]:
                            rescorers[bk] = VllmRescorer(backend=backend, **rescorer_kwargs)
                        else:
                            rescorers[bk] = rescoring_cls(model=model, **rescorer_kwargs)
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

            if backend is not None:
                backend.close()
                del backend
            if config["use_vllm"]:
                del tokenizer
                del formatter

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
