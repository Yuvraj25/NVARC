import bz2
import gc
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
from arc_rescoring import PrefixCachedRescorer
from arc_search import ASSISTANT_TOKEN_ID, EOS_ID, USER_TOKEN_ID, default_max_score, inference_turbo_dfs

logging.disable(logging.WARNING)


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


def worker(rank, queue, end_time):
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="../input/qwen3_4b_grids15_sft139/",
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
    max_score = default_max_score()

    if rerun_mode:
        test_path = "../input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    else:
        test_path = "../input/arc-prize-2024/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)
    dir_outputs = "../inference_outputs"
    os.makedirs(dir_outputs, exist_ok=True)

    while not queue.empty():
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)
        puzzle_ds = arc_test_set.change_keys([key])
        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

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

                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)

                for subkey_id, scored_beams in dfs_result:
                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []

                    if bk not in rescorers:
                        rescorers[bk] = PrefixCachedRescorer(
                            model=model,
                            tokenizer=tokenizer,
                            formatter=formatter,
                            puzzle_ds_multi=puzzle_ds_multi,
                            base_key=bk,
                            max_seq_length=max_seq_length,
                            max_new_tokens=max_new_tokens,
                            seed=hash(bk) % 1024**2,
                        )

                    for beam_score, beam_tokens in scored_beams:
                        array = formatter.convert_tokens_to_array(beam_tokens)
                        if array is None:
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                        grid_id = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            augmented_scores = rescorers[bk].score_solution(solution)
                            known_scores[grid_id] = augmented_scores

                        decoded_result.append(
                            {
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            }
                        )

                    if len(decoded_result):
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
