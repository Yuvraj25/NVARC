import argparse
import bz2
import gc
import hashlib
import importlib
import json
import os
import pickle
import sys
from typing import Optional

import torch
from unsloth import FastLanguageModel
from peft import PeftModel

from arc_loader import ArcDataset, QwenFormatter
from arc_search import ARC_TOKENS, EOS_ID
from arc_sglang import ArcSglangBackend, SglangConfig


MAX_SEQ_LENGTH = 8192
ARC_TOKEN_NAMES = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "\\n",
    15: "<|im_end|>",
}


def token_name(token_id):
    return ARC_TOKEN_NAMES.get(token_id, str(token_id))


def module_path(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"error": repr(exc)}
    return {
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
    }


def print_module_paths():
    print("MODULE_PATHS", json.dumps(
        {
            name: module_path(name)
            for name in [
                "torch",
                "transformers",
                "tokenizers",
                "sglang",
                "sklearn",
                "numpy",
                "unsloth",
                "peft",
            ]
        },
        sort_keys=True,
    ))
    print("SYS_PATH_HEAD", json.dumps(sys.path[:12]))
    print("PYTHONPATH", os.environ.get("PYTHONPATH", ""))


def iter_token_logprobs(row):
    if row is None:
        return
    if (
        len(row) in (2, 3)
        and isinstance(row[0], (list, tuple))
        and isinstance(row[1], (list, tuple))
        and all(isinstance(token_id, int) for token_id in row[1])
    ):
        for logprob, token_id in zip(row[0], row[1]):
            yield float(logprob), int(token_id)
        return
    for item in row:
        logprob, token_id = item[:2]
        yield float(logprob), int(token_id)


def fmt_top(logprobs, limit):
    top = sorted(logprobs.items(), key=lambda item: item[1], reverse=True)[:limit]
    return " ".join(f"{token_name(token_id)}:{logprob:.6f}" for token_id, logprob in top)


def max_abs_diff(left, right):
    rows = []
    for token_id in ARC_TOKENS:
        rows.append((abs(left[token_id] - right[token_id]), token_id, left[token_id], right[token_id]))
    return max(rows)


def load_hf_model_and_tokenizer(model_path, adapter_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = FastLanguageModel.for_inference(model)
    model.eval()
    return model, tokenizer


def build_eval_prompt(tokenizer, test_path, puzzle_key, subkey):
    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    arc_test_set = ArcDataset.from_file(test_path)
    puzzle_ds = arc_test_set.change_keys([puzzle_key])
    puzzle_ds_multi = puzzle_ds.split_multi_replies()
    eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(
        formatter=formatter,
        name="input",
        max_len=MAX_SEQ_LENGTH - max_new_tokens,
    )
    subkeys = sorted(eval_ds.keys)
    if subkey is None:
        subkey = subkeys[0]
    if subkey not in eval_ds.keys:
        raise ValueError(f"Unknown subkey {subkey!r}. Available: {subkeys}")
    prompt_text = eval_ds.get(subkey, formatter)["input"]
    prompt_token_ids = tokenizer.encode(prompt_text)
    return formatter, subkeys, subkey, prompt_text, prompt_token_ids


def hf_next_arc_logprobs(model, prefix_token_ids):
    device = next(model.parameters()).device
    input_ids = torch.tensor([prefix_token_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
        logprobs = outputs.logits[0, -1].float().log_softmax(-1).detach().cpu()
    result = {token_id: float(logprobs[token_id].item()) for token_id in ARC_TOKENS}
    del outputs
    del input_ids
    del logprobs
    return result


def sglang_next_arc_logprobs(backend, prefix_token_ids):
    return backend.next_arc_logprobs([prefix_token_ids])[0]


def load_answer_tokens_from_vanilla_output(path, formatter, tokenizer, subkey):
    with bz2.BZ2File(path, "rb") as f:
        rows = pickle.load(f)
    if not rows:
        raise RuntimeError(f"No vanilla candidates in {path}")
    solution = min(rows, key=lambda row: row["beam_score"])["solution"]
    augmented_solution = ArcDataset.forward_mod(solution.tolist(), subkey)
    answer_text = formatter.fmt_reply([augmented_solution])
    return tokenizer.encode(answer_text), answer_text


def compare_step(label, hf_logprobs, sg_logprobs, top_k, target_token_id=None):
    diff_abs, diff_token_id, hf_value, sg_value = max_abs_diff(hf_logprobs, sg_logprobs)
    fields = {
        "label": label,
        "max_abs_diff": diff_abs,
        "max_diff_token": token_name(diff_token_id),
        "hf_at_max": hf_value,
        "sg_at_max": sg_value,
        "hf_top": fmt_top(hf_logprobs, top_k),
        "sg_top": fmt_top(sg_logprobs, top_k),
    }
    if target_token_id is not None:
        fields.update(
            {
                "target_token": token_name(target_token_id),
                "hf_target": hf_logprobs[target_token_id],
                "sg_target": sg_logprobs[target_token_id],
                "target_delta_sg_minus_hf": sg_logprobs[target_token_id] - hf_logprobs[target_token_id],
            }
        )
    print("STEP_COMPARE", json.dumps(fields, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--puzzle-key", default="136b0064")
    parser.add_argument("--subkey", default=None)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--vanilla-output-path", default=None)
    parser.add_argument("--sglang-tp-size", type=int, default=1)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=None)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {args.adapter_path}")
    if args.vanilla_output_path and not os.path.exists(args.vanilla_output_path):
        raise FileNotFoundError(f"Vanilla output path does not exist: {args.vanilla_output_path}")

    print_module_paths()
    print("PROBE_ARGS", json.dumps(vars(args), sort_keys=True))

    hf_model, tokenizer = load_hf_model_and_tokenizer(args.model_path, args.adapter_path)
    formatter, subkeys, subkey, prompt_text, prompt_token_ids = build_eval_prompt(
        tokenizer=tokenizer,
        test_path=args.test_path,
        puzzle_key=args.puzzle_key,
        subkey=args.subkey,
    )
    print("EVAL_SUBKEYS", json.dumps(subkeys))
    print("SELECTED_SUBKEY", subkey)
    print("TOKENIZER", json.dumps(module_path(type(tokenizer).__module__)))
    print(
        "PROMPT",
        json.dumps(
            {
                "text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "text_len": len(prompt_text),
                "token_count": len(prompt_token_ids),
                "token_sha256": hashlib.sha256(json.dumps(prompt_token_ids).encode("utf-8")).hexdigest(),
                "tail_ids": prompt_token_ids[-32:],
                "eos_id": EOS_ID,
                "arc_tokens": ARC_TOKENS,
            },
            sort_keys=True,
        ),
    )

    backend = ArcSglangBackend(
        SglangConfig(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            tensor_parallel_size=args.sglang_tp_size,
            mem_fraction_static=args.sglang_mem_fraction_static,
            max_model_len=MAX_SEQ_LENGTH,
        )
    )
    try:
        hf_first = hf_next_arc_logprobs(hf_model, prompt_token_ids)
        sg_first = sglang_next_arc_logprobs(backend, prompt_token_ids)
        compare_step("first", hf_first, sg_first, args.top_k)

        if args.vanilla_output_path and args.trace_steps > 0:
            answer_token_ids, answer_text = load_answer_tokens_from_vanilla_output(
                args.vanilla_output_path,
                formatter,
                tokenizer,
                subkey,
            )
            print(
                "ANSWER",
                json.dumps(
                    {
                        "token_count": len(answer_token_ids),
                        "text_sha256": hashlib.sha256(answer_text.encode("utf-8")).hexdigest(),
                        "tail_ids": answer_token_ids[-32:],
                    },
                    sort_keys=True,
                ),
            )
            limit = min(args.trace_steps, len(answer_token_ids))
            hf_nll = 0.0
            sg_nll = 0.0
            for step in range(limit):
                prefix = prompt_token_ids + answer_token_ids[:step]
                target_token_id = answer_token_ids[step]
                hf_row = hf_next_arc_logprobs(hf_model, prefix)
                sg_row = sglang_next_arc_logprobs(backend, prefix)
                hf_nll += -hf_row[target_token_id]
                sg_nll += -sg_row[target_token_id]
                compare_step(f"teacher_forced_{step}", hf_row, sg_row, args.top_k, target_token_id)
                print(
                    "TRACE_NLL",
                    json.dumps(
                        {
                            "step": step,
                            "target_token": token_name(target_token_id),
                            "hf_cumulative_nll": hf_nll,
                            "sg_cumulative_nll": sg_nll,
                            "delta_sg_minus_hf": sg_nll - hf_nll,
                        },
                        sort_keys=True,
                    ),
                )
    finally:
        backend.close()
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
