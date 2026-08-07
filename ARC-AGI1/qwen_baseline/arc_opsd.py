import copy
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from arc_loader import ArcDataset, QwenFormatter
from arc_search import ARC_TOKENS, EOS_ID, PAD_ID


TEACHER_ADAPTER_NAME = "opsd_teacher"


@dataclass
class OpsdPairSplit:
    puzzle_key: str
    reserved_pair_index: int
    sft_pair_indices: list[int]
    reduced_dataset: ArcDataset
    reserved_pair: dict[str, Any]


@dataclass
class OpsdExample:
    h_key: str
    g_key: str
    is_cross_view: bool
    student_prompt: str
    teacher_prompt: str
    gold_reply: str
    transformed_input: list[list[int]]
    transformed_output: list[list[int]]
    privileged_input: list[list[int]]
    privileged_output: list[list[int]]


def deterministic_reserved_pair_index(puzzle_key: str, num_train_pairs: int) -> int:
    if num_train_pairs < 1:
        raise ValueError("A puzzle must contain at least one training pair")
    # Keep the rule simple and stable across Python processes and versions.
    seed = sum((offset + 1) * ord(char) for offset, char in enumerate(puzzle_key))
    return seed % num_train_pairs


def split_puzzle_for_opsd(
    puzzle_ds: ArcDataset,
    puzzle_key: str,
    reserved_pair_index: Optional[int] = None,
) -> OpsdPairSplit:
    if puzzle_key not in puzzle_ds.queries:
        raise KeyError(f"Unknown puzzle key: {puzzle_key}")
    query = copy.deepcopy(puzzle_ds.queries[puzzle_key])
    train = query["train"]
    if len(train) < 3:
        raise ValueError(f"OPSD requires at least 3 training pairs, got {len(train)} for {puzzle_key}")
    if reserved_pair_index is None:
        reserved_pair_index = deterministic_reserved_pair_index(puzzle_key, len(train))
    if not 0 <= reserved_pair_index < len(train):
        raise IndexError(f"reserved_pair_index={reserved_pair_index} is invalid for {len(train)} pairs")

    reserved_pair = copy.deepcopy(train[reserved_pair_index])
    sft_pair_indices = [index for index in range(len(train)) if index != reserved_pair_index]
    query["train"] = [copy.deepcopy(train[index]) for index in sft_pair_indices]
    reduced_dataset = ArcDataset(
        queries={puzzle_key: query},
        replies={},
        keys=[puzzle_key],
        is_orig=puzzle_ds.is_orig,
    )
    return OpsdPairSplit(
        puzzle_key=puzzle_key,
        reserved_pair_index=reserved_pair_index,
        sft_pair_indices=sft_pair_indices,
        reduced_dataset=reduced_dataset,
        reserved_pair=reserved_pair,
    )


def _transformed_reserved_pair(view_ds: ArcDataset, view_key: str) -> dict[str, Any]:
    return {
        "input": copy.deepcopy(view_ds.queries[view_key]["test"][0]["input"]),
        "output": copy.deepcopy(view_ds.replies[view_key][0]),
    }


def _same_transformed_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return np.array_equal(left["input"], right["input"]) and np.array_equal(left["output"], right["output"])


def build_opsd_examples(
    split: OpsdPairSplit,
    formatter: QwenFormatter,
    color_permutations: int = 2,
    cross_view_probability: float = 0.2,
    seed: int = 42,
) -> list[OpsdExample]:
    if color_permutations < 1:
        raise ValueError("color_permutations must be positive")
    if not 0.0 <= cross_view_probability <= 1.0:
        raise ValueError("cross_view_probability must be in [0, 1]")

    source = ArcDataset(
        queries={
            split.puzzle_key: {
                "train": copy.deepcopy(split.reduced_dataset.queries[split.puzzle_key]["train"]),
                "test": [{"input": copy.deepcopy(split.reserved_pair["input"])}],
            }
        },
        replies={split.puzzle_key: [copy.deepcopy(split.reserved_pair["output"])]},
        keys=[split.puzzle_key],
    )
    views = source.augment(n=color_permutations, shfl_keys=False, seed=seed)
    view_keys = list(views.keys)
    transformed_pairs = {key: _transformed_reserved_pair(views, key) for key in view_keys}
    rng = random.Random(seed)
    examples = []

    for h_key in view_keys:
        want_cross = rng.random() < cross_view_probability
        g_key = h_key
        if want_cross:
            alternatives = [
                key
                for key in view_keys
                if key != h_key and not _same_transformed_pair(transformed_pairs[key], transformed_pairs[h_key])
            ]
            if not alternatives:
                continue
            g_key = alternatives[rng.randrange(len(alternatives))]

        h_sample = views.get(h_key, formatter)
        g_pair = transformed_pairs[g_key]
        privileged_demo = formatter.fmt_train([{"input": g_pair["input"], "output": g_pair["output"]}])
        examples.append(
            OpsdExample(
                h_key=h_key,
                g_key=g_key,
                is_cross_view=(g_key != h_key),
                student_prompt=h_sample["input"],
                teacher_prompt=privileged_demo + h_sample["input"],
                gold_reply=h_sample["reply"],
                transformed_input=copy.deepcopy(views.queries[h_key]["test"][0]["input"]),
                transformed_output=copy.deepcopy(views.replies[h_key][0]),
                privileged_input=copy.deepcopy(g_pair["input"]),
                privileged_output=copy.deepcopy(g_pair["output"]),
            )
        )
    return examples


def exact_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"Student and teacher logits must align, got {student_logits.shape} and {teacher_logits.shape}")
    student_log_prob = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_prob = student_log_prob.exp()
    per_position = (student_prob * (student_log_prob - teacher_log_prob)).sum(dim=-1)
    return per_position.mean(), per_position


def _completion_logits(model, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    if not prompt_ids or not completion_ids:
        raise ValueError("Both prompt_ids and completion_ids must be non-empty")
    input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    start = len(prompt_ids) - 1
    return outputs.logits[0, start : start + len(completion_ids)]


def _gold_metrics(logits: torch.Tensor, gold_ids: list[int]) -> dict[str, Any]:
    targets = torch.tensor(gold_ids, device=logits.device, dtype=torch.long)
    log_prob = torch.log_softmax(logits.float(), dim=-1)
    token_nll = -log_prob[torch.arange(len(gold_ids), device=logits.device), targets]
    legal_ids = torch.tensor(ARC_TOKENS, device=logits.device, dtype=torch.long)
    legal_argmax = legal_ids[logits[:, legal_ids].argmax(dim=-1)]
    return {
        "nll": float(token_nll.sum().detach().cpu()),
        "mean_nll": float(token_nll.mean().detach().cpu()),
        "teacher_forced_greedy_exact": bool(torch.equal(legal_argmax, targets)),
    }


def _adapter_parameter_matches(name: str, adapter_name: str) -> bool:
    return f".{adapter_name}." in name or name.endswith(f".{adapter_name}")


def activate_adapter(model, adapter_name: str, trainable: bool) -> list[torch.nn.Parameter]:
    model.set_adapter(adapter_name)
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        if trainable and _adapter_parameter_matches(name, adapter_name):
            parameter.requires_grad = True
            selected.append(parameter)
    if trainable and not selected:
        raise RuntimeError(f"No trainable parameters found for adapter {adapter_name!r}")
    return selected


def clone_frozen_teacher_adapter(
    model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    student_adapter_name: str = "default",
    teacher_adapter_name: str = TEACHER_ADAPTER_NAME,
) -> None:
    if teacher_adapter_name in model.peft_config:
        raise RuntimeError(f"Adapter {teacher_adapter_name!r} already exists; puzzle reset is incomplete")
    teacher_config = copy.deepcopy(model.peft_config[student_adapter_name])
    model.add_adapter(teacher_adapter_name, teacher_config)
    student_state = get_peft_model_state_dict(model, adapter_name=student_adapter_name)
    result = set_peft_model_state_dict(model, student_state, adapter_name=teacher_adapter_name)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"Unexpected keys while cloning teacher adapter: {unexpected[:5]}")
    teacher_state = get_peft_model_state_dict(model, adapter_name=teacher_adapter_name)
    if student_state.keys() != teacher_state.keys():
        raise RuntimeError("Student and cloned teacher adapter state keys do not match")
    mismatched = [
        key
        for key in student_state
        if not torch.equal(student_state[key].detach(), teacher_state[key].detach())
    ]
    if mismatched:
        raise RuntimeError(f"Cloned teacher adapter weights do not match the student: {mismatched[:5]}")
    activate_adapter(model, teacher_adapter_name, trainable=False)


def remove_teacher_adapter(model, teacher_adapter_name: str = TEACHER_ADAPTER_NAME) -> None:
    if teacher_adapter_name in getattr(model, "peft_config", {}):
        model.delete_adapter(teacher_adapter_name)


class _ArcOnlyLogitsProcessor:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        masked = torch.full_like(scores, -torch.inf)
        masked[:, ARC_TOKENS] = scores[:, ARC_TOKENS]
        return masked


@torch.no_grad()
def _student_rollout(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> list[int]:
    from transformers import LogitsProcessorList

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor([prompt_ids], device=model.device, dtype=torch.long)
    generated = model.generate(
        input_ids=input_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        logits_processor=LogitsProcessorList([_ArcOnlyLogitsProcessor()]),
        use_cache=True,
    )
    return generated[0, len(prompt_ids) :].tolist()


def _first_divergence(rollout_ids: list[int], gold_ids: list[int]) -> Optional[int]:
    for index, (rollout_token, gold_token) in enumerate(zip(rollout_ids, gold_ids)):
        if rollout_token != gold_token:
            return index
    if len(rollout_ids) != len(gold_ids):
        return min(len(rollout_ids), len(gold_ids))
    return None


def _region_kl(per_position: torch.Tensor, first_divergence: Optional[int]) -> dict[str, Optional[float]]:
    values = per_position.detach().float().cpu()
    if first_divergence is None:
        return {
            "kl_before_divergence": float(values.mean()) if len(values) else None,
            "kl_at_divergence": None,
            "kl_after_divergence": None,
        }

    def mean_or_none(region):
        return float(region.mean()) if len(region) else None

    return {
        "kl_before_divergence": mean_or_none(values[:first_divergence]),
        "kl_at_divergence": float(values[first_divergence]) if first_divergence < len(values) else None,
        "kl_after_divergence": mean_or_none(values[first_divergence + 1 :]),
    }


def _illegal_probability_mass(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return float((1.0 - probabilities[:, ARC_TOKENS].sum(dim=-1)).mean().detach().cpu())


def run_opsd_correction(
    model,
    tokenizer,
    formatter: QwenFormatter,
    examples: list[OpsdExample],
    max_seq_length: int,
    max_updates: int = 16,
    learning_rate: float = 5e-5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    lambda_ce: float = 0.0,
    seed: int = 42,
    student_adapter_name: str = "default",
    teacher_adapter_name: str = TEACHER_ADAPTER_NAME,
) -> dict[str, Any]:
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    if lambda_ce < 0.0:
        raise ValueError("lambda_ce must be non-negative")

    trainable_parameters = activate_adapter(model, student_adapter_name, trainable=True)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.0)
    model.train()
    logs = []
    accepted = 0
    started_at = time.perf_counter()

    for attempt, example in enumerate(examples):
        if accepted >= max_updates:
            break
        prompt_student_ids = tokenizer.encode(example.student_prompt)
        prompt_teacher_ids = tokenizer.encode(example.teacher_prompt)
        gold_ids = tokenizer.encode(example.gold_reply)
        if max(len(prompt_student_ids), len(prompt_teacher_ids)) + len(gold_ids) > max_seq_length:
            logs.append(
                {
                    "attempt": attempt,
                    "accepted": False,
                    "reason": "sequence_too_long",
                    "h_key": example.h_key,
                    "g_key": example.g_key,
                    "is_cross_view": example.is_cross_view,
                    "student_prompt_tokens": len(prompt_student_ids),
                    "teacher_prompt_tokens": len(prompt_teacher_ids),
                    "gold_tokens": len(gold_ids),
                }
            )
            continue

        activate_adapter(model, student_adapter_name, trainable=False)
        with torch.no_grad():
            student_gold_logits = _completion_logits(model, prompt_student_ids, gold_ids)
            student_gold = _gold_metrics(student_gold_logits, gold_ids)
        activate_adapter(model, teacher_adapter_name, trainable=False)
        with torch.no_grad():
            teacher_gold_logits = _completion_logits(model, prompt_teacher_ids, gold_ids)
            teacher_gold = _gold_metrics(teacher_gold_logits, gold_ids)

        gate_passed = teacher_gold["nll"] < student_gold["nll"]
        if example.is_cross_view:
            gate_passed = gate_passed and teacher_gold["teacher_forced_greedy_exact"]
        base_log = {
            "attempt": attempt,
            "h_key": example.h_key,
            "g_key": example.g_key,
            "is_cross_view": example.is_cross_view,
            "student_prompt_tokens": len(prompt_student_ids),
            "teacher_prompt_tokens": len(prompt_teacher_ids),
            "gold_tokens": len(gold_ids),
            "student_gold_nll": student_gold["nll"],
            "teacher_gold_nll": teacher_gold["nll"],
            "student_gold_mean_nll": student_gold["mean_nll"],
            "teacher_gold_mean_nll": teacher_gold["mean_nll"],
            "student_greedy_exact": student_gold["teacher_forced_greedy_exact"],
            "teacher_greedy_exact": teacher_gold["teacher_forced_greedy_exact"],
            "teacher_minus_student_nll": teacher_gold["nll"] - student_gold["nll"],
        }
        if not gate_passed:
            logs.append({**base_log, "accepted": False, "reason": "teacher_advantage_gate"})
            continue

        activate_adapter(model, student_adapter_name, trainable=False)
        model.eval()
        rollout_started_at = time.perf_counter()
        rollout_ids = _student_rollout(
            model,
            prompt_ids=prompt_student_ids,
            max_new_tokens=len(gold_ids),
            seed=seed + attempt,
            temperature=temperature,
            top_p=top_p,
        )
        rollout_time = time.perf_counter() - rollout_started_at
        if not rollout_ids:
            logs.append({**base_log, "accepted": False, "reason": "empty_rollout", "rollout_time_s": rollout_time})
            continue

        teacher_started_at = time.perf_counter()
        activate_adapter(model, teacher_adapter_name, trainable=False)
        with torch.no_grad():
            teacher_logits = _completion_logits(model, prompt_teacher_ids, rollout_ids).detach()
        teacher_time = time.perf_counter() - teacher_started_at

        student_started_at = time.perf_counter()
        activate_adapter(model, student_adapter_name, trainable=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        student_logits = _completion_logits(model, prompt_student_ids, rollout_ids)
        opsd_loss, per_position_kl = exact_reverse_kl(student_logits, teacher_logits)
        total_loss = opsd_loss
        gold_ce_value = 0.0
        if lambda_ce:
            gold_logits = _completion_logits(model, prompt_student_ids, gold_ids)
            targets = torch.tensor(gold_ids, device=gold_logits.device, dtype=torch.long)
            gold_ce = torch.nn.functional.cross_entropy(gold_logits.float(), targets)
            total_loss = total_loss + lambda_ce * gold_ce
            gold_ce_value = float(gold_ce.detach().cpu())
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        backward_time = time.perf_counter() - student_started_at

        first_divergence = _first_divergence(rollout_ids, gold_ids)
        student_log_prob = torch.log_softmax(student_logits.detach().float(), dim=-1)
        teacher_log_prob = torch.log_softmax(teacher_logits.float(), dim=-1)
        max_log_ratio = float((student_log_prob - teacher_log_prob).max().cpu())
        decoded_grid = formatter.convert_tokens_to_array(rollout_ids)
        accepted += 1
        logs.append(
            {
                **base_log,
                "accepted": True,
                "optimizer_step": accepted,
                "rollout_token_ids": rollout_ids,
                "rollout_valid_grid": decoded_grid is not None,
                "rollout_has_eos": rollout_ids[-1] == EOS_ID,
                "rollout_truncated": rollout_ids[-1] != EOS_ID and len(rollout_ids) == len(gold_ids),
                "first_divergence": first_divergence,
                "opsd_kl": float(opsd_loss.detach().cpu()),
                "gold_ce": gold_ce_value,
                "total_loss": float(total_loss.detach().cpu()),
                "max_student_teacher_log_ratio": max_log_ratio,
                "student_illegal_probability_mass": _illegal_probability_mass(student_logits.detach()),
                "teacher_illegal_probability_mass": _illegal_probability_mass(teacher_logits),
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "rollout_time_s": rollout_time,
                "teacher_scoring_time_s": teacher_time,
                "student_backward_time_s": backward_time,
                **_region_kl(per_position_kl, first_divergence),
            }
        )

    activate_adapter(model, student_adapter_name, trainable=False)
    return {
        "attempted_examples": len(logs),
        "accepted_updates": accepted,
        "max_updates": max_updates,
        "wall_time_s": time.perf_counter() - started_at,
        "examples": logs,
    }
