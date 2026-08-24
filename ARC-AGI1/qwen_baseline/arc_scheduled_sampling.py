from typing import Any

import torch


DIGIT_TOKEN_IDS = tuple(range(10))
IGNORE_INDEX = -100


def _final_labeled_span(labels: torch.Tensor) -> tuple[int, int] | None:
    """Return [start, end) for the final contiguous completion span."""
    positions = torch.nonzero(labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    position_list = positions.tolist()
    final_group_start = len(position_list) - 1
    while (
        final_group_start > 0
        and position_list[final_group_start - 1] + 1 == position_list[final_group_start]
    ):
        final_group_start -= 1
    return position_list[final_group_start], position_list[-1] + 1


def mix_final_answer_with_restricted_argmax(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    mix_probability: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mix digit tokens in the final assistant answer with teacher-forced argmax.

    The first-pass logit at position t-1 predicts the answer token at position t.
    Newline/EOS and all earlier demonstration answers remain untouched.
    """
    if not 0.0 <= mix_probability <= 1.0:
        raise ValueError(f"mix_probability must be in [0, 1], got {mix_probability}")
    if input_ids.shape != labels.shape:
        raise ValueError(f"input_ids and labels must match, got {input_ids.shape} and {labels.shape}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(f"logits prefix must match input_ids, got {logits.shape} and {input_ids.shape}")
    if logits.shape[-1] < len(DIGIT_TOKEN_IDS):
        raise ValueError(f"logits vocabulary is too small: {logits.shape[-1]}")

    mixed = input_ids.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    eligible_total = 0
    selected_total = 0
    changed_total = 0
    predicted_wrong_total = 0
    restricted_gold_nll_sum = 0.0
    first_changed_offsets = []

    for row in range(input_ids.shape[0]):
        span = _final_labeled_span(labels[row])
        if span is None:
            continue
        start, end = span
        position_tensor = torch.arange(max(start, 1), end, device=labels.device)
        span_gold = labels[row].index_select(0, position_tensor)
        digit_mask = span_gold.ge(DIGIT_TOKEN_IDS[0]) & span_gold.le(DIGIT_TOKEN_IDS[-1])
        position_tensor = position_tensor[digit_mask]
        if position_tensor.numel() == 0:
            continue

        position_tensor = position_tensor.to(device=logits.device, dtype=torch.long)
        previous_positions = position_tensor - 1
        digit_logits = logits[row].index_select(0, previous_positions)[..., : len(DIGIT_TOKEN_IDS)]
        predicted = digit_logits.argmax(dim=-1)
        gold = labels[row].index_select(0, position_tensor)
        wrong = predicted.ne(gold)
        selected_cpu = torch.rand(position_tensor.numel(), generator=generator).lt(mix_probability)
        selected = selected_cpu.to(device=logits.device)
        changed = selected & wrong

        replacement_positions = position_tensor[selected]
        if replacement_positions.numel():
            mixed[row, replacement_positions] = predicted[selected].to(mixed.dtype)

        gold_log_probs = digit_logits.float().log_softmax(dim=-1).gather(1, gold[:, None]).squeeze(1)
        restricted_gold_nll_sum += float((-gold_log_probs).sum().item())
        eligible_total += int(position_tensor.numel())
        selected_total += int(selected.sum().item())
        changed_total += int(changed.sum().item())
        predicted_wrong_total += int(wrong.sum().item())
        changed_indices = torch.nonzero(changed, as_tuple=False).flatten()
        if changed_indices.numel():
            first_changed_offsets.append(
                int(position_tensor[int(changed_indices[0].item())].item()) - start
            )

    diagnostic = {
        "eligible_digit_tokens": eligible_total,
        "selected_digit_tokens": selected_total,
        "changed_digit_tokens": changed_total,
        "predicted_wrong_digit_tokens": predicted_wrong_total,
        "teacher_forced_digit_accuracy": (
            1.0 - predicted_wrong_total / eligible_total if eligible_total else None
        ),
        "realized_change_fraction": changed_total / eligible_total if eligible_total else None,
        "restricted_gold_nll": restricted_gold_nll_sum / eligible_total if eligible_total else None,
        "first_changed_offset": min(first_changed_offsets) if first_changed_offsets else None,
    }
    return mixed, diagnostic


def make_one_pass_scheduled_sampling_trainer_class(base_trainer_class):
    class OnePassScheduledSamplingTrainer(base_trainer_class):
        def __init__(
            self,
            *args,
            scheduled_sampling_warmup_steps: int,
            scheduled_sampling_mix_probability: float,
            scheduled_sampling_seed: int,
            **kwargs,
        ):
            self.scheduled_sampling_warmup_steps = int(scheduled_sampling_warmup_steps)
            self.scheduled_sampling_mix_probability = float(scheduled_sampling_mix_probability)
            self.scheduled_sampling_seed = int(scheduled_sampling_seed)
            self.scheduled_sampling_diagnostics = []
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            step = int(self.state.global_step)
            if step < self.scheduled_sampling_warmup_steps:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

            probe_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            with torch.no_grad():
                probe_outputs = model(**probe_inputs)
                mixed_input_ids, diagnostic = mix_final_answer_with_restricted_argmax(
                    input_ids=inputs["input_ids"],
                    labels=inputs["labels"],
                    logits=probe_outputs.logits,
                    mix_probability=self.scheduled_sampling_mix_probability,
                    seed=self.scheduled_sampling_seed + step,
                )
            del probe_outputs

            diagnostic["step"] = step
            self.scheduled_sampling_diagnostics.append(diagnostic)
            mixed_inputs = dict(inputs)
            mixed_inputs["input_ids"] = mixed_input_ids
            return super().compute_loss(model, mixed_inputs, return_outputs=return_outputs, **kwargs)

        def scheduled_sampling_summary(self) -> dict[str, Any]:
            rows = self.scheduled_sampling_diagnostics
            totals = {
                key: sum(int(row[key]) for row in rows)
                for key in (
                    "eligible_digit_tokens",
                    "selected_digit_tokens",
                    "changed_digit_tokens",
                    "predicted_wrong_digit_tokens",
                )
            }
            eligible = totals["eligible_digit_tokens"]
            nll_values = [row["restricted_gold_nll"] for row in rows if row["restricted_gold_nll"] is not None]
            return {
                "warmup_steps": self.scheduled_sampling_warmup_steps,
                "mix_probability": self.scheduled_sampling_mix_probability,
                "scheduled_sampling_steps": len(rows),
                **totals,
                "teacher_forced_digit_accuracy": (
                    1.0 - totals["predicted_wrong_digit_tokens"] / eligible if eligible else None
                ),
                "realized_change_fraction": totals["changed_digit_tokens"] / eligible if eligible else None,
                "mean_restricted_gold_nll": sum(nll_values) / len(nll_values) if nll_values else None,
                "per_step": rows,
            }

    return OnePassScheduledSamplingTrainer
