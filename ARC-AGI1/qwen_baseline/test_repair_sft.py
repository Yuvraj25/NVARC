import unittest

import torch

from repair_mining import format_reply, grid_to_string
from repair_sft import (
    build_training_mixture,
    build_zero_mask_noop_example,
    add_and_initialize_repair_token,
    ordinary_solve_prompt,
    tokenize_completion_only,
)
from train_repair_adapter import summarize_lora_b


def repair_record(index=0):
    solve_prompt = (
        "<|im_start|>user\n01<|im_end|><|im_start|>assistant\n"
    )
    prediction = [[1, 0]]
    gold = [[0, 1]]
    mask = [[1, 1]]
    return {
        "record_type": "repair_failure",
        "anchor_id": f"anchor-{index % 3}",
        "puzzle_id": f"puzzle-{index}",
        "prediction": prediction,
        "input": (
            solve_prompt
            + format_reply(prediction)
            + "<|im_start|>user\n<REPAIR>\n"
            + grid_to_string(mask)
            + "<|im_end|><|im_start|>assistant\n"
        ),
        "reply": format_reply(gold),
    }


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


class FakeArcTokenizer:
    def __init__(self):
        self.vocab = {str(index): index for index in range(10)}
        self.vocab.update({"Ċ": 10, "user": 11, "assistant": 12, "<|endoftext|>": 13,
                           "<|im_start|>": 14, "<|im_end|>": 15})
        self.additional_special_tokens = ["<|im_start|>", "<|im_end|>"]

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, mapping):
        added = 0
        self.additional_special_tokens = list(mapping["additional_special_tokens"])
        for token in self.additional_special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[token] for token in tokens]


class FakeArcModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Embedding(16, 3)
        self.output = torch.nn.Linear(3, 16, bias=False)
        with torch.no_grad():
            self.input.weight.copy_(torch.arange(48).reshape(16, 3))
            self.output.weight.copy_(torch.arange(48, 96).reshape(16, 3))

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output

    def resize_token_embeddings(self, size, mean_resizing=False):
        old_input = self.input.weight.detach().clone()
        old_output = self.output.weight.detach().clone()
        self.input = torch.nn.Embedding(size, 3)
        self.output = torch.nn.Linear(3, size, bias=False)
        with torch.no_grad():
            self.input.weight[: len(old_input)].copy_(old_input)
            self.output.weight[: len(old_output)].copy_(old_output)


class RepairSftTest(unittest.TestCase):
    def test_lora_update_summary_detects_nonzero_b_weights(self):
        model = torch.nn.Module()
        model.lora_B = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            model.lora_B.weight.zero_()
            model.lora_B.weight[1, 0] = 0.25
        summary = summarize_lora_b(model)
        self.assertEqual(summary["tensors"], 1)
        self.assertEqual(summary["elements"], 6)
        self.assertEqual(summary["nonzero_elements"], 1)
        self.assertEqual(summary["max_abs"], 0.25)

    def test_repair_token_uses_structural_embedding_mean(self):
        tokenizer = FakeArcTokenizer()
        model = FakeArcModel()
        expected_input = model.input.weight[[11, 12, 14, 15]].mean(dim=0)
        expected_output = model.output.weight[[11, 12, 14, 15]].mean(dim=0)
        token_id = add_and_initialize_repair_token(model, tokenizer)
        self.assertEqual(token_id, 16)
        self.assertEqual(len(tokenizer), 17)
        torch.testing.assert_close(model.input.weight[token_id], expected_input)
        torch.testing.assert_close(model.output.weight[token_id], expected_output)

    def test_recovers_ordinary_prompt_without_wrong_candidate(self):
        record = repair_record()
        prompt = ordinary_solve_prompt(record)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<REPAIR>", prompt)
        self.assertNotIn(format_reply(record["prediction"]), prompt)

    def test_zero_mask_noop_uses_gold_as_candidate_and_target(self):
        record = repair_record()
        noop = build_zero_mask_noop_example(record)
        self.assertEqual(noop["record_type"], "repair_noop")
        self.assertIn(format_reply([[0, 1]]), noop["input"])
        self.assertIn("<REPAIR>\n00<|im_end|>", noop["input"])
        self.assertEqual(noop["reply"], record["reply"])

    def test_mixture_targets_84_15_1_and_is_reproducible(self):
        records = [repair_record(index) for index in range(84)]
        first, first_manifest = build_training_mixture(records)
        second, second_manifest = build_training_mixture(records)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["counts"], {
            "repair_failure": 84,
            "repair_noop": 1,
            "solve_replay": 15,
        })

    def test_completion_only_masks_entire_prompt(self):
        example = {"input": "abc", "reply": "de"}
        tokenized = tokenize_completion_only(example, FakeTokenizer(), max_seq_length=5)
        self.assertEqual(tokenized["input_ids"], [97, 98, 99, 100, 101])
        self.assertEqual(tokenized["labels"], [-100, -100, -100, 100, 101])

    def test_completion_only_rejects_overlong_sequences(self):
        with self.assertRaises(ValueError):
            tokenize_completion_only(
                {"input": "abc", "reply": "de"},
                FakeTokenizer(),
                max_seq_length=4,
            )


if __name__ == "__main__":
    unittest.main()
