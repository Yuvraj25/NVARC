from collections import Counter
import unittest

from nvarc_continued_pretraining import (
    SOURCE_WEIGHTS,
    WeightedNVARCCorpus,
    build_wall_clock_plan,
    tokenize_all_assistant_outputs,
    weighted_source_schedule,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


class Rows:
    def __init__(self, source, count=7):
        self.rows = [
            {
                "puzzle_name": f"{source}-{index}",
                "messages": [
                    {"role": "user", "content": str(index)},
                    {"role": "assistant", "content": str((index + 1) % 10)},
                ],
            }
            for index in range(count)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def paired_messages(count):
    messages = []
    for index in range(count):
        messages.extend(
            [
                {"role": "user", "content": f"u{index}"},
                {"role": "assistant", "content": f"a{index}"},
            ]
        )
    return messages


class ContinuedPretrainingDataTests(unittest.TestCase):
    def test_source_schedule_has_exact_requested_mix_and_is_deterministic(self):
        first = weighted_source_schedule(seed=123)
        second = weighted_source_schedule(seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(Counter(first), Counter(SOURCE_WEIGHTS))

    def test_every_assistant_output_receives_loss(self):
        tokenizer = CharacterTokenizer()
        tokenized = tokenize_all_assistant_outputs(
            paired_messages(3), tokenizer, max_seq_length=8192
        )
        self.assertEqual(tokenized["assistant_outputs"], 3)
        self.assertEqual(len(tokenized["assistant_token_counts"]), 3)
        self.assertEqual(
            tokenized["supervised_tokens"],
            sum(label != -100 for label in tokenized["labels"]),
        )
        self.assertGreater(tokenized["supervised_tokens"], 0)
        self.assertLess(tokenized["supervised_tokens"], tokenized["sequence_tokens"])

    def test_overlong_record_drops_complete_leading_pairs_only(self):
        tokenizer = CharacterTokenizer()
        two = tokenize_all_assistant_outputs(
            paired_messages(2), tokenizer, max_seq_length=8192
        )
        three_trimmed = tokenize_all_assistant_outputs(
            paired_messages(3), tokenizer, max_seq_length=two["sequence_tokens"]
        )
        self.assertEqual(three_trimmed["dropped_leading_pairs"], 1)
        self.assertEqual(three_trimmed["assistant_outputs"], 2)
        self.assertEqual(three_trimmed["sequence_tokens"], two["sequence_tokens"])

    def test_invalid_role_order_is_rejected(self):
        tokenizer = CharacterTokenizer()
        with self.assertRaisesRegex(ValueError, "expected 'user'"):
            tokenize_all_assistant_outputs(
                [
                    {"role": "assistant", "content": "1"},
                    {"role": "user", "content": "2"},
                ],
                tokenizer,
                max_seq_length=8192,
            )

    def test_lazy_corpus_selection_and_tokenization(self):
        sources = {source: Rows(source) for source in SOURCE_WEIGHTS}
        corpus = WeightedNVARCCorpus(
            sources,
            CharacterTokenizer(),
            max_seq_length=8192,
            virtual_length=200,
            seed=99,
        )
        self.assertEqual(len(corpus), 200)
        selected = [corpus.selection(index).source for index in range(100)]
        self.assertEqual(Counter(selected), Counter(SOURCE_WEIGHTS))
        sample = corpus[17]
        self.assertIn(sample["source"], SOURCE_WEIGHTS)
        self.assertEqual(sample["assistant_outputs"], 1)
        self.assertGreater(sample["supervised_tokens"], 0)

    def test_wall_clock_plan_has_30_70_boundary_and_rotating_intervals(self):
        plan = build_wall_clock_plan(
            started=100.0,
            budget_seconds=11.5 * 3600,
            canon_only_fraction=0.30,
            checkpoint_seconds=3 * 3600,
        )
        self.assertEqual(plan.stage_boundary, 100.0 + 3.45 * 3600)
        self.assertEqual(plan.deadline, 100.0 + 11.5 * 3600)
        self.assertEqual(
            plan.periodic_checkpoints,
            (100.0 + 3 * 3600, 100.0 + 6 * 3600, 100.0 + 9 * 3600),
        )


if __name__ == "__main__":
    unittest.main()
