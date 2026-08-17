import json
from pathlib import Path

from global_eval_curriculum import (
    build_exact_curriculum_records,
    format_completion_record,
    load_evaluation_training_tasks,
)
from repair_sft import tokenize_completion_only


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


def example_tasks():
    pair = lambda value: {"input": [[value]], "output": [[(value + 1) % 10]]}
    return {"b": [pair(0), pair(1), pair(2)], "a": [pair(3), pair(4)]}


def test_load_discards_test_pairs(tmp_path: Path):
    path = tmp_path / "challenges.json"
    path.write_text(
        json.dumps(
            {
                "task": {
                    "train": [
                        {"input": [[0]], "output": [[1]]},
                        {"input": [[2]], "output": [[3]]},
                    ],
                    "test": [{"input": [[9]]}],
                }
            }
        )
    )
    assert load_evaluation_training_tasks(path) == {
        "task": [
            {"input": [[0]], "output": [[1]]},
            {"input": [[2]], "output": [[3]]},
        ]
    }


def test_load_skips_one_pair_task_only_from_global_curriculum(tmp_path: Path):
    path = tmp_path / "challenges.json"
    path.write_text(
        json.dumps(
            {
                "one_pair": {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]]}],
                },
                "two_pairs": {
                    "train": [
                        {"input": [[3]], "output": [[4]]},
                        {"input": [[5]], "output": [[6]]},
                    ],
                    "test": [{"input": [[7]]}],
                },
            }
        )
    )
    assert list(load_evaluation_training_tasks(path)) == ["two_pairs"]


def test_exact_count_balanced_targets_and_determinism():
    first = build_exact_curriculum_records(example_tasks(), views_per_task=20)
    second = build_exact_curriculum_records(example_tasks(), views_per_task=20)
    assert first == second
    assert len(first) == 40
    assert [record["task_id"] for record in first[:20]] == ["a"] * 20
    for task_id, pair_count in [("a", 2), ("b", 3)]:
        counts = [
            sum(record["task_id"] == task_id and record["target_index"] == index for record in first)
            for index in range(pair_count)
        ]
        assert max(counts) - min(counts) <= 1


def test_only_final_reply_receives_loss():
    tokenizer = CharacterTokenizer()
    record = build_exact_curriculum_records({"task": example_tasks()["b"]}, views_per_task=1)[0]
    formatted = format_completion_record(record, tokenizer, max_seq_length=8192)
    tokenized = tokenize_completion_only(formatted, tokenizer, max_seq_length=8192)
    prompt_length = len(tokenizer.encode(formatted["input"]))
    assert all(label == -100 for label in tokenized["labels"][:prompt_length])
    assert tokenized["labels"][prompt_length:] == tokenizer.encode(formatted["reply"])
    # Demonstration outputs are visible inside input_ids despite being unlabeled.
    assert "assistant" in formatted["input"]


def test_overlong_prompt_drops_earliest_demonstrations():
    tokenizer = CharacterTokenizer()
    record = build_exact_curriculum_records({"task": example_tasks()["b"]}, views_per_task=1)[0]
    one_demo = {**record, "demonstrations": record["demonstrations"][1:]}
    one_demo["demonstration_indices"] = record["demonstration_indices"][1:]
    one_demo_formatted = format_completion_record(one_demo, tokenizer, max_seq_length=8192)
    formatted = format_completion_record(
        record,
        tokenizer,
        max_seq_length=one_demo_formatted["sequence_tokens"],
    )
    assert formatted["dropped_demonstration_indices"] == [record["demonstration_indices"][0]]
