from arc_loader import ArcDataset
from arc_solver import _build_eval_batches, _collator_model_features, _descending_nll_order


class _FakeTokenizer:
    def encode(self, text):
        return list(text)


class _FakeDataset:
    def __init__(self, lengths):
        self.keys = list(lengths)
        self.lengths = lengths

    def get(self, key, formatter):
        return {"input": "x" * self.lengths[key]}


def test_non_16_view_batches_group_equal_token_lengths():
    lengths = {
        **{f"task_0.view{i}": 100 for i in range(12)},
        **{f"task_0.rot90.view{i}": 104 for i in range(12)},
    }
    dataset = _FakeDataset(lengths)

    batches = _build_eval_batches(
        dataset,
        tokenizer=_FakeTokenizer(),
        formatter=object(),
    )

    flattened = [key for batch in batches for key in batch]
    assert sorted(flattened) == sorted(dataset.keys)
    assert len(flattened) == len(set(flattened))
    assert all(1 <= len(batch) <= 4 for batch in batches)
    assert all(len({lengths[key] for key in batch}) == 1 for batch in batches)


def test_non_16_view_batches_require_tokenizer_aware_grouping():
    dataset = _FakeDataset({f"task_0.view{i}": 100 + i % 2 for i in range(24)})

    try:
        _build_eval_batches(dataset)
    except ValueError as error:
        assert "tokenizer-aware batching" in str(error)
    else:
        raise AssertionError("Expected non-16-view batching to reject missing tokenizer")


def test_non_16_view_batches_support_batch_eight_without_mixing_lengths():
    lengths = {
        **{f"task_0.view{i}": 100 for i in range(12)},
        **{f"task_0.rot90.view{i}": 104 for i in range(12)},
    }
    dataset = _FakeDataset(lengths)

    batches = _build_eval_batches(
        dataset,
        tokenizer=_FakeTokenizer(),
        formatter=object(),
        batch_size=8,
    )

    flattened = [key for batch in batches for key in batch]
    assert sorted(flattened) == sorted(dataset.keys)
    assert [len(batch) for batch in batches] == [8, 4, 8, 4]
    assert all(len({lengths[key] for key in batch}) == 1 for batch in batches)


def test_descending_nll_order_is_stable_for_ties():
    assert _descending_nll_order([0.5, 1.25, 1.25, 0.75]) == [1, 2, 3, 0]


def test_collator_model_features_removes_dataset_metadata():
    tokenizer = _FakeTokenizer()
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    row = {
        "key": "task_0.view0",
        "text": "ignored metadata",
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
    }

    assert _collator_model_features(row, tokenizer) == {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
    }


def test_shared_views_match_descriptors_across_test_outputs():
    dataset = ArcDataset(
        {
            "task": {
                "train": [
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                "test": [{"input": [[5]]}, {"input": [[6]]}],
            }
        },
        keys=["task"],
    )

    shared = dataset.augment(n=3, seed=2).split_multi_replies_shared_views()
    output_0 = {
        key.split(".", 1)[1] for key in shared.keys if key.startswith("task_0.")
    }
    output_1 = {
        key.split(".", 1)[1] for key in shared.keys if key.startswith("task_1.")
    }

    assert len(output_0) == 24
    assert output_0 == output_1
    for descriptor in output_0:
        assert (
            shared.queries[f"task_0.{descriptor}"]["train"]
            == shared.queries[f"task_1.{descriptor}"]["train"]
        )
