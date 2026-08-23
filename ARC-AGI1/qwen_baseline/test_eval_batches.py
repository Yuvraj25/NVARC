from arc_solver import _build_eval_batches


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
