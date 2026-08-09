import json

from arc_loader import ArcDataset


def example_order_from_descriptor(descriptor: str, num_examples: int):
    part = next((part for part in descriptor.split(".") if part.startswith("ex")), None)
    if part is None:
        return list(range(num_examples))
    encoded = part[2:]
    values = encoded.split("-") if "-" in encoded else list(encoded)
    order = [int(value) for value in values if value]
    if sorted(order) != list(range(num_examples)):
        raise ValueError(
            f"Descriptor {descriptor!r} has invalid example order for {num_examples} examples"
        )
    return order


def apply_selected_augmentations(puzzle_ds_multi, descriptors):
    queries = {}
    keys = []
    for basekey in puzzle_ds_multi.keys:
        source = puzzle_ds_multi.queries[basekey]
        for descriptor in descriptors:
            subkey = f"{basekey}.{descriptor}" if descriptor else basekey
            order = example_order_from_descriptor(descriptor, len(source["train"]))
            transformed_train = []
            for pair in source["train"]:
                transformed_train.append(
                    {
                        field: ArcDataset.forward_mod(value, subkey)
                        for field, value in pair.items()
                    }
                )
            transformed_test = [
                {
                    field: ArcDataset.forward_mod(value, subkey)
                    for field, value in pair.items()
                }
                for pair in source["test"]
            ]
            keys.append(subkey)
            queries[subkey] = {
                "train": [transformed_train[index] for index in order],
                "test": transformed_test,
            }
    return ArcDataset(queries=queries, replies={}, keys=keys)


def prepare_selected_eval_ds(
    puzzle_ds_multi,
    descriptors,
    formatter,
    max_seq_length: int,
    max_new_tokens: int,
):
    eval_ds = apply_selected_augmentations(puzzle_ds_multi, descriptors)
    return eval_ds.cut_to_len(
        formatter=formatter,
        name="input",
        max_len=max_seq_length - max_new_tokens,
    )


def load_selected_augmentations(path: str, puzzle_key: str):
    with open(path) as handle:
        selected = json.load(handle)
    descriptors = selected.get(puzzle_key)
    if descriptors is None:
        raise KeyError(f"No selected augmentations for {puzzle_key} in {path}")
    if not isinstance(descriptors, list) or not all(isinstance(value, str) for value in descriptors):
        raise ValueError(f"Selected augmentations for {puzzle_key} must be a list of strings")
    if len(descriptors) != len(set(descriptors)):
        raise ValueError(f"Duplicate selected augmentations for {puzzle_key}: {descriptors}")
    return descriptors
