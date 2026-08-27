import unittest

import torch

from arc_canon import CanonACState, ResidualCanon1d


class ResidualCanon1dTests(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self):
        layer = ResidualCanon1d(3, kernel_size=4, zero_init=True)
        inputs = torch.randn(2, 7, 3)
        torch.testing.assert_close(layer(inputs), inputs, rtol=0, atol=0)

    def test_boundary_uses_only_available_predecessors(self):
        layer = ResidualCanon1d(1, kernel_size=4, zero_init=True)
        with torch.no_grad():
            # Current, t-1, t-2, t-3.
            layer.weight[:, 0] = torch.tensor([1.0, 10.0, 100.0, 1000.0])
        inputs = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        # Includes the explicit residual in addition to the Canon mixture.
        expected = torch.tensor([[[2.0], [14.0], [126.0], [1238.0]]])
        torch.testing.assert_close(layer(inputs), expected)

    def test_full_sequence_matches_incremental_steps(self):
        torch.manual_seed(3)
        layer = ResidualCanon1d(5, kernel_size=4, zero_init=False)
        inputs = torch.randn(2, 9, 5)
        full = layer(inputs)
        state = layer.initial_state(2, dtype=inputs.dtype, device=inputs.device)
        pieces = []
        for index in range(inputs.shape[1]):
            output, state = layer.step(inputs[:, index : index + 1], state)
            pieces.append(output)
        incremental = torch.cat(pieces, dim=1)
        torch.testing.assert_close(incremental, full)

    def test_sequence_boundary_blocks_previous_record(self):
        layer = ResidualCanon1d(1, kernel_size=2, zero_init=True)
        with torch.no_grad():
            layer.weight[:, 0] = torch.tensor([0.0, 1.0])
        inputs = torch.tensor([[[2.0], [3.0], [5.0], [7.0]]])
        sequence_ids = torch.tensor([[0, 0, 1, 1]])
        expected = torch.tensor([[[2.0], [5.0], [5.0], [12.0]]])
        torch.testing.assert_close(layer(inputs, sequence_ids=sequence_ids), expected)

    def test_sibling_steps_do_not_mutate_parent_state(self):
        layer = ResidualCanon1d(2, kernel_size=3, zero_init=False)
        parent = layer.initial_state(1, dtype=torch.float32, device="cpu")
        parent_before = parent.clone()
        _, left = layer.step(torch.tensor([[[1.0, 2.0]]]), parent)
        _, right = layer.step(torch.tensor([[[8.0, 9.0]]]), parent)
        torch.testing.assert_close(parent, parent_before, rtol=0, atol=0)
        self.assertFalse(torch.equal(left, right))

    def test_state_batch_selection_is_branch_local(self):
        tensors = tuple(torch.arange(12).reshape(3, 2, 2) + offset for offset in (0, 20))
        state = CanonACState(a=tensors, c=tuple(value + 40 for value in tensors))
        selected = state.select(torch.tensor([2, 0]))
        torch.testing.assert_close(selected.a[0], tensors[0][[2, 0]])
        torch.testing.assert_close(selected.c[1], (tensors[1] + 40)[[2, 0]])


if __name__ == "__main__":
    unittest.main()
