"""TokenLoss, through a stand-in job that records the batches it is handed, over
a validation split whose tokens are their own positions."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from pydantic import ValidationError

from artifacts.core.split import Split, TagRanges
from artifacts.stages.pretraining.evaluations import TokenLoss


def measured(evaluation: TokenLoss, tokens: int, sequence_length: int = 8) -> tuple[dict, list[np.ndarray]]:
    """Returns what `evaluation` measures on a validation split of `tokens`
    positions, with each batch's loss its inputs' mean, and the inputs of every
    batch it read, in order."""
    batches = []

    def batch_loss(model, inputs, targets):
        np.testing.assert_array_equal(targets, inputs + 1)
        batches.append(inputs.numpy())
        return inputs.double().mean()

    job = SimpleNamespace(
        artifact=SimpleNamespace(model=SimpleNamespace(sequence_length=sequence_length)),
        to_device=lambda arrays: tuple(torch.from_numpy(array) for array in arrays),
        batch_loss=batch_loss,
    )
    split = Split(np.arange(tokens, dtype=np.uint16), TagRanges.untagged(tokens))
    worker = SimpleNamespace(progress={})
    metrics = evaluation.measure(job, model=None, dataset=SimpleNamespace(valid=split), worker=worker)
    assert "val_batch" not in worker.progress
    return metrics, batches


class TokenLossTests(unittest.TestCase):
    def test_it_reads_the_split_in_order_in_consecutive_sequences(self):
        # 66 tokens: 8 sequences of 8 in 2 full batches; 25: 3 sequences, one short batch
        for tokens, shapes in ((66, [(4, 8), (4, 8)]), (25, [(3, 8)])):
            with self.subTest(tokens=tokens):
                metrics, batches = measured(TokenLoss(eval_every=1, batch_size=4), tokens)
                self.assertEqual([batch.shape for batch in batches], shapes)
                self.assertEqual(batches[0].dtype, np.int64)
                read = np.concatenate([batch.ravel() for batch in batches])
                np.testing.assert_array_equal(read, np.arange(len(read)))
                means = torch.tensor([batch.mean() for batch in batches], dtype=torch.float64)
                self.assertEqual(set(metrics), {"val_loss", "val_loss_var"})
                self.assertAlmostEqual(metrics["val_loss"].item(), means.mean().item())
                self.assertAlmostEqual(metrics["val_loss_var"].item(), means.var(correction=0).item())

    def test_a_split_shorter_than_one_sequence_is_refused(self):
        for tokens in (8, 0):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "fewer than one sequence"):
                measured(TokenLoss(eval_every=1, batch_size=4), tokens)

    def test_batch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValidationError, "batch_size"):
            TokenLoss(eval_every=1, batch_size=0)


if __name__ == "__main__":
    unittest.main()
