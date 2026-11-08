"""The samplers a pretraining leg can declare, and its evaluations can read with.

A sampler is the recipe for which token positions each step reads from a split:
one Sampler subclass per recipe, its fields its parameters. The leg passes the
shape it wants filled (batch size, sequence length, seed, end step), so a
sampler holds none of it. Numpy loads inside the methods, so importing this
module from Pretraining costs nothing.

In the examples, `split` is 100 tokens that are their own positions:
`Split(np.arange(100, dtype=np.uint16), TagRanges.untagged(100))`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import Field

from artifacts.core.artifact import ParamConfig

if TYPE_CHECKING:
    import numpy as np

    from artifacts.core.split import Split


class Sampler(ParamConfig):
    """The base of every sampler a leg declares: `starts` is the recipe, and
    `batches`, shared by every sampler, gathers the windows it names.

    A field annotated `Sampler` holds any subclass, and its manifest names that
    subclass under "class" (artifacts.core.manifest). The base declares no
    fields, so no subclass's manifest carries one it did not declare.
    """

    def starts(
        self, split: Split, batch_size: int, sequence_length: int, seed: int, end_step: int
    ) -> Callable[[int], np.ndarray]:
        """Returns the function from a step to the batch_size positions of
        split.tokens its windows start at.

        It depends on this sampler's fields, the split and the arguments alone,
        so any step can be asked for in any order, and a resumed leg reads
        exactly what the interrupted one would have. A split that cannot serve
        every step up to end_step raises here, before the first.

        Args:
            split: The bound split the windows are drawn from.
            batch_size: How many windows each step reads.
            sequence_length: Tokens per window, plus the next one as its target.
            seed: The leg's seed.
            end_step: The last step the leg takes.

        Returns:
            The function from a step number, from 1, to an int64 array of starts.
        """
        raise NotImplementedError

    def batches(
        self, split: Split, batch_size: int, sequence_length: int, seed: int, end_step: int
    ) -> Callable[[int], tuple[np.ndarray, np.ndarray]]:
        """Returns the function from a step to its (inputs, targets), int64
        arrays of shape (batch_size, sequence_length) read at `starts(step)`.

            RandomWindows().batches(split, batch_size=4, sequence_length=8, seed=0, end_step=100)(1)[0][0]
              -> array([48, 49, 50, 51, 52, 53, 54, 55])

        Args:
            split: The bound split the windows are drawn from.
            batch_size: How many windows each step reads.
            sequence_length: Tokens per window; targets are the same window one on.
            seed: The leg's seed.
            end_step: The last step the leg takes.

        Returns:
            The function from a step number, from 1, to its inputs and targets.
        """
        import numpy as np

        starts = self.starts(split, batch_size, sequence_length, seed, end_step)
        tokens = split.tokens

        def batch(step: int) -> tuple[np.ndarray, np.ndarray]:
            at = starts(step)
            # int64 because Embedding needs long
            inputs = np.stack([tokens[i : i + sequence_length] for i in at]).astype(np.int64)
            targets = np.stack([tokens[i + 1 : i + sequence_length + 1] for i in at]).astype(np.int64)
            return inputs, targets

        return batch


class RandomWindows(Sampler):
    """Windows at uniformly random positions, drawn afresh each step by
    `default_rng((seed, step))`: windows overlap and repeat, and no step's
    choice depends on another's.

        RandomWindows().starts(split, batch_size=4, sequence_length=8, seed=0, end_step=100)(2)
          -> array([90,  7, 26, 37])
    """

    def starts(
        self, split: Split, batch_size: int, sequence_length: int, seed: int, end_step: int
    ) -> Callable[[int], np.ndarray]:
        import numpy as np

        high = len(split.tokens) - sequence_length
        if high < 1:
            raise ValueError(
                f"the training split holds {len(split.tokens)} tokens, fewer than one "
                f"sequence of {sequence_length} and its next token"
            )
        return lambda step: np.random.default_rng((seed, step)).integers(0, high, size=batch_size)


class Epochs(Sampler):
    """Every sequence once per epoch, in an order shuffled afresh each epoch.

    The split is cut into (n - 1) // sequence_length sequences placed
    sequence_length apart, so each has its next token as a target. An epoch is
    one permutation of them in batches of batch_size; the partial batch at its
    end is dropped. Epoch e's permutation, and an offset that moves every
    sequence along by up to the tokens no sequence covers, come from
    `default_rng((seed, e))`, so a different tail goes unread each epoch.
    Epochs count from the run's step 1, so a leg continuing another continues
    its epoch. `max_epochs`, when set, refuses a leg whose end_step lies past
    that many epochs.

        Epochs(max_epochs=2).starts(split, batch_size=4, sequence_length=8, seed=0, end_step=6)(1)
          -> array([75, 19, 59, 35])
    """

    max_epochs: int | None = Field(default=None, gt=0)

    def starts(
        self, split: Split, batch_size: int, sequence_length: int, seed: int, end_step: int
    ) -> Callable[[int], np.ndarray]:
        import numpy as np

        length = len(split.tokens)
        sequences = (length - 1) // sequence_length
        steps_per_epoch = sequences // batch_size
        if steps_per_epoch < 1:
            raise ValueError(
                f"the training split holds {length} tokens, {sequences} sequences of "
                f"{sequence_length}: fewer than one batch of {batch_size}"
            )
        if self.max_epochs is not None and end_step > self.max_epochs * steps_per_epoch:
            raise ValueError(
                f"end_step {end_step} is past {self.max_epochs} epochs of {steps_per_epoch} "
                f"steps; the last step they serve is {self.max_epochs * steps_per_epoch}"
            )
        spare = (length - 1) % sequence_length

        @lru_cache(maxsize=1)
        def order(epoch: int) -> np.ndarray:
            rng = np.random.default_rng((seed, epoch))
            offset = rng.integers(0, spare + 1)
            return offset + rng.permutation(sequences) * sequence_length

        def starts(step: int) -> np.ndarray:
            epoch, batch = divmod(step - 1, steps_per_epoch)
            return order(epoch)[batch * batch_size : (batch + 1) * batch_size]

        return starts


class Sweep(Sampler):
    """Every sequence once, in order: the split is cut into (n - 1) //
    sequence_length sequences placed sequence_length apart from position 0, and
    step s reads the s-th batch_size of them. Nothing is random, so the seed is
    unused; a leg whose end_step lies past the last full batch is refused.

        Sweep().starts(split, batch_size=4, sequence_length=8, seed=0, end_step=3)(2)
          -> array([32, 40, 48, 56])
    """

    def starts(
        self, split: Split, batch_size: int, sequence_length: int, seed: int, end_step: int
    ) -> Callable[[int], np.ndarray]:
        import numpy as np

        sequences = (len(split.tokens) - 1) // sequence_length
        if end_step > sequences // batch_size:
            raise ValueError(
                f"the split holds {len(split.tokens)} tokens, {sequences} sequences of "
                f"{sequence_length}: {sequences // batch_size} batches of {batch_size}, "
                f"fewer than end_step {end_step}"
            )
        return lambda step: ((step - 1) * batch_size + np.arange(batch_size)) * sequence_length

