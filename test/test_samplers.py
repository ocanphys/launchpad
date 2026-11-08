"""The samplers, over splits whose tokens are their own
positions, so a window's first token is where it starts."""

import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pydantic import ValidationError

from artifacts.core.split import Split, TagRanges
from artifacts.mappeddataset.tokenstream import TokenStream
from artifacts.stages.pretraining.samplers import Epochs, RandomWindows, Sweep

# The starts DataSet.train_batches and MappedDataSet.train_batches drew before
# the sampler existed, keyed by (batch_size, sequence_length, seed, tokens) and step.
TRAIN_BATCHES_STARTS = {
    (4, 8, 0, 100): {1: [48, 81, 91, 51], 2: [90, 7, 26, 37], 3: [46, 82, 87, 79], 50: [22, 71, 53, 77], 1000: [34, 36, 82, 73]},
    (3, 5, 7, 1000): {1: [865, 766, 871], 2: [279, 276, 152], 3: [724, 970, 535], 50: [463, 212, 323], 1000: [488, 138, 196]},
    (8, 16, 123, 5000): {
        1: [3324, 1586, 4507, 859, 2150, 3129, 3208, 4391],
        2: [2780, 1822, 2499, 4737, 1496, 3463, 4905, 3108],
        3: [711, 2094, 4582, 946, 1831, 2062, 3517, 3602],
        50: [3383, 1075, 3666, 3134, 1569, 1424, 3039, 4739],
        1000: [849, 1905, 3862, 3636, 2860, 2687, 4189, 1043],
    },
}
# MappedDataSet.train_batches(3, 6, seed=5) over 10..21 and 100..111 with 257 between.
TRAIN_BATCHES_ACROSS_A_BOUNDARY = {
    5: [[19, 20, 21, 257, 100, 101], [100, 101, 102, 103, 104, 105], [103, 104, 105, 106, 107, 108]],
    7: [[15, 16, 17, 18, 19, 20], [19, 20, 21, 257, 100, 101], [103, 104, 105, 106, 107, 108]],
}
# What Epochs draws, pinned when it shipped: a change to these is a change to
# every Epochs leg's batches under the same identity.
EPOCHS_STARTS = {
    (4, 8, 0, 100): {1: [75, 19, 59, 35], 2: [43, 91, 3, 27], 3: [51, 83, 67, 11], 4: [2, 58, 10, 42], 5: [74, 26, 90, 34], 6: [50, 82, 66, 18]},
    (8, 16, 123, 5000): {
        1: [2032, 3776, 1872, 3520, 4112, 864, 2336, 1664],
        38: [96, 528, 704, 4288, 1680, 3728, 1936, 3888],
        39: [1168, 3424, 80, 4832, 2400, 3504, 4656, 2448],
        77: [2821, 4565, 37, 3893, 2517, 1173, 405, 3957],
    },
}


def positions(length: int) -> Split:
    """A split of `length` tokens, each its own position."""
    return Split(np.arange(length, dtype=np.uint16), TagRanges.untagged(length))


class RandomWindowsTests(unittest.TestCase):
    def test_it_draws_exactly_what_train_batches_drew(self):
        for (batch_size, length, seed, tokens), steps in TRAIN_BATCHES_STARTS.items():
            get = RandomWindows().batches(positions(tokens), batch_size, length, seed, end_step=1000)
            for step, starts in steps.items():
                with self.subTest(case=(batch_size, length, seed, tokens), step=step):
                    inputs, targets = get(step)
                    self.assertEqual((inputs.shape, inputs.dtype), ((batch_size, length), np.int64))
                    np.testing.assert_array_equal(inputs, np.array(starts)[:, None] + np.arange(length))
                    np.testing.assert_array_equal(targets, inputs + 1)

    def test_it_draws_exactly_what_train_batches_drew_across_a_source_boundary(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / "first.bin", Path(directory) / "second.bin"]
            for path, tokens in zip(paths, (range(10, 22), range(100, 112))):
                path.write_bytes(array("H", tokens).tobytes())
            stream = TokenStream(paths, separator=257)
            get = RandomWindows().batches(Split(stream, TagRanges.untagged(len(stream))), 3, 6, seed=5, end_step=8)
            for step, rows in TRAIN_BATCHES_ACROSS_A_BOUNDARY.items():
                inputs, targets = get(step)
                np.testing.assert_array_equal(inputs, rows)
                np.testing.assert_array_equal(targets[:, :-1], inputs[:, 1:])

    def test_a_split_shorter_than_one_sequence_is_refused(self):
        with self.assertRaisesRegex(ValueError, "fewer than one sequence of 8"):
            RandomWindows().starts(positions(8), 4, 8, seed=0, end_step=1)


class EpochsTests(unittest.TestCase):
    def test_it_draws_what_it_drew_when_it_shipped(self):
        for (batch_size, length, seed, tokens), steps in EPOCHS_STARTS.items():
            get = Epochs().starts(positions(tokens), batch_size, length, seed, end_step=100)
            for step, starts in steps.items():
                with self.subTest(case=(batch_size, length, seed, tokens), step=step):
                    self.assertEqual(get(step).tolist(), starts)

    def test_each_epoch_reads_every_sequence_once_from_an_offset_of_its_own(self):
        # 1003 tokens: 100 sequences of 10, 2 tokens spare; 14 batches of 7, the last 2 sequences dropped
        get = Epochs().starts(positions(1003), 7, 10, seed=3, end_step=56)
        epochs = [np.concatenate([get(step) for step in range(1 + 14 * e, 15 + 14 * e)]) for e in range(4)]
        offsets = set()
        for starts in epochs:
            offset = starts.min() % 10
            offsets.add(offset)
            self.assertLessEqual(offset, 2)
            self.assertEqual(len(set((starts - offset) // 10)), 98)
            self.assertTrue(np.all((starts - offset) % 10 == 0))
            self.assertLessEqual(starts.max() + 10, 1002)  # every window has its next token
        self.assertGreater(len(offsets), 1)
        self.assertEqual(len({tuple(starts) for starts in epochs}), 4)

    def test_any_step_reads_the_same_in_any_order(self):
        """What a resumed leg relies on: a closure asked for a step mid-epoch,
        or after a later epoch, returns what a straight run did."""
        straight = Epochs().starts(positions(1003), 7, 10, seed=3, end_step=56)
        expected = {step: straight(step).tolist() for step in range(1, 57)}
        resumed = Epochs().starts(positions(1003), 7, 10, seed=3, end_step=56)
        for step in [20, 56, 1, 33, *range(56, 0, -1)]:
            self.assertEqual(resumed(step).tolist(), expected[step])

    def test_batches_are_the_windows_at_its_starts(self):
        split = positions(1003)
        starts = Epochs().starts(split, 7, 10, seed=3, end_step=20)(17)
        inputs, targets = Epochs().batches(split, 7, 10, seed=3, end_step=20)(17)
        np.testing.assert_array_equal(inputs, starts[:, None] + np.arange(10))
        np.testing.assert_array_equal(targets, inputs + 1)

    def test_a_split_shorter_than_one_batch_is_refused(self):
        # 40 tokens hold 4 sequences of 9, 9 tokens hold none
        for tokens in (40, 9):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "fewer than one batch of 5"):
                Epochs().starts(positions(tokens), 5, 9, seed=0, end_step=1)

    def test_max_epochs_refuses_a_leg_ending_past_them(self):
        # 1003 tokens in batches of 7 sequences of 10: 14 steps an epoch
        Epochs(max_epochs=2).starts(positions(1003), 7, 10, seed=0, end_step=28)
        with self.assertRaisesRegex(ValueError, "end_step 29 is past 2 epochs of 14 steps; the last step they serve is 28"):
            Epochs(max_epochs=2).starts(positions(1003), 7, 10, seed=0, end_step=29)
        Epochs().starts(positions(1003), 7, 10, seed=0, end_step=10**9)
        with self.assertRaisesRegex(ValidationError, "max_epochs"):
            Epochs(max_epochs=0)


class SweepTests(unittest.TestCase):
    def test_it_reads_every_full_batch_of_sequences_once_in_order(self):
        # 1003 tokens hold 100 sequences of 10: 14 batches of 7, 2 sequences in none
        starts = Sweep().starts(positions(1003), 7, 10, seed=0, end_step=14)
        read = np.concatenate([starts(step) for step in range(1, 15)])
        np.testing.assert_array_equal(read, np.arange(98) * 10)

    def test_a_leg_ending_past_the_last_full_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "14 batches of 7, fewer than end_step 15"):
            Sweep().starts(positions(1003), 7, 10, seed=0, end_step=15)


if __name__ == "__main__":
    unittest.main()
