"""MappedDataSet's bound view: one memmap-like stream per split, stitched
across source boundaries, against a temporary root."""

import json
import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from artifacts.core.artifact import Artifact
from artifacts.core.storage import write_manifest
from artifacts.mappeddataset import MappedDataSet
from artifacts.sources import SourceURL
from artifacts.tokenizers.bpe import Tokenizer

FIRST = SourceURL(name="first", url="https://example.org/first.txt")
SECOND = SourceURL(name="second", url="https://example.org/second.txt")
THIRD = SourceURL(name="third", url="https://example.org/third.txt")
TOKENIZER = Tokenizer(
    vocab_size=1000,
    special_tokens=("<pad>", "<|endoftext|>"),
    sources=(FIRST, SECOND, THIRD),
)
NO_EOT = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(FIRST,))
EOT = 257


def declare(artifact: Artifact, root: Path) -> None:
    write_manifest(root, artifact.artifact_path, artifact.to_manifest())


def build_tokenizer(root: Path) -> None:
    """A tokenizer.json with no merges, so <|endoftext|> encodes to EOT."""
    declare(TOKENIZER, root)
    vocab = {str(i): bytes([i]).decode("latin-1") for i in range(256)}
    vocab.update({"256": "<pad>", str(EOT): "<|endoftext|>"})
    TOKENIZER.paths(root)["tokenizer"].write_text(
        json.dumps(
            {
                "vocab_size": len(vocab),
                "special_tokens": TOKENIZER.special_tokens,
                "vocab": vocab,
                "merges": [],
            }
        )
    )


def bound(root: Path, train: tuple, valid: tuple, contents: dict) -> MappedDataSet:
    """The dataset over `train`/`valid`, declared and bound, with each
    source's tokens.bin holding `contents[source]`."""
    dataset = MappedDataSet.from_sources(TOKENIZER, train, valid)
    declare(dataset, root)
    for source in (*dataset.train_set, *dataset.valid_set):
        path = source.paths(root)["tokens"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(array("H", contents[source.source]).tobytes())
    return dataset.bind(root)


class MappedDataSetTests(unittest.TestCase):
    def test_multiple_sources_require_eot_at_construction_in_either_split(self):
        for train, valid, split in (
            ((FIRST, SECOND), (THIRD,), "train_set"),
            ((FIRST,), (SECOND, THIRD), "valid_set"),
        ):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, f"MappedDataSet.{split}"):
                    MappedDataSet.from_sources(NO_EOT, train, valid)

    def test_a_dataset_with_no_sources_at_all_is_refused(self):
        """It owns no files and borrows completion from its sources, so with
        neither split filled `status(root).complete` would be vacuously true
        and the thing would read as done on an empty volume. One empty split
        is still fine: the other one's tokens are what it is done by."""
        with self.assertRaisesRegex(ValueError, "no sources"):
            MappedDataSet.from_sources(TOKENIZER, (), ())
        self.assertEqual(MappedDataSet.from_sources(TOKENIZER, (FIRST,), ()).valid_set, ())

    def test_order_and_separator_are_identity(self):
        one_way = MappedDataSet.from_sources(TOKENIZER, (FIRST, SECOND), ())
        other_way = MappedDataSet.from_sources(TOKENIZER, (SECOND, FIRST), ())
        self.assertNotEqual(one_way.uid, other_way.uid)
        plain = Artifact.digest([s.uid for s in one_way.train_set], [])
        self.assertNotEqual(one_way.uid, f"mapped-{plain}")
        single = MappedDataSet.from_sources(TOKENIZER, (FIRST,), ())
        self.assertEqual(single.uid, f"mapped-{Artifact.digest([s.uid for s in single.train_set], [])}")

    def test_stream_reads_like_the_concatenation_with_eot_between_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tokenizer(root)
            dataset = bound(
                root, (FIRST, SECOND, THIRD), (THIRD,), {FIRST: [1, 2], SECOND: [3], THIRD: [4, 5]}
            )
            train = dataset.train_tokens
            self.assertEqual(len(train), 7)
            self.assertEqual(train.shape, (7,))
            self.assertEqual(train[:].tolist(), [1, 2, EOT, 3, EOT, 4, 5])
            self.assertIsInstance(train[0:2], np.memmap)
            self.assertEqual(train[1:4].tolist(), [2, EOT, 3])
            self.assertEqual(train[2:3].tolist(), [EOT])
            self.assertEqual(train[-1], 5)
            self.assertEqual(train[2], EOT)
            self.assertEqual(train[5:99].tolist(), [4, 5])
            self.assertEqual(train[3:3].tolist(), [])
            with self.assertRaises(IndexError):
                train[7]
            with self.assertRaises(ValueError):
                train[::2]
            valid = dataset.valid_tokens
            self.assertEqual(valid[:].tolist(), [4, 5])
            self.assertIsInstance(valid[:], np.memmap)

    def test_an_empty_source_still_gets_its_separator(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            build_tokenizer(root)
            dataset = bound(
                root, (FIRST, SECOND, THIRD), (), {FIRST: [1, 2], SECOND: [], THIRD: [4, 5]}
            )
            self.assertEqual(dataset.train_tokens[:].tolist(), [1, 2, EOT, EOT, 4, 5])
            self.assertEqual(dataset.train_tokens[2:4].tolist(), [EOT, EOT])
            self.assertEqual(len(dataset.valid_tokens), 0)

    def test_bind_needs_every_tokens_bin(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = MappedDataSet.from_sources(TOKENIZER, (FIRST,), (SECOND,))
            declare(dataset, root)
            with self.assertRaisesRegex(FileNotFoundError, "not built"):
                dataset.bind(root)

    def test_train_batches_draw_windows_of_train_tokens_by_seed_and_step(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "bind"):
                MappedDataSet.from_sources(NO_EOT, (FIRST,), ()).train_batches(4, 8, seed=0)
            dataset = bound(root, (FIRST,), (), {FIRST: list(range(100))})
            get_train_batch = dataset.train_batches(4, 8, seed=0)
            inputs, targets = get_train_batch(3)
            self.assertEqual((inputs.shape, inputs.dtype), ((4, 8), np.int64))
            # tokens are their own positions, so every window counts up by one
            np.testing.assert_array_equal(inputs, inputs[:, :1] + np.arange(8))
            np.testing.assert_array_equal(targets, inputs + 1)
            np.testing.assert_array_equal(get_train_batch(3)[0], inputs)

    def test_valid_batches_cover_valid_tokens_in_order(self):
        # 66 tokens: 8 sequences of 8 in 2 full batches; 25: 3 sequences, one short batch
        for tokens, total, shape in ((66, 2, (4, 8)), (25, 1, (3, 8))):
            with self.subTest(tokens=tokens), TemporaryDirectory() as directory:
                dataset = bound(Path(directory), (FIRST,), (SECOND,), {FIRST: range(100), SECOND: range(tokens)})
                count, get_valid_batch = dataset.valid_batches(4, 8)
                inputs, targets = get_valid_batch(count - 1)
                self.assertEqual((count, inputs.shape, inputs.dtype), (total, shape, np.int64))
                np.testing.assert_array_equal(inputs.ravel(), np.arange((count - 1) * inputs.size, count * inputs.size))
                np.testing.assert_array_equal(targets, inputs + 1)
        for tokens in (8, 0):
            with self.subTest(tokens=tokens), TemporaryDirectory() as directory:
                dataset = bound(Path(directory), (FIRST,), (SECOND,), {FIRST: range(100), SECOND: range(tokens)})
                with self.assertRaisesRegex(ValueError, "fewer than one sequence"):
                    dataset.valid_batches(4, 8)


if __name__ == "__main__":
    unittest.main()
