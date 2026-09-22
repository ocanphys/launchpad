"""Dataset validation and uint16 source boundaries against a temporary root."""

import json
import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from artifacts.core.artifact import MANIFEST, Artifact, _digest
from artifacts.dataset import DataSet
from artifacts.dataset.jobs import DataSetJob
from artifacts.sources import SourceURL
from artifacts.tokenized import TokenizedSource
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


class DataSetTests(unittest.TestCase):
    def test_multiple_sources_require_eot_at_construction_in_either_split(self):
        for train, valid, split in (
            ([FIRST, SECOND], [THIRD], "train_set"),
            ([FIRST], [SECOND, THIRD], "valid_set"),
        ):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, split) as error:
                    DataSet.from_sources(NO_EOT, train, valid)
                self.assertIn("<|endoftext|>", str(error.exception))
                self.assertIn("special_tokens", str(error.exception))

    def test_direct_construction_checks_every_tokenizer_in_a_merged_split(self):
        sources = (TokenizedSource(TOKENIZER, FIRST), TokenizedSource(NO_EOT, SECOND))
        for train, valid in ((sources, ()), ((), sources)):
            with (
                self.subTest(train=train, valid=valid),
                self.assertRaisesRegex(ValueError, "special_tokens"),
            ):
                DataSet(train_set=train, valid_set=valid)

    def test_declared_eot_is_enough_before_tokenizer_is_built(self):
        dataset = DataSet.from_sources(TOKENIZER, (FIRST, SECOND), (THIRD,))
        self.assertFalse(TOKENIZER.bound)
        self.assertEqual(Artifact.from_manifest(dataset.to_manifest()), dataset)

    def test_multiple_sources_have_a_distinct_cache_identity(self):
        for train, valid in (([FIRST, SECOND], []), ([], [FIRST, SECOND])):
            with self.subTest(train=train, valid=valid):
                dataset = DataSet.from_sources(TOKENIZER, train, valid)
                plain_digest = _digest(
                    [source.uid for source in dataset.train_set],
                    [source.uid for source in dataset.valid_set],
                )
                self.assertNotEqual(dataset.uid, f"dataset-{plain_digest}")

    def test_empty_and_single_source_splits_need_no_eot_or_tokenizer_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = TokenizedSource(NO_EOT, FIRST)
            path = source.paths(root)["tokens"]
            path.parent.mkdir(parents=True)
            content = array("H", [7, 300, 65535]).tobytes()
            path.write_bytes(content)
            for train, valid in (
                ([], []),
                ([FIRST], []),
                ([], [FIRST]),
                ([FIRST], [FIRST]),
            ):
                with self.subTest(train=train, valid=valid):
                    dataset = DataSet.from_sources(NO_EOT, train, valid)
                    (root / dataset.artifact_path).mkdir(parents=True)  # as declaration would
                    DataSetJob(dataset).run(root, Mock())
                    self.assertEqual(
                        dataset.paths(root)["training set"].read_bytes(),
                        content if train else b"",
                    )
                    self.assertEqual(
                        dataset.paths(root)["validation set"].read_bytes(),
                        content if valid else b"",
                    )
                    self.assertEqual(
                        dataset.uid,
                        f"dataset-{_digest([s.uid for s in dataset.train_set], [s.uid for s in dataset.valid_set])}",
                    )

    def test_merge_inserts_actual_eot_id_at_each_file_boundary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / TOKENIZER.artifact_path
            folder.mkdir(parents=True)
            (folder / MANIFEST).write_text(json.dumps(TOKENIZER.to_manifest()))
            # Training can stop below vocab_size; use the ID from the saved vocab.
            vocab = {str(i): bytes([i]).decode("latin-1") for i in range(256)}
            vocab.update({"256": "<pad>", "257": "<|endoftext|>"})
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
            dataset = DataSet.from_sources(TOKENIZER, (FIRST, SECOND, THIRD), (THIRD, FIRST))
            (root / dataset.artifact_path).mkdir(parents=True)  # as declaration would
            for contents, expected_train, expected_valid in (
                (([1, 2], [3], [4, 5]), [1, 2, 257, 3, 257, 4, 5], [4, 5, 257, 1, 2]),
                (([1, 2], [], [4, 5]), [1, 2, 257, 257, 4, 5], [4, 5, 257, 1, 2]),
                (([], [3], []), [257, 3, 257], [257]),
            ):
                with self.subTest(contents=contents):
                    for source, tokens in zip(dataset.train_set, contents):
                        path = source.paths(root)["tokens"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(array("H", tokens).tobytes())
                    DataSetJob(dataset).run(root, Mock())
                    for split, expected in (
                        ("training set", expected_train),
                        ("validation set", expected_valid),
                    ):
                        self.assertEqual(
                            dataset.paths(root)[split].read_bytes(),
                            array("H", expected).tobytes(),
                        )

    def test_bound_dataset_maps_the_bins_it_wrote(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = DataSet.from_sources(NO_EOT, (FIRST,), ())
            (root / dataset.artifact_path).mkdir(parents=True)
            (root / dataset.artifact_path / MANIFEST).write_text(
                json.dumps(dataset.to_manifest())
            )
            path = dataset.train_set[0].paths(root)["tokens"]
            path.parent.mkdir(parents=True)
            path.write_bytes(array("H", [7, 300, 65535]).tobytes())
            DataSetJob(dataset).run(root, Mock())
            self.assertFalse(dataset.bound)
            with self.assertRaisesRegex(RuntimeError, "bind"):
                dataset.train_tokens
            bound = dataset.bind(root)
            self.assertIsInstance(bound.train_tokens, np.memmap)
            self.assertEqual(bound.train_tokens[1:].tolist(), [300, 65535])
            self.assertEqual(len(bound.valid_tokens), 0)


if __name__ == "__main__":
    unittest.main()
