"""Declaration against a temporary root: preview, commit, every blocker, and
the notebook copy a run's declaration leaves behind."""

import io
import os
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import lab
from artifacts.core.artifact import MANIFEST, Resources
from artifacts.core.SGD.training import (
    LoopConfig,
    LRSchedule,
    OptimizerParameters,
    TrainingParameters,
)
from artifacts.dataset import DataSet
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer
from config import STORAGE
from models.transformer import ModelParameters

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
TOKENIZER = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(ODYSSEY,))
TOKENS = TokenizedSource(tokenizer=TOKENIZER, source=ODYSSEY)
PRETRAINING = Pretraining(
    run_id="r",
    dataset=DataSet.from_sources(TOKENIZER, (ODYSSEY,), (ODYSSEY,)),
    tokenizer=TOKENIZER,
    model="models.transformer",
    model_parameters=ModelParameters(
        vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32,
        num_heads=2, rope_theta=10000, device="cpu",
    ),
    training_parameters=TrainingParameters(
        total_steps=4, batch_size=4, max_norm=1,
        lr_schedule=LRSchedule(1e-3, 1e-4, 1, 100),
        optimizer="torch.optim.AdamW",
        optimizer_parameters=OptimizerParameters(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
    ),
    loop_config=LoopConfig(val_every=2, gpu_check_every=0, checkpoint_every=2),
)
NOTEBOOK = b'{"cells": []}'


def declare(artifact, root, **options) -> lab.DeclarationReport:
    with redirect_stdout(io.StringIO()):
        return lab.declare(artifact, root=root, **options)


def states(report: lab.DeclarationReport) -> dict[str, str]:
    return {row["path"]: row["state"] for row in report.rows}


class DeclareTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_outside_a_container_the_mount_path_is_a_plain_folder(self):
        # STORAGE on a laptop is a path with no volume behind it: no reload,
        # no commit, and a preview simply reports what is (not) there.
        from config import STORAGE

        report = declare(ODYSSEY, STORAGE)
        self.assertEqual(states(report), {"sources/odyssey": "new"})

    def test_preview_against_empty_storage_writes_nothing(self):
        report = declare(TOKENS, self.root)
        self.assertEqual(set(states(report).values()), {"new"})
        self.assertFalse(report.blockers)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_commit_publishes_in_dependency_order_and_reports_it(self):
        report = declare(TOKENS, self.root, commit=True)
        self.assertEqual(
            [row["path"] for row in report.rows if row["created"]],
            ["sources/odyssey", f"tokenizers/{TOKENIZER.uid}", TOKENS.artifact_path.as_posix()],
        )
        self.assertEqual(set(states(report).values()), {"declared"})
        for row in report.rows:
            self.assertTrue((self.root / row["path"] / MANIFEST).is_file())

    def test_redeclaring_keeps_the_existing_manifest_and_reports_drift(self):
        declare(SourceURL(name="odyssey", url=ODYSSEY.url, commit="old"), self.root, commit=True)
        path = self.root / "sources/odyssey" / MANIFEST
        before = path.read_text()

        requested = SourceURL(name="odyssey", url=ODYSSEY.url, commit="new")
        report = declare(requested, self.root, commit=True)
        self.assertEqual(states(report), {"sources/odyssey": "declared"})
        self.assertTrue(report.rows[0]["drift"])
        self.assertFalse(report.rows[0]["created"])
        self.assertEqual(path.read_text(), before)

        with self.assertRaises(lab.DeclarationError) as refused:
            declare(requested, self.root, commit=True, strict_commit=True)
        self.assertTrue(refused.exception.report.blockers)

    def test_a_different_definition_at_a_published_path_is_a_conflict(self):
        declare(ODYSSEY, self.root, commit=True)
        elsewhere = SourceURL(name="odyssey", url="https://example.org/other.txt")

        report = declare(elsewhere, self.root)  # preview: a report, not an error
        self.assertEqual(states(report), {"sources/odyssey": "conflict"})
        self.assertEqual(
            report.rows[0]["differences"],
            {"parameters.url": [ODYSSEY.url, elsewhere.url]},
        )

        with self.assertRaises(lab.DeclarationError):
            declare(elsewhere, self.root, commit=True)

    def test_a_conflicting_leaf_stops_the_whole_commit(self):
        declare(ODYSSEY, self.root, commit=True)
        elsewhere = SourceURL(name="odyssey", url="https://example.org/other.txt")
        tokens = TokenizedSource(
            tokenizer=Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(elsewhere,)),
            source=elsewhere,
        )
        with self.assertRaises(lab.DeclarationError):
            declare(tokens, self.root, commit=True)
        self.assertFalse((self.root / "tokenizers").exists())  # nothing written

    def test_an_owned_file_without_a_manifest_blocks(self):
        body = self.root / "sources/odyssey/body.txt"
        body.parent.mkdir(parents=True)
        body.touch()
        report = declare(ODYSSEY, self.root)
        self.assertEqual(states(report), {"sources/odyssey": "undeclared"})
        with self.assertRaises(lab.DeclarationError):
            declare(ODYSSEY, self.root, commit=True)

    def test_an_unreadable_manifest_is_a_conflict_at_its_own_path(self):
        path = self.root / "sources/odyssey" / MANIFEST
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        report = declare(TOKENS, self.root)
        self.assertEqual(states(report)["sources/odyssey"], "conflict")
        self.assertEqual(states(report)[f"tokenizers/{TOKENIZER.uid}"], "new")

    def test_a_commit_overwrites_differing_resources_and_nothing_else(self):
        declare(SourceURL(name="odyssey", url=ODYSSEY.url, commit="old"), self.root, commit=True)
        wanted = SourceURL(
            name="odyssey", url=ODYSSEY.url, commit="new", allocated_resources=Resources(cpu=4.0)
        )

        report = declare(wanted, self.root)  # preview: reported, not written
        self.assertFalse(report.blockers)
        self.assertFalse(report.rows[0]["updated"])
        self.assertIn('resources differ, requested apply on commit: {} -> {"cpu": 4.0}', report.render())
        self.assertEqual(lab.Artifact.load("sources/odyssey", self.root).allocated_resources, Resources())

        report = declare(wanted, self.root, commit=True)
        self.assertTrue(report.rows[0]["updated"])
        self.assertIn('resources updated: {} -> {"cpu": 4.0}', report.render())
        stored = lab.Artifact.load("sources/odyssey", self.root)
        self.assertEqual(stored.allocated_resources, Resources(cpu=4.0))
        self.assertEqual(stored.commit, "old")

        report = declare(wanted, self.root, commit=True)  # settled: nothing to write
        self.assertFalse(report.rows[0]["updated"])
        self.assertEqual(report.rows[0]["resources"], [asdict(Resources(cpu=4.0))] * 2)

    def test_an_interrupted_declaration_is_completed_by_repeating_it(self):
        declare(TOKENS, self.root, commit=True)
        tokenizer_manifest = self.root / "tokenizers" / TOKENIZER.uid / MANIFEST
        source_manifest = self.root / "sources/odyssey" / MANIFEST
        tokenizer_manifest.unlink()
        before = source_manifest.read_text()

        report = declare(TOKENS, self.root, commit=True)
        self.assertEqual(
            [row["path"] for row in report.rows if row["created"]],
            [f"tokenizers/{TOKENIZER.uid}"],
        )
        self.assertEqual(source_manifest.read_text(), before)

    def test_completion_shows_through_the_report(self):
        declare(ODYSSEY, self.root, commit=True)
        (self.root / "sources/odyssey/body.txt").touch()
        self.assertEqual(states(declare(TOKENS, self.root))["sources/odyssey"], "done")

    def test_a_committed_run_keeps_a_copy_of_the_notebook_that_declared_it(self):
        copy = self.root / "runs" / "r" / f"declare-{PRETRAINING.uid}.ipynb"
        declare(PRETRAINING, self.root, notebook=NOTEBOOK)  # preview
        self.assertFalse((self.root / "runs").exists())

        declare(PRETRAINING, self.root, commit=True, notebook=NOTEBOOK)
        self.assertEqual(copy.read_bytes(), NOTEBOOK)

        # a redeclaration created nothing, so it overwrites nothing
        declare(PRETRAINING, self.root, commit=True, notebook=b"later")
        self.assertEqual(copy.read_bytes(), NOTEBOOK)

    def test_a_shared_artifact_or_a_kernel_without_a_notebook_leaves_no_copy(self):
        declare(TOKENS, self.root, commit=True, notebook=NOTEBOOK)
        self.assertFalse((self.root / "runs").exists())
        with patch.dict(os.environ, {}, clear=True):  # no JPY_SESSION_NAME, no IPython kernel
            declare(PRETRAINING, self.root, commit=True)
        self.assertEqual(list((self.root / "runs" / "r").iterdir()), [self.root / "runs" / "r" / "pretraining"])

    def test_the_current_notebook_is_the_file_jupyter_named_for_the_kernel(self):
        absolute = self.root / "declare.ipynb"
        absolute.write_bytes(NOTEBOOK)
        with patch.dict(os.environ, {"JPY_SESSION_NAME": str(absolute)}):
            self.assertEqual(lab.current_notebook(), NOTEBOOK)
        with patch.dict(os.environ, {"JPY_SESSION_NAME": "runs/missing.ipynb"}), self.assertRaises(FileNotFoundError):
            lab.current_notebook()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(lab.current_notebook())
        # a relative name is relative to the volume, where the lab's jupyter is rooted
        with patch.dict(os.environ, {"JPY_SESSION_NAME": "runs/x.ipynb"}), patch.object(Path, "read_bytes", lambda p: bytes(str(p), "utf8")):
            self.assertEqual(lab.current_notebook(), bytes(f"{STORAGE}/runs/x.ipynb", "utf8"))


if __name__ == "__main__":
    unittest.main()
