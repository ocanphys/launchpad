"""Declaration against a temporary root: preview, commit, every blocker, and
the notebook copy a run's declaration leaves behind."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import lab
from artifacts.core.artifact import Artifact, Resources
from artifacts.core.SGD.training import (
    LoopConfig,
    LRSchedule,
    OptimizerParameters,
    TrainingParameters,
)
from artifacts.core.storage import MANIFEST
from artifacts.dataset import DataSet
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer
from config import STORAGE
from models.transformer import Transformer

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
TOKENIZER = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(ODYSSEY,))
TOKENS = TokenizedSource(tokenizer=TOKENIZER, source=ODYSSEY)
PRETRAINING = Pretraining(
    run_id="r",
    dataset=DataSet.from_sources(TOKENIZER, (ODYSSEY,), (ODYSSEY,)),
   
    model=Transformer(
        vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32,
        num_heads=2, rope_theta=10000,
    ),
    training_parameters=TrainingParameters(
        total_steps=4, batch_size=4, max_norm=1,
        lr_schedule=LRSchedule(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=1, cosine_cycle_iters=100),
        optimizer="torch.optim.AdamW",
        optimizer_parameters=OptimizerParameters(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
    ),
    loop_config=LoopConfig(val_every=2, gpu_check_every=2, checkpoint_every=2),
)
NOTEBOOK = b'{"cells": []}'


def declare(artifact, root, **options) -> lab.DeclarationReport:
    with redirect_stdout(io.StringIO()):
        return lab.declare(artifact, root=root, **options)


def states(report: lab.DeclarationReport) -> dict[str, str]:
    return {row["path"]: row["state"] for row in report.rows}


def declaring_at(commit: str):
    """Returns a patch making this process's commit `commit`, as a later checkout would."""
    return patch("lab._head", return_value=commit)


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
        with declaring_at("old"):
            declare(ODYSSEY, self.root, commit=True)
        path = self.root / "sources/odyssey" / MANIFEST
        before = path.read_text()

        with declaring_at("new"):
            report = declare(ODYSSEY, self.root, commit=True)
            self.assertEqual(states(report), {"sources/odyssey": "already declared"})
            self.assertTrue(report.rows[0]["drift"])
            self.assertEqual(report.rows[0]["declaring_commit"], ["old", "new"])
            self.assertFalse(report.rows[0]["created"])
            self.assertEqual(path.read_text(), before)

            with self.assertRaises(lab.DeclarationError) as refused:
                declare(ODYSSEY, self.root, commit=True, strict_commit=True)
        self.assertTrue(refused.exception.report.blockers)

    def test_every_manifest_written_records_the_declaring_commit_at_its_root_only(self):
        with declaring_at("abc"):
            declare(TOKENS, self.root, commit=True)
        for row in declare(TOKENS, self.root).rows:
            data = json.loads((self.root / row["path"] / MANIFEST).read_text())
            self.assertEqual(data["declaring_commit"], "abc")
            embedded = json.dumps(data["dependencies"])
            self.assertNotIn("declaring_commit", embedded)
            self.assertNotIn("allocated_resources", embedded)

    def test_a_new_artifact_over_older_dependencies_is_drift_on_their_rows(self):
        with declaring_at("old"):
            declare(TOKENIZER, self.root, commit=True)
        with declaring_at("new"):
            report = declare(TOKENS, self.root)
            drift = {row["path"]: row["declaring_commit"] for row in report.rows if row["drift"]}
            self.assertEqual(drift, {"sources/odyssey": ["old", "new"], f"tokenizers/{TOKENIZER.uid}": ["old", "new"]})
            with self.assertRaises(lab.DeclarationError):
                declare(TOKENS, self.root, commit=True, strict_commit=True)
        self.assertFalse((self.root / TOKENS.artifact_path).exists())

    def test_a_dependency_of_a_loaded_artifact_requests_no_resources(self):
        source = SourceURL(name="odyssey", url=ODYSSEY.url, allocated_resources=Resources(cpu=2.0))
        tokenizer = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(source,))
        declare(tokenizer, self.root, commit=True)
        loaded = Artifact.load(f"tokenizers/{tokenizer.uid}", self.root)
        (embedded,) = loaded.sources
        self.assertIsNone(embedded.allocated_resources)

        report = declare(TokenizedSource(tokenizer=loaded, source=embedded), self.root, commit=True)
        self.assertFalse(report.rows[0]["updated"])
        self.assertEqual(Artifact.load("sources/odyssey", self.root).allocated_resources, Resources(cpu=2.0))

        (self.root / "sources/odyssey" / MANIFEST).unlink()  # new at its path again: platform defaults
        declare(loaded, self.root, commit=True)
        self.assertEqual(Artifact.load("sources/odyssey", self.root).allocated_resources, Resources())

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
        with declaring_at("old"):
            declare(ODYSSEY, self.root, commit=True)
        wanted = SourceURL(name="odyssey", url=ODYSSEY.url, allocated_resources=Resources(cpu=4.0))

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
        self.assertEqual(stored.declaring_commit, "old")

        report = declare(wanted, self.root, commit=True)  # settled: nothing to write
        self.assertFalse(report.rows[0]["updated"])
        self.assertEqual(report.rows[0]["resources"], [Resources(cpu=4.0).model_dump()] * 2)

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
