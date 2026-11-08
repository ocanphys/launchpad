"""Declaration against a temporary root: preview, commit, every blocker, and
the notebook copy a declaration leaves behind."""

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
from artifacts.core.SGD.lr_schedule import WarmupCosine
from artifacts.core.SGD.optimizers import AdamW
from artifacts.core.SGD.training import (
    LoopConfig,
    TrainingParameters,
)
from artifacts.core.storage import MANIFEST
from artifacts.dataset import DataSet
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.stages.pretraining.evaluations import TokenLoss
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer
from config import LAB_WORKSPACE
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
        lr_schedule=WarmupCosine(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=1, cosine_cycle_iters=100),
        optimizer=AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
    ),
    loop_config=LoopConfig(evaluations=(TokenLoss(eval_every=2, batch_size=4),), gpu_check_every=2, checkpoint_every=2),
)
NOTEBOOK = b'{"cells": [{"cell_type": "markdown", "source": "ephemeral"}, {"cell_type": "code", "source": "x = 1"}]}'


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

    def test_a_commit_never_rewrites_stored_resources(self):
        with declaring_at("old"):
            declare(ODYSSEY, self.root, commit=True)
        before = (self.root / "sources/odyssey" / MANIFEST).read_text()
        wanted = SourceURL(name="odyssey", url=ODYSSEY.url, allocated_resources=Resources(cpu=4.0))

        report = declare(wanted, self.root, commit=True)
        self.assertFalse(report.blockers)
        self.assertFalse(report.rows[0]["updated"])
        self.assertIn('resources differ, stored kept: {} -> {"cpu": 4.0}', report.render())
        self.assertEqual((self.root / "sources/odyssey" / MANIFEST).read_text(), before)

    def test_a_commit_adds_requested_tags_and_never_removes_one(self):
        tagged = SourceURL(name="odyssey", url=ODYSSEY.url, tags={"project": ["lm"]})
        with declaring_at("old"):
            declare(tagged, self.root, commit=True)
        wanted = SourceURL(name="odyssey", url=ODYSSEY.url, tags={"project": ["lm", "sweep"], "experiment": ["e"]})
        merged = {"project": ["lm", "sweep"], "experiment": ["e"]}

        report = declare(wanted, self.root)  # preview: reported, not written
        self.assertIn(f"tags to add on commit: {json.dumps(tagged.tags)} -> {json.dumps(merged)}", report.render())
        self.assertEqual(lab.Artifact.load("sources/odyssey", self.root).tags, tagged.tags)

        report = declare(wanted, self.root, commit=True)
        self.assertTrue(report.rows[0]["updated"])
        self.assertIn(f"tags added: {json.dumps(tagged.tags)} -> {json.dumps(merged)}", report.render())
        stored = lab.Artifact.load("sources/odyssey", self.root)
        self.assertEqual((stored.tags, stored.declaring_commit), (merged, "old"))

        # fewer tags, or none, remove nothing and write nothing
        for fewer in (tagged, ODYSSEY):
            report = declare(fewer, self.root, commit=True)
            self.assertFalse(report.rows[0]["updated"])
            self.assertEqual(lab.Artifact.load("sources/odyssey", self.root).tags, merged)

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

    def test_a_committed_artifact_keeps_a_copy_of_the_notebook_that_declared_it(self):
        copy = self.root / PRETRAINING.artifact_path / "declaring.ipynb"
        declare(PRETRAINING, self.root, notebook=NOTEBOOK)  # preview
        self.assertFalse((self.root / "runs").exists())

        report = declare(PRETRAINING, self.root, commit=True, notebook=NOTEBOOK)
        heading, *rest = json.loads(copy.read_text())["cells"]
        self.assertEqual(heading["source"], f"# {PRETRAINING.artifact_path}")
        self.assertEqual(rest, json.loads(NOTEBOOK)["cells"][1:])
        self.assertTrue(report.render().endswith(f"\n\ndeclared on artifact path:\n{PRETRAINING.artifact_path}"))

        # a first cell of code is kept, under the heading
        code_first = {"cells": json.loads(NOTEBOOK)["cells"][1:]}
        iliad = SourceURL(name="iliad", url="https://example.org/iliad.txt")
        declare(iliad, self.root, commit=True, notebook=json.dumps(code_first).encode())
        heading, *rest = json.loads((self.root / iliad.artifact_path / "declaring.ipynb").read_text())["cells"]
        self.assertEqual((heading["source"], rest), (f"# {iliad.artifact_path}", code_first["cells"]))

        # a redeclaration created nothing, so it overwrites nothing and declares nowhere
        written = copy.read_bytes()
        report = declare(PRETRAINING, self.root, commit=True, notebook=b"later")
        self.assertEqual(copy.read_bytes(), written)
        self.assertNotIn("declared on artifact path", report.render())

    def test_only_the_declared_artifact_keeps_a_copy_and_only_from_a_notebook(self):
        declare(TOKENS, self.root, commit=True, notebook=NOTEBOOK)
        self.assertTrue((self.root / TOKENS.artifact_path / "declaring.ipynb").exists())
        self.assertFalse((self.root / "sources/odyssey/declaring.ipynb").exists())
        with patch.dict(os.environ, {}, clear=True):  # no JPY_SESSION_NAME, no IPython kernel
            declare(PRETRAINING, self.root, commit=True)
        self.assertFalse((self.root / PRETRAINING.artifact_path / "declaring.ipynb").exists())

    def test_the_current_notebook_is_the_file_jupyter_named_for_the_kernel(self):
        absolute = self.root / "declare.ipynb"
        absolute.write_bytes(NOTEBOOK)
        with patch.dict(os.environ, {"JPY_SESSION_NAME": str(absolute)}):
            self.assertEqual(lab.current_notebook(), NOTEBOOK)
        with patch.dict(os.environ, {"JPY_SESSION_NAME": "runs/missing.ipynb"}), self.assertRaises(FileNotFoundError):
            lab.current_notebook()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(lab.current_notebook())
        # a relative name is relative to the workspace, where the lab's jupyter is rooted
        with patch.dict(os.environ, {"JPY_SESSION_NAME": "new.ipynb"}), patch.object(Path, "read_bytes", lambda p: bytes(str(p), "utf8")):
            self.assertEqual(lab.current_notebook(), bytes(f"{LAB_WORKSPACE}/new.ipynb", "utf8"))


if __name__ == "__main__":
    unittest.main()
