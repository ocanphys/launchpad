"""Experiment creation against a temporary root: one graph for every run, each
path declared once, tags only added, and an experiment folder that exists
only once everything it promises is written."""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import lab
from artifacts.core.artifact import Artifact
from training.lr_schedule import WarmupCosine
from training.optimizers import AdamW
from training.leg import LoopConfig, TrainingParameters
from artifacts.core.storage import MANIFEST
from artifacts.dataset import DataSet
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.stages.pretraining.evaluations import TokenLoss
from artifacts.tokenizers.bpe import Tokenizer
from models.transformer import Transformer

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
TOKENIZER = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(ODYSSEY,))
DATASET = DataSet.from_sources(TOKENIZER, (ODYSSEY,), (ODYSSEY,))
UID = "study"
CELL = 'lab.create_experiment("study", runs, commit=True)'
NOTEBOOK = json.dumps({"cells": [
    {"cell_type": "markdown", "source": "temporary"},
    {"cell_type": "code", "source": [CELL]},
]}).encode()
LATER_CELL = 'lab.create_experiment("study", more_runs, commit=True, extend="add")'
LATER = json.dumps({"cells": [
    {"cell_type": "markdown", "source": "temporary"},
    {"cell_type": "code", "source": [LATER_CELL]},
]}).encode()


def run(batch_size: int, **extra) -> Pretraining:
    return Pretraining(
        run_id=f"{UID}-bs{batch_size}",
        dataset=DATASET,
        model=Transformer(vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2, rope_theta=10000),
        training_parameters=TrainingParameters(
            total_steps=4, batch_size=batch_size, max_norm=1,
            lr_schedule=WarmupCosine(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=1, cosine_cycle_iters=100),
            optimizer=AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
        ),
        loop_config=LoopConfig(evaluations=(TokenLoss(eval_every=2, batch_size=4),), gpu_check_every=2, checkpoint_every=2),
        **extra,
    )


RUNS = [run(32), run(64)]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def create(self, artifacts=RUNS, uid=UID, **options) -> lab.DeclarationReport:
        options.setdefault("notebook", NOTEBOOK)
        with redirect_stdout(io.StringIO()):
            return lab.create_experiment(uid, artifacts, root=self.root, **options)

    def manifests(self) -> dict[str, str]:
        return {str(path.relative_to(self.root)): path.read_text() for path in self.root.rglob(MANIFEST)}

    def test_a_preview_writes_nothing(self):
        report = self.create(commit=False)
        self.assertEqual({row["state"] for row in report.rows}, {"new"})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_every_run_resolves_into_one_graph_each_shared_path_once(self):
        report = self.create(commit=True)
        paths = [row["path"] for row in report.rows]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(paths.count(f"tokenizers/{TOKENIZER.uid}"), 1)
        self.assertEqual(paths[-2:], [r.artifact_path.as_posix() for r in RUNS])
        self.assertTrue(all(row["created"] for row in report.rows))

        rendered = report.render()
        self.assertTrue(rendered.endswith("declared on artifact paths:\n" + "\n".join(paths[-2:])))

    def test_runs_are_tagged_with_the_experiment_and_their_dependencies_are_not(self):
        self.create(RUNS[:1] + [run(64, tags={"project": ["lm"]})], commit=True)
        tags = {r.artifact_path: Artifact.load(r.artifact_path, self.root).tags for r in RUNS}
        self.assertEqual(list(tags.values()), [{"experiment": [UID]}, {"project": ["lm"], "experiment": [UID]}])
        self.assertEqual(Artifact.load(TOKENIZER.artifact_path, self.root).tags, {})

    def test_the_experiment_keeps_the_notebook_and_each_new_run_a_copy(self):
        self.create(commit=True)
        kept = json.loads((self.root / "experiments" / UID / "declaration.ipynb").read_text())["cells"]
        self.assertEqual([cell["source"] for cell in kept], [f"# experiments/{UID}", [CELL]])
        for r in RUNS:
            copy = json.loads((self.root / r.artifact_path / "declaring.ipynb").read_text())["cells"]
            self.assertEqual(copy[0]["source"], f"# {r.artifact_path.as_posix()}")
        self.assertFalse((self.root / TOKENIZER.artifact_path / "declaring.ipynb").exists())
        self.assertEqual(sorted(p.name for p in (self.root / "experiments").iterdir()), [UID])

    def test_the_experiments_notebook_lists_every_artifact_tagged_with_it(self):
        self.create(commit=True)
        with redirect_stdout(io.StringIO()):  # a run tagged later belongs too, a run of another experiment not
            lab.declare(run(128, tags={"experiment": [UID]}), root=self.root, commit=True)
            lab.declare(run(256, tags={"experiment": ["other"]}), root=self.root, commit=True)
        tagged = sorted([*RUNS, run(128)], key=lambda r: r.artifact_path)
        self.assertEqual(lab.load_experiment(UID, self.root), tagged)

        heading, code = json.loads((self.root / "experiments" / UID / "notebook.ipynb").read_text())["cells"]
        self.assertEqual(heading["source"], f"# experiments/{UID}")
        namespace, load = {}, lab.load_experiment
        with patch("lab.load_experiment", lambda uid: load(uid, self.root)):
            exec(code["source"], namespace)
        self.assertEqual(namespace["artifacts"], tagged)

    def test_a_run_given_twice_is_one_run(self):
        report = self.create([RUNS[0], run(32)], commit=True)
        self.assertEqual([row["path"] for row in report.rows].count(RUNS[0].artifact_path.as_posix()), 1)
        self.assertEqual(report.roots, [RUNS[0].artifact_path.as_posix()])

    def test_an_existing_run_is_reused_and_only_gains_the_experiment(self):
        with redirect_stdout(io.StringIO()):
            lab.declare(run(32, tags={"experiment": ["earlier"]}), root=self.root, commit=True, notebook=b'{"cells": []}')
        path = self.root / RUNS[0].artifact_path
        before = (path / "declaring.ipynb").read_bytes()

        report = self.create(commit=True)
        row = next(row for row in report.rows if row["path"] == RUNS[0].artifact_path.as_posix())
        self.assertEqual((row["created"], row["updated"]), (False, True))
        stored = Artifact.load(RUNS[0].artifact_path, self.root)
        self.assertEqual((stored, stored.tags), (RUNS[0], {"experiment": ["earlier", UID]}))
        self.assertEqual((path / "declaring.ipynb").read_bytes(), before)

    def test_a_shared_dependency_already_declared_is_untouched(self):
        with redirect_stdout(io.StringIO()):
            lab.declare(SourceURL(name="odyssey", url=ODYSSEY.url, tags={"project": ["lm"]}), root=self.root, commit=True)
        before = (self.root / "sources/odyssey" / MANIFEST).read_text()
        self.create(commit=True)
        self.assertEqual((self.root / "sources/odyssey" / MANIFEST).read_text(), before)

    def test_a_taken_experiment_uid_is_refused_and_left_as_it_was(self):
        self.create(commit=True)
        kept = (self.root / "experiments" / UID / "declaration.ipynb").read_bytes()
        for commit in (False, True):
            with self.assertRaisesRegex(FileExistsError, "extend='add'"):
                self.create([run(128)], commit=commit)
        self.assertEqual((self.root / "experiments" / UID / "declaration.ipynb").read_bytes(), kept)
        self.assertFalse((self.root / run(128).artifact_path).exists())

    def test_an_extension_declares_its_runs_and_keeps_its_notebook_beside_the_first(self):
        self.create(commit=True)
        folder = self.root / "experiments" / UID
        first, listing = (folder / "declaration.ipynb").read_bytes(), (folder / "notebook.ipynb").read_bytes()
        self.create([run(128)], commit=True, extend="add", notebook=LATER)
        self.create([run(256)], commit=True, extend="add", notebook=LATER)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ["declaration.ipynb", "declaration2.ipynb", "declaration3.ipynb", "notebook.ipynb"])
        self.assertEqual((folder / "declaration.ipynb").read_bytes(), first)  # the one the dashboard opens
        self.assertEqual((folder / "notebook.ipynb").read_bytes(), listing)
        for name in ("declaration2.ipynb", "declaration3.ipynb"):
            kept = json.loads((folder / name).read_text())["cells"]
            self.assertEqual([cell["source"] for cell in kept], [f"# experiments/{UID}", [LATER_CELL]])
        self.assertEqual(lab.load_experiment(UID, self.root), sorted([*RUNS, run(128), run(256)], key=lambda r: r.artifact_path))
        self.assertTrue((self.root / run(128).artifact_path / "declaring.ipynb").exists())

    def test_an_overwriting_extension_replaces_the_first_declaration(self):
        self.create(commit=True)
        self.create([run(128)], commit=True, extend="overwrite", notebook=LATER)
        folder = self.root / "experiments" / UID
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["declaration.ipynb", "notebook.ipynb"])
        kept = json.loads((folder / "declaration.ipynb").read_text())["cells"]
        self.assertEqual([cell["source"] for cell in kept], [f"# experiments/{UID}", [LATER_CELL]])
        self.assertIn(run(128), lab.load_experiment(UID, self.root))

    def test_extending_an_experiment_not_there_yet_creates_it(self):
        self.create(commit=True, extend="add")
        folder = self.root / "experiments" / UID
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["declaration.ipynb", "notebook.ipynb"])

    def test_a_preview_of_an_extension_writes_nothing(self):
        self.create(commit=True)
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        report = self.create([run(128)], commit=False, extend="add")
        self.assertEqual({row["path"]: row["state"] for row in report.rows}[run(128).artifact_path.as_posix()], "new")
        self.assertEqual(sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")), before)

    def test_an_experiment_uid_must_be_one_plain_folder_name(self):
        for uid in ("", "a/b", "..", ".staging-x", "a b"):
            with self.assertRaisesRegex(ValueError, "experiment_uid"):
                self.create(uid=uid)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.create([])

    def test_a_conflict_anywhere_writes_nothing(self):
        Path(self.root / "sources/odyssey").mkdir(parents=True)
        (self.root / "sources/odyssey" / MANIFEST).write_text("{not json")
        with self.assertRaises(lab.DeclarationError):
            self.create(commit=True)
        self.assertEqual(self.manifests(), {"sources/odyssey/manifest.json": "{not json"})
        self.assertEqual(list((self.root / "experiments").iterdir()), [])

    def test_two_definitions_at_one_path_are_refused_before_anything(self):
        other = SourceURL(name="odyssey", url="https://example.org/elsewhere.txt")
        with self.assertRaisesRegex(ValueError, "different definitions at sources/odyssey"):
            self.create([RUNS[0], other], commit=True)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_failed_commit_leaves_no_experiment_and_a_rerun_completes_it(self):
        written = []

        def fail_on_third(root, path, manifest):
            if len(written) == 2:
                raise OSError("disk went away")
            written.append(path)
            write(root, path, manifest)

        write = lab.write_manifest
        with patch("lab.write_manifest", fail_on_third), self.assertRaisesRegex(OSError, "disk went away"):
            self.create(commit=True)
        self.assertEqual(list((self.root / "experiments").iterdir()), [])
        self.assertEqual(len(self.manifests()), 2)

        report = self.create(commit=True)
        self.assertTrue((self.root / "experiments" / UID / "declaration.ipynb").exists())
        self.assertTrue(all(row["state"] != "new" for row in report.rows))

    def test_a_commit_keeps_the_saved_notebook_and_refuses_an_unsaved_one(self):
        kernel = SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=["", CELL]))
        unsaved = json.dumps({"cells": [{"cell_type": "code", "source": "runs = []"}]}).encode()
        with patch("IPython.core.getipython.get_ipython", return_value=kernel):
            with patch("lab.current_notebook", return_value=unsaved), self.assertRaisesRegex(ValueError, "save it"):
                self.create(commit=True, notebook=None)
            self.assertFalse((self.root / "experiments").exists())
            with patch("lab.current_notebook", return_value=NOTEBOOK):
                self.create(commit=True, notebook=None)
        self.assertTrue((self.root / "experiments" / UID / "declaration.ipynb").exists())

    def test_a_commit_outside_a_notebook_is_refused(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "run this in one"):
            self.create(commit=True, notebook=None)
        # a script a notebook started: the notebook's name, and no kernel
        with patch("lab.current_notebook", return_value=NOTEBOOK), patch("IPython.core.getipython.get_ipython", return_value=None):
            with self.assertRaisesRegex(ValueError, "run this in one"):
                self.create(commit=True, notebook=None)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_commit_on_the_volume_tells_the_launcher(self):
        """One message for the experiment, and one per declare."""
        sent = []
        with (
            patch.object(lab, "STORAGE", self.root),
            patch.object(lab, "refresh"),
            patch.object(lab, "save"),
            patch.object(lab, "refreshes", SimpleNamespace(put=sent.append)),
        ):
            self.create(commit=True)
            with redirect_stdout(io.StringIO()):
                lab.declare(ODYSSEY, root=self.root, commit=True, notebook=NOTEBOOK)
<<<<<<< HEAD
        self.assertEqual(sent, [
            {"experiment_uid": UID, "artifact_paths": [r.artifact_path.as_posix() for r in RUNS], "event": "experiment created"},
            {"artifact_path": ODYSSEY.artifact_path.as_posix(), "call_id": "launcher", "event": "declared"},
=======
            self.create([run(128)], commit=True, extend="add", notebook=LATER)
        self.assertEqual(sent, [
            {"experiment_uid": UID, "artifact_paths": [r.artifact_path.as_posix() for r in RUNS], "event": "experiment created"},
            {"artifact_path": ODYSSEY.artifact_path.as_posix(), "call_id": "launcher", "event": "declared"},
            {"experiment_uid": UID, "artifact_paths": [run(128).artifact_path.as_posix()], "event": "experiment extended"},
>>>>>>> 53dccf3 (profiling and experiment declare-extend)
        ])


if __name__ == "__main__":
    unittest.main()
