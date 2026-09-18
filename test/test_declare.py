"""Declaration against a temporary root: preview, commit, and every blocker."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import lab
from artifacts.core.artifact import MANIFEST, Resources
from artifacts.sources import SourceURL
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
TOKENIZER = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(ODYSSEY,))
TOKENS = TokenizedSource(tokenizer=TOKENIZER, source=ODYSSEY)


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

        report = declare(ODYSSEY, Path(STORAGE))
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

    def test_resource_differences_are_reported_without_blocking(self):
        declare(ODYSSEY, self.root, commit=True)
        wanted = SourceURL(name="odyssey", url=ODYSSEY.url, allocated_resources=Resources(cpu=4.0))
        report = declare(wanted, self.root, commit=True)
        self.assertFalse(report.blockers)
        self.assertEqual(
            report.rows[0]["differences"]["allocated_resources"][1], {"cpu": 4.0, "gpu_type": None, "gpu_count": None}
        )

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


if __name__ == "__main__":
    unittest.main()
