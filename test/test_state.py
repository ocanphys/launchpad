"""One state entry per declared path, against a temporary root."""

import io
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import lab
import main
from artifacts.core.artifact import MANIFEST, Artifact
from artifacts.sources import Source
from artifacts.tokenizers.bpe import TokenizedSource, Tokenizer

ODYSSEY = Source(name="odyssey", url="https://example.org/odyssey.txt")
TOKENIZER = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(ODYSSEY,))
TOKENS = TokenizedSource(tokenizer=TOKENIZER, source=ODYSSEY)


@dataclass(frozen=True)
class Virtual(Artifact):
    """Owns no files and nothing produces it: done when its sources are."""

    producer: ClassVar[str | None] = None

    sources: tuple[Source, ...]

    @property
    def uid(self) -> str:
        return "virtual"

    @property
    def artifact_path(self) -> Path:
        return Path("virtual")

    @property
    def files(self) -> dict[str, str]:
        return {}

    def completion_paths(self, root: Path) -> list[Path]:
        return [source.paths(root)["raw text"] for source in self.sources]


def declare(artifact, root):
    with redirect_stdout(io.StringIO()):
        lab.declare(artifact, root=root, commit=True)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        main.leases, main.beats = {}, {}  # plain dicts answer .items() like the Dicts do

    def tearDown(self):
        self.directory.cleanup()

    def test_one_entry_per_manifest_with_blocking_from_the_same_map(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey/body.txt").touch()
        entries = main.state(self.root)
        tokenizer = f"tokenizers/{TOKENIZER.uid}"
        self.assertEqual(set(entries), {"sources/odyssey", tokenizer, TOKENS.artifact_path.as_posix()})
        self.assertEqual(entries["sources/odyssey"]["status"], "done")
        self.assertTrue(entries[tokenizer]["ready"])
        self.assertEqual(entries[tokenizer]["depends_on"], ["sources/odyssey"])
        tokens = entries[TOKENS.artifact_path.as_posix()]
        self.assertEqual(tokens["blocked_by"], [tokenizer])
        self.assertFalse(tokens["ready"])
        self.assertEqual(tokens["durable_progress"], {"phase": "files", "done": 0, "total": 1})

    def test_a_dependency_with_no_manifest_blocks(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey" / MANIFEST).unlink()
        entries = main.state(self.root)
        self.assertNotIn("sources/odyssey", entries)
        self.assertEqual(entries[f"tokenizers/{TOKENIZER.uid}"]["blocked_by"], ["sources/odyssey"])

    def test_a_corrupt_manifest_is_its_own_error_entry_only(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey" / MANIFEST).write_text("{not json")
        entries = main.state(self.root)
        broken = entries["sources/odyssey"]
        self.assertEqual(broken["status"], "conflict")
        self.assertIn("not readable JSON", broken["error"])
        self.assertFalse(broken["ready"])
        self.assertEqual(entries[f"tokenizers/{TOKENIZER.uid}"]["status"], "declared")

    def test_an_artifact_nothing_produces_is_never_ready(self):
        declare(Virtual(sources=(ODYSSEY,)), self.root)
        entries = main.state(self.root)
        self.assertEqual(entries["virtual"]["status"], "declared")
        self.assertFalse(entries["virtual"]["ready"])
        (self.root / "sources/odyssey/body.txt").touch()
        self.assertTrue(main.state(self.root)["virtual"]["done"])

    def test_the_lease_snapshot_rides_along(self):
        declare(ODYSSEY, self.root)
        main.leases = {"sources/odyssey": {"call_id": "fc-1"}}
        main.beats = {"fc-1": {"last_beat_ts": 10**12, "progress": {"bytes": 3}}}
        entry = main.state(self.root)["sources/odyssey"]
        self.assertEqual(entry["call_id"], "fc-1")
        self.assertTrue(entry["active"])
        self.assertEqual(entry["live_progress"], {"bytes": 3})


if __name__ == "__main__":
    unittest.main()
