"""One state entry per declared path, against a temporary root.

Every fact on an entry comes from one snapshot of the leases and beats and
one image of the volume, so each test sets those three and reads the map
once. `now` is pinned so an entry's liveness is a function of the snapshot
alone.
"""

import io
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar
from unittest.mock import patch

import lab
import main
from artifacts.core.artifact import MANIFEST, Artifact
from artifacts.dataset import DataSet
from artifacts.sources import SourceURL
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer
from config import FLATLINE, HEARTBEAT_SECONDS, STARTUP_GRACE_SECONDS

ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
TOKENIZER = Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=(ODYSSEY,))
TOKENS = TokenizedSource(tokenizer=TOKENIZER, source=ODYSSEY)
TOKENIZER_PATH = f"tokenizers/{TOKENIZER.uid}"
DATASET = DataSet(train_set=(TOKENS,), valid_set=(TOKENS,))  # two files of its own

NOW = 1_000_000.0
FLATLINE_SECONDS = FLATLINE * HEARTBEAT_SECONDS


@dataclass(frozen=True)
class Virtual(Artifact):
    """Owns no files and nothing produces it: done when its sources are."""

    producer: ClassVar[str | None] = None

    sources: tuple[Artifact, ...]

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
        return [path for source in self.sources for path in source.completion_paths(root)]


def declare(artifact, root):
    with redirect_stdout(io.StringIO()):
        lab.declare(artifact, root=root, commit=True)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        main.leases, main.beats = {}, {}  # plain dicts answer .items() like the Dicts do
        main.resolved.clear()  # every test declares its own root under the same paths
        clock = patch.object(main.time, "time", lambda: NOW)
        clock.start()
        self.addCleanup(clock.stop)

    def tearDown(self):
        self.directory.cleanup()

    def state(self):
        return main.state(self.root)

    def build(self, artifact, count=None):
        """Touches `count` of `artifact`'s files (all of them by default)."""
        for path in list(artifact.paths(self.root).values())[:count]:
            path.touch()

    def lease(self, path, call_id, beat_age=None, progress=None, grant_age=STARTUP_GRACE_SECONDS + 1):
        """One granted lease, beating `beat_age` seconds ago, or never."""
        main.leases[path] = {"call_id": call_id, "granted_ts": NOW - grant_age}
        if beat_age is not None:
            main.beats[call_id] = {"last_beat_ts": NOW - beat_age, "progress": progress}

    # --- the map ---------------------------------------------------------------

    def test_an_empty_root_is_an_empty_map(self):
        self.assertEqual(self.state(), {})

    def test_one_entry_per_manifest_with_blocking_from_the_same_map(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey/body.txt").touch()
        entries = self.state()
        self.assertEqual(set(entries), {"sources/odyssey", TOKENIZER_PATH, TOKENS.artifact_path.as_posix()})
        self.assertEqual(entries["sources/odyssey"]["verdict"], "done")
        self.assertEqual(entries[TOKENIZER_PATH]["verdict"], "runnable")
        self.assertEqual(entries[TOKENIZER_PATH]["depends_on"], ["sources/odyssey"])
        tokens = entries[TOKENS.artifact_path.as_posix()]
        self.assertEqual(tokens["blocked_by"], [TOKENIZER_PATH])
        self.assertEqual(tokens["verdict"], "blocked")
        self.assertEqual(tokens["durable_progress"], {"phase": "files", "done": 0, "total": 1})

    def test_every_entry_has_the_same_keys(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey" / MANIFEST).write_text("{not json")
        keys = {frozenset(entry) for entry in self.state().values()}
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys.pop(), {
            "type", "status", "error", "depends_on", "parameters", "blocked_by", "done", "ready",
            "verdict", "call_id", "active", "last_heartbeat", "live_progress", "durable_progress",
        })

    def test_parameters_are_the_artifacts_own_fields_only(self):
        declare(TOKENS, self.root)
        entry = self.state()[TOKENIZER_PATH]
        self.assertEqual(entry["type"], "Tokenizer")
        self.assertEqual(entry["parameters"], {"special_tokens": ["<pad>"], "vocab_size": 1000})

    def test_a_partial_artifact_with_its_dependencies_done_is_runnable(self):
        declare(DATASET, self.root)
        for dependency in (ODYSSEY, TOKENIZER, TOKENS):
            self.build(dependency)
        self.build(DATASET, count=1)
        entry = self.state()[DATASET.artifact_path.as_posix()]
        self.assertEqual(entry["status"], "partial")
        self.assertEqual(entry["verdict"], "runnable")
        self.assertEqual(entry["durable_progress"], {"phase": "files", "done": 1, "total": 2})

    # --- dependencies ----------------------------------------------------------

    def test_a_dependency_with_no_manifest_blocks(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey" / MANIFEST).unlink()
        entries = self.state()
        self.assertNotIn("sources/odyssey", entries)
        self.assertEqual(entries[TOKENIZER_PATH]["blocked_by"], ["sources/odyssey"])
        self.assertEqual(entries[TOKENIZER_PATH]["verdict"], "blocked")

    def test_a_dependency_that_is_partial_blocks(self):
        declare(Virtual(sources=(DATASET,)), self.root)
        for dependency in (ODYSSEY, TOKENIZER, TOKENS):
            self.build(dependency)
        self.build(DATASET, count=1)
        dataset = DATASET.artifact_path.as_posix()
        entries = self.state()
        self.assertEqual(entries[dataset]["status"], "partial")
        self.assertEqual(entries["virtual"]["blocked_by"], [dataset])

    def test_a_corrupt_dependency_blocks_and_is_its_own_error_entry_only(self):
        declare(TOKENS, self.root)
        (self.root / "sources/odyssey" / MANIFEST).write_text("{not json")
        entries = self.state()
        broken = entries["sources/odyssey"]
        self.assertEqual(broken["status"], "conflict")
        self.assertIn("not readable JSON", broken["error"])
        self.assertEqual(broken["verdict"], "failed")
        self.assertFalse(broken["ready"])
        self.assertEqual(broken["type"], None)
        self.assertEqual(broken["durable_progress"], None)
        tokenizer = entries[TOKENIZER_PATH]
        self.assertEqual(tokenizer["status"], "declared")
        self.assertEqual(tokenizer["blocked_by"], ["sources/odyssey"])

    def test_a_manifest_copied_to_the_wrong_folder_is_a_conflict(self):
        declare(ODYSSEY, self.root)
        copy = self.root / "sources/iliad"
        copy.mkdir()
        (copy / MANIFEST).write_text((self.root / "sources/odyssey" / MANIFEST).read_text())
        entry = self.state()["sources/iliad"]
        self.assertEqual(entry["status"], "conflict")
        self.assertIn("describes sources/odyssey", entry["error"])
        self.assertEqual(entry["verdict"], "failed")

    def test_an_artifact_nothing_produces_is_never_ready(self):
        declare(Virtual(sources=(ODYSSEY,)), self.root)
        entries = self.state()
        self.assertEqual(entries["virtual"]["status"], "declared")
        self.assertFalse(entries["virtual"]["ready"])
        self.assertEqual(entries["virtual"]["verdict"], "blocked")
        (self.root / "sources/odyssey/body.txt").touch()
        entry = self.state()["virtual"]
        self.assertTrue(entry["done"])
        self.assertEqual(entry["verdict"], "done")

    def test_a_path_once_read_is_not_read_again(self):
        declare(ODYSSEY, self.root)
        self.assertEqual(self.state()["sources/odyssey"]["type"], "SourceURL")
        (self.root / "sources/odyssey" / MANIFEST).write_text("{not json")
        self.assertEqual(self.state()["sources/odyssey"]["type"], "SourceURL")
        main.resolved.clear()
        self.assertEqual(self.state()["sources/odyssey"]["verdict"], "failed")

    # --- leases and beats --------------------------------------------------------

    def test_a_beating_lease_is_running_with_its_live_progress(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=1, progress={"bytes": 3})
        entry = self.state()["sources/odyssey"]
        self.assertEqual(entry["call_id"], "fc-1")
        self.assertTrue(entry["active"])
        self.assertEqual(entry["last_heartbeat"], NOW - 1)
        self.assertEqual(entry["live_progress"], {"bytes": 3})
        self.assertEqual(entry["verdict"], "running")
        self.assertTrue(entry["ready"])

    def test_a_lease_that_never_beat_is_failed_and_still_ready(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1")
        entry = self.state()["sources/odyssey"]
        self.assertEqual(entry["call_id"], "fc-1")
        self.assertFalse(entry["active"])
        self.assertEqual(entry["last_heartbeat"], None)
        self.assertEqual(entry["live_progress"], None)
        self.assertEqual(entry["verdict"], "failed")
        self.assertTrue(entry["ready"])

    def test_a_recent_lease_starts_without_a_modal_lookup(self):
        declare(ODYSSEY, self.root)
        with patch.object(main.modal.FunctionCall, "from_id", side_effect=AssertionError("Modal lookup")) as lookup:
            self.assertEqual(self.state()["sources/odyssey"]["verdict"], "runnable")
            self.lease("sources/odyssey", "fc-1", grant_age=0)
            entry = self.state()["sources/odyssey"]
        lookup.assert_not_called()
        self.assertEqual(entry["verdict"], "starting")
        self.assertFalse(entry["active"])
        self.assertIsNone(entry["last_heartbeat"])
        self.assertIsNone(entry["live_progress"])
        self.assertTrue(entry["ready"])

    def test_startup_expires_at_the_configured_grace_boundary(self):
        declare(ODYSSEY, self.root)
        with patch.object(main, "STARTUP_GRACE_SECONDS", 10):
            for age, verdict in ((9.999, "starting"), (10, "failed"), (11, "failed")):
                with self.subTest(age=age):
                    self.lease("sources/odyssey", "fc-1", grant_age=age)
                    self.assertEqual(self.state()["sources/odyssey"]["verdict"], verdict)

    def test_a_grant_from_the_future_is_starting(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", grant_age=-30)
        self.assertEqual(self.state()["sources/odyssey"]["verdict"], "starting")

    def test_a_grant_without_a_timestamp_is_not_starting(self):
        declare(ODYSSEY, self.root)
        for grant in ({"call_id": "fc-1"}, {"call_id": "fc-1", "granted_ts": None}):
            with self.subTest(grant=grant):
                main.leases["sources/odyssey"] = grant
                self.assertEqual(self.state()["sources/odyssey"]["verdict"], "failed")

    def test_any_heartbeat_ends_startup_even_before_the_grace_expires(self):
        declare(ODYSSEY, self.root)
        with patch.object(main, "STARTUP_GRACE_SECONDS", FLATLINE_SECONDS * 3):
            for beat_age, verdict in ((1, "running"), (FLATLINE_SECONDS + 1, "failed")):
                with self.subTest(beat_age=beat_age):
                    self.lease("sources/odyssey", "fc-1", beat_age=beat_age,
                               grant_age=FLATLINE_SECONDS + 2)
                    self.assertEqual(self.state()["sources/odyssey"]["verdict"], verdict)

    def test_starting_only_reaches_its_own_artifact_in_either_scan_position(self):
        other = SourceURL(name="zephyr", url="https://example.org/zephyr.txt")
        declare(ODYSSEY, self.root)
        declare(other, self.root)
        paths = (ODYSSEY.artifact_path.as_posix(), other.artifact_path.as_posix())
        for starting_path, idle_path in (paths, paths[::-1]):
            with self.subTest(starting_path=starting_path):
                main.leases.clear()
                self.lease(starting_path, "fc-1", grant_age=0)
                entries = self.state()
                self.assertEqual(entries[starting_path]["verdict"], "starting")
                self.assertEqual(entries[idle_path]["verdict"], "runnable")

    def test_launch_checks_declaration_only_after_startup_grace_expires(self):
        path = "sources/odyssey"
        with patch.object(main.modal.FunctionCall, "from_id", side_effect=AssertionError("Modal lookup")) as lookup:
            self.lease(path, "fc-1", grant_age=STARTUP_GRACE_SECONDS - 0.001)
            with patch.object(main.Artifact, "load", side_effect=AssertionError("volume read")):
                launched, message = main.attempt_launch(path, self.root)
            self.assertFalse(launched)
            self.assertIn("already a call running", message)

            self.lease(path, "fc-1", grant_age=STARTUP_GRACE_SECONDS)
            launched, message = main.attempt_launch(path, self.root)
            self.assertFalse(launched)
            self.assertIn("nothing declared", message)
        lookup.assert_not_called()

    def test_a_launch_refused_by_a_running_call_is_logged_under_that_call(self):
        """Not the launcher's own row: it belongs on the page of the call
        holding the artifact, beside what that call was doing when someone
        tried to relaunch it."""
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=0)
        with self.assertLogs("leasebook") as captured:
            launched, message = main.attempt_launch("sources/odyssey", self.root)
        self.assertFalse(launched)
        self.assertIn("already a call running", message)
        self.assertEqual([(r.call_id, r.source) for r in captured.records], [("fc-1", "launcher")])

    def test_launch_refuses_what_the_map_would_not_call_runnable(self):
        declare(TOKENS, self.root)
        self.build(ODYSSEY)
        launched, message = main.attempt_launch(TOKENS.artifact_path.as_posix(), self.root)
        self.assertFalse(launched)
        self.assertEqual(message, f"{TOKENS.artifact_path.as_posix()}: blocked on ['{TOKENIZER_PATH}']")
        launched, message = main.attempt_launch("sources/odyssey", self.root)
        self.assertFalse(launched)
        self.assertEqual(message, "sources/odyssey: already done")

    def test_a_stale_beat_is_failed_and_keeps_its_last_heartbeat_but_not_its_progress(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=FLATLINE_SECONDS + 1, progress={"bytes": 3})
        entry = self.state()["sources/odyssey"]
        self.assertFalse(entry["active"])
        self.assertEqual(entry["last_heartbeat"], NOW - FLATLINE_SECONDS - 1)
        self.assertEqual(entry["live_progress"], None)
        self.assertEqual(entry["durable_progress"], {"phase": "files", "done": 0, "total": 1})
        self.assertEqual(entry["verdict"], "failed")

    def test_the_flatline_is_a_closed_boundary(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=FLATLINE_SECONDS)
        self.assertFalse(self.state()["sources/odyssey"]["active"])
        main.beats["fc-1"]["last_beat_ts"] = NOW - FLATLINE_SECONDS + 0.001
        self.assertTrue(self.state()["sources/odyssey"]["active"])

    def test_a_slow_snapshot_does_not_age_a_beat(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=1)
        clock = [NOW]

        class Slow(dict):
            def items(self):  # the scan takes longer than the whole window
                clock[0] += FLATLINE_SECONDS * 2
                return dict.items(self)

        main.beats = Slow(main.beats)
        with patch.object(main.time, "time", lambda: clock[0]):
            self.assertTrue(self.state()["sources/odyssey"]["active"])

    def test_a_beat_from_the_future_is_live(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=-30)
        self.assertTrue(self.state()["sources/odyssey"]["active"])

    def test_a_call_that_marked_its_last_beat_exited_is_not_running(self):
        """A worker marks the beat it leaves on the way out, so the row stops
        saying "running" the moment the call is over rather than a flatline
        later -- while its files, committed just before, are already there."""
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=0)
        main.beats["fc-1"]["exited"] = True
        entry = self.state()["sources/odyssey"]
        self.assertFalse(entry["active"])
        self.assertEqual(entry["last_heartbeat"], NOW)
        self.assertEqual(entry["verdict"], "failed")
        self.build(ODYSSEY)
        self.assertEqual(self.state()["sources/odyssey"]["verdict"], "done")

    def test_a_beat_with_no_progress_is_live_without_one(self):
        declare(ODYSSEY, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=1)
        del main.beats["fc-1"]["progress"]
        entry = self.state()["sources/odyssey"]
        self.assertTrue(entry["active"])
        self.assertEqual(entry["live_progress"], None)

    def test_a_beat_whose_lease_was_released_is_no_call_at_all(self):
        declare(ODYSSEY, self.root)
        main.beats["fc-1"] = {"last_beat_ts": NOW, "progress": {"bytes": 3}}
        entry = self.state()["sources/odyssey"]
        self.assertEqual(entry["call_id"], None)
        self.assertFalse(entry["active"])
        self.assertEqual(entry["last_heartbeat"], None)
        self.assertEqual(entry["live_progress"], None)
        self.assertEqual(entry["verdict"], "runnable")

    def test_a_lease_only_reaches_its_own_artifact(self):
        declare(TOKENS, self.root)
        self.lease("sources/odyssey", "fc-1", beat_age=1)
        entries = self.state()
        self.assertEqual(entries["sources/odyssey"]["verdict"], "running")
        self.assertEqual(entries[TOKENIZER_PATH]["call_id"], None)
        self.assertEqual(entries[TOKENIZER_PATH]["verdict"], "blocked")

    # --- verdict precedence --------------------------------------------------------

    def test_done_wins_over_a_live_call(self):
        declare(ODYSSEY, self.root)
        (self.root / "sources/odyssey/body.txt").touch()
        self.lease("sources/odyssey", "fc-1", beat_age=1)
        entry = self.state()["sources/odyssey"]
        self.assertTrue(entry["active"])
        self.assertEqual(entry["verdict"], "done")
        self.assertFalse(entry["ready"])

    def test_done_wins_over_a_stale_lease(self):
        declare(ODYSSEY, self.root)
        (self.root / "sources/odyssey/body.txt").touch()
        self.lease("sources/odyssey", "fc-1", beat_age=FLATLINE_SECONDS + 1)
        self.assertEqual(self.state()["sources/odyssey"]["verdict"], "done")

    def test_a_live_call_wins_over_a_conflict(self):
        declare(ODYSSEY, self.root)
        (self.root / "sources/odyssey" / MANIFEST).write_text("{not json")
        self.lease("sources/odyssey", "fc-1", beat_age=1)
        entry = self.state()["sources/odyssey"]
        self.assertEqual(entry["status"], "conflict")
        self.assertEqual(entry["verdict"], "running")

    def test_a_stale_lease_wins_over_blocked(self):
        declare(TOKENS, self.root)
        self.lease(TOKENIZER_PATH, "fc-1", beat_age=FLATLINE_SECONDS + 1)
        entry = self.state()[TOKENIZER_PATH]
        self.assertEqual(entry["blocked_by"], ["sources/odyssey"])
        self.assertFalse(entry["ready"])
        self.assertEqual(entry["verdict"], "failed")


if __name__ == "__main__":
    unittest.main()
