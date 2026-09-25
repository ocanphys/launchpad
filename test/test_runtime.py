"""system.runtime.initialize_worker: what a call leaves behind, on every way
out, with the Modal Dicts and the mount stubbed."""

import logging
import time
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from modal.exception import InputCancellation

from system import lease_protocol, logs, runtime


class Dict(dict):
    put = dict.__setitem__


class Volume:
    def __init__(self, reload_error: Exception | None = None):
        self.reload_error = reload_error
        self.commits = 0

    def reload(self) -> None:
        if self.reload_error:
            raise self.reload_error

    def commit(self) -> None:
        self.commits += 1


class Queue(list):
    """The `refreshes` Queue, keeping each message with what was already true
    when it was put: how many times the mount had committed, and what the
    call's beat said -- None for a call that has not beaten, "live" or
    "exited" otherwise. A launcher sent to look before those are true is a
    launcher that finds what it already knew."""

    volume: Volume
    beats: dict

    def put(self, message: dict) -> None:
        beat = self.beats.get(message["call_id"])
        self.append({
            **message,
            "commits": self.volume.commits,
            "beat": beat and ("exited" if beat["exited"] else "live"),
        })


class InitializeWorkerTests(unittest.TestCase):
    ARTIFACT = "runs/toy/pretraining"

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.beats, self.call_logs, self.refreshes = Dict(), Dict(), Queue()
        self.refreshes.beats = self.beats
        self.patches = [
            patch("modal.current_function_call_id", return_value=None),  # the call is "local"
            patch.object(logs, "HEARTBEAT_SECONDS", 0.01),  # the log publisher's cadence
            patch.object(runtime, "HEARTBEAT_SECONDS", 0.01),  # the heartbeat's own
            patch.object(logs, "call_logs", self.call_logs),
            patch.object(runtime, "beats", self.beats),
            patch.object(runtime, "refreshes", self.refreshes),
            patch.object(lease_protocol, "leases", {self.ARTIFACT: lease_protocol.new_grant("local", "Pretraining")}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        logging.getLogger().handlers.clear()
        self.directory.cleanup()

    def mount(self, **kwargs) -> Volume:
        """The stubbed mount, which the queue stub reads its commit count off."""
        self.refreshes.volume = volume = Volume(**kwargs)
        return volume

    def container(self) -> list[str]:
        return [row["msg"].splitlines()[0] for row in self.call_logs["local:livedict:worker"]]

    def test_a_reload_that_fails_is_still_a_logged_call(self):
        volume = self.mount(reload_error=RuntimeError("open files on the mount"))
        with self.assertRaises(RuntimeError), runtime.initialize_worker(self.ARTIFACT, volume):
            self.fail("the body must not run on a mount that did not reload")
        self.assertEqual(self.container()[0], "failed under a held lease")
        self.assertIn("open files on the mount", self.call_logs["local:livedict:worker"][0]["msg"])
        self.assertEqual(self.beats["local"]["artifact_path"], self.ARTIFACT)
        self.assertEqual(volume.commits, 1)
        self.assertEqual(self.refreshes[-1], {"artifact_path": self.ARTIFACT, "call_id": "local", "event": "failed", "commits": 1, "beat": "exited"})

    def test_a_held_exception_does_not_pin_the_frames_that_raised(self):
        tokens = self.root / "tokens.bin"
        tokens.write_bytes(np.arange(8, dtype=np.uint16).tobytes())
        ref = None

        def run():
            nonlocal ref
            stream = np.memmap(tokens, dtype=np.uint16, mode="r")
            ref = weakref.ref(stream)
            raise ValueError(f"boom at {stream[0]}")

        held = None
        try:
            with runtime.initialize_worker(self.ARTIFACT, self.mount()):
                run()
        except ValueError as exc:
            held = exc  # a runtime that keeps the exception object, as Modal's does
        self.assertIsNotNone(held)
        self.assertIsNone(ref(), "the memmap outlived the call")

    def test_a_finished_call_publishes_its_rows_and_opens_no_file(self):
        volume = self.mount()
        with runtime.initialize_worker(self.ARTIFACT, volume) as worker:
            worker.log.info("counting")
        self.assertEqual(
            self.container()[1:],
            ["counting", "commit: lease held (attempt 1, try 1/5)", "done", "committed (done); telling the launcher"],
        )
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(volume.commits, 1)

    def test_a_finished_call_tells_the_launcher_after_it_has_committed(self):
        with runtime.initialize_worker(self.ARTIFACT, self.mount()):
            pass
        self.assertEqual(self.refreshes[-1], {"artifact_path": self.ARTIFACT, "call_id": "local", "event": "done", "commits": 1, "beat": "exited"})

    def test_a_call_that_lost_its_lease_says_so_in_its_message(self):
        lease_protocol.leases[self.ARTIFACT] = lease_protocol.new_grant("someone-else", "Pretraining")
        with self.assertRaises(lease_protocol.LeaseLost), runtime.initialize_worker(self.ARTIFACT, self.mount()):
            self.fail("the body must not run under a lease held by another call")
        self.assertEqual(self.refreshes[-1]["event"], "lease lost")

    def test_a_cancelled_call_is_a_call_that_lost_its_lease(self):
        """`cancel_call` drops the grant and then asks Modal to stop the call,
        so a cancel reaches the container as a fence that finds no grant. A
        read that succeeded and found nothing is an answer, so the call stops
        on the first try rather than waiting out LEASE_RETRIES backoffs."""
        lease_protocol.leases.clear()
        started = time.monotonic()
        with self.assertRaises(lease_protocol.LeaseLost), runtime.initialize_worker(self.ARTIFACT, self.mount()):
            self.fail("the body must not run with no grant naming this call")
        self.assertLess(time.monotonic() - started, lease_protocol.LEASE_BACKOFF)
        self.assertIn("this call was cancelled", self.container()[0])
        self.assertEqual(self.refreshes[-1]["event"], "lease lost")

    def test_the_cancellation_modal_delivers_is_the_same_ending(self):
        """Modal raises its own cancellation into the call whenever the
        container's heartbeat next reaches it, which can be after the fence has
        already stopped the job -- or, between fences, before it."""
        with self.assertRaises(InputCancellation), runtime.initialize_worker(self.ARTIFACT, self.mount()):
            raise InputCancellation("Input was cancelled by user")
        self.assertEqual(self.refreshes[-1]["event"], "lease lost")

    def test_two_calls_on_one_artifact_stage_through_different_files(self):
        """The window `cancel_call` opens: it drops the lease before Modal stops
        the container, so a launch made in between is granted while the old call
        may still be writing. Each call stages through its own temporary file,
        so their bytes never mix, and only the one holding the lease renames."""
        def lost(label: str) -> None:
            raise lease_protocol.LeaseLost(label)

        path = self.root / "tokens.bin"
        log = logging.getLogger("test")
        holder = runtime.Worker(self.ARTIFACT, "fc-new", log, lambda label: None, {})
        superseded = runtime.Worker(self.ARTIFACT, "fc-old", log, lost, {})
        with self.assertRaises(lease_protocol.LeaseLost):
            with superseded.publishing(path) as stale, holder.publishing(path) as fresh:
                stale.write(b"stale")
                fresh.write(b"fresh")
                staged = sorted(p.name for p in self.root.iterdir())

        self.assertEqual(staged, ["tokens.bin.fc-new.tmp", "tokens.bin.fc-old.tmp"])
        self.assertEqual(path.read_bytes(), b"fresh")
        self.assertEqual(list(self.root.glob("*.tmp")), [], "the superseded call left its file behind")

    def test_a_running_call_announces_itself_with_a_beat_already_readable(self):
        """The message that turns the row the launcher granted from
        "starting" into "running". The heartbeat's first pass beats before it
        sends it, rather than waiting out an interval first: a launcher that
        reloaded while this call had no heartbeat would find the starting call
        it already knew about. Sent once, however long the call runs."""
        with runtime.initialize_worker(self.ARTIFACT, self.mount()) as worker:
            worker.log.info("working")
            beaten = set()
            while len(beaten) < 3:  # several heartbeat passes, none of which announces again
                beaten.add(self.beats["local"]["last_beat_ts"])
        self.assertEqual(self.refreshes[0], {"artifact_path": self.ARTIFACT, "call_id": "local", "event": "started", "commits": 0, "beat": "live"})
        self.assertEqual([message["event"] for message in self.refreshes], ["started", "done"])

    def test_a_call_shorter_than_one_heartbeat_announces_both_ends(self):
        """Neither message waits out a heartbeat interval, so a call that is
        over before the first wait still tells the launcher twice, and the
        beat the last one trails says the call has gone."""
        with patch.object(runtime, "HEARTBEAT_SECONDS", 30), runtime.initialize_worker(self.ARTIFACT, self.mount()):
            pass
        self.assertEqual([message["event"] for message in self.refreshes], ["started", "done"])
        self.assertEqual(self.refreshes[-1]["beat"], "exited")

    def test_the_last_beat_says_the_call_has_exited(self):
        """What keeps a finished call from reading as live until its beats
        flatline: the pass after the call ends marks the beat, and only it."""
        beats = []
        with runtime.initialize_worker(self.ARTIFACT, self.mount()):
            while len(beats) < 2:  # two live passes, so `exited` is not just "the first beat"
                if self.beats.get("local") and self.beats["local"] not in beats:
                    beats.append(dict(self.beats["local"]))
        self.assertEqual([beat["exited"] for beat in beats], [False, False])
        self.assertTrue(self.beats["local"]["exited"])


if __name__ == "__main__":
    unittest.main()
