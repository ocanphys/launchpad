"""system.runtime.initialize_worker: what a call leaves behind, on every way
out, with the Modal Dicts and the mount stubbed."""

import logging
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from system import lease_protocol, runtime


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


class InitializeWorkerTests(unittest.TestCase):
    ARTIFACT = "runs/toy/pretraining"

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.beats, self.call_logs = Dict(), Dict()
        self.patches = [
            patch("modal.current_function_call_id", return_value=None),  # the call is "local"
            patch.object(runtime, "HEARTBEAT_SECONDS", 0.01),
            patch.object(runtime, "beats", self.beats),
            patch.object(runtime, "call_logs", self.call_logs),
            patch.object(lease_protocol, "leases", {self.ARTIFACT: lease_protocol.new_grant("local", "Pretraining")}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        logging.getLogger().handlers.clear()
        self.directory.cleanup()

    def container(self) -> list[str]:
        return [row["msg"].splitlines()[0] for row in self.call_logs["local:container"]]

    def test_a_reload_that_fails_is_still_a_logged_call(self):
        volume = Volume(reload_error=RuntimeError("open files on the mount"))
        with self.assertRaises(RuntimeError), runtime.initialize_worker(self.ARTIFACT, volume):
            self.fail("the body must not run on a mount that did not reload")
        self.assertEqual(self.container(), ["failed under a held lease"])
        self.assertIn("open files on the mount", self.call_logs["local:container"][0]["msg"])
        self.assertEqual(self.beats["local"]["artifact_path"], self.ARTIFACT)
        self.assertEqual(volume.commits, 1)

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
            with runtime.initialize_worker(self.ARTIFACT, Volume()):
                run()
        except ValueError as exc:
            held = exc  # a runtime that keeps the exception object, as Modal's does
        self.assertIsNotNone(held)
        self.assertIsNone(ref(), "the memmap outlived the call")

    def test_a_finished_call_publishes_its_rows_and_opens_no_file(self):
        volume = Volume()
        with runtime.initialize_worker(self.ARTIFACT, volume) as worker:
            worker.log.info("counting")
        self.assertEqual(self.container()[1:], ["counting", "commit: lease held (attempt 1, try 1/5)", "done"])
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(volume.commits, 1)


if __name__ == "__main__":
    unittest.main()
