"""system.logs: the buffer that keeps one call's records, and the two passes
that move log rows between the Dicts and the volume."""

import json
import logging
import threading
import time
import unittest
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from system import logs
from system.logs import (
    BufferHandler,
    launcher_log,
    load_snapshot_from_volume,
    save_snapshot_to_volume,
)


class Dict(dict):
    put = dict.__setitem__


class Volume:
    commits = 0

    def commit(self) -> None:
        self.commits += 1


def row(msg: str, ts: float = 1.0) -> dict:
    return {"ts": ts, "level": "INFO", "logger": "job", "msg": msg}


class BufferHandlerTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("job")
        self.logger.setLevel(logging.DEBUG)
        self.handlers = []

    def tearDown(self):
        for handler in self.handlers:
            logging.getLogger().removeHandler(handler)

    def attach(self, call_id: str) -> BufferHandler:
        buffer = BufferHandler(call_id)
        logging.getLogger().addHandler(buffer)
        self.handlers.append(buffer)
        return buffer

    def test_a_traceback_is_folded_into_the_row(self):
        with patch("modal.current_function_call_id", return_value="fc-a"):
            buffer = self.attach("fc-a")
            self.logger.info("started %d", 3)
            try:
                raise ValueError("boom")
            except ValueError:
                self.logger.exception("failed")
        self.assertEqual([r["msg"].splitlines()[0] for r in buffer.rows], ["started 3", "failed"])
        self.assertIn("ValueError: boom", buffer.rows[1]["msg"])
        self.assertEqual((buffer.rows[0]["level"], buffer.rows[0]["logger"]), ("INFO", "job"))

    def test_each_call_keeps_only_its_own_records(self):
        with patch("modal.current_function_call_id", return_value="fc-a"):
            a, b = self.attach("fc-a"), self.attach("fc-b")
            self.logger.info("from a")
        with patch("modal.current_function_call_id", return_value="fc-b"):
            self.logger.info("from b")
        self.assertEqual([r["msg"] for r in a.rows], ["from a"])
        self.assertEqual([r["msg"] for r in b.rows], ["from b"])

    def test_outside_a_container_the_call_is_local(self):
        with patch("modal.current_function_call_id", return_value=None):
            buffer = self.attach("local")
            self.logger.info("here")
        self.assertEqual(len(buffer.rows), 1)


class LauncherLoggingTests(unittest.TestCase):
    """`start_launcher_logging`: the whole container's records, republished
    whole on every tick, once more on stop, and a tick that fails is a row."""

    def setUp(self):
        self.store = Dict()
        self.enterContext(patch.object(logs, "setup_logging"))
        self.enterContext(patch.object(logs, "call_logs", self.store))
        self.enterContext(patch.object(logs, "HEARTBEAT_SECONDS", 3600))  # only the stop publishes
        self.stop = logs.start_launcher_logging()

    def tearDown(self):
        self.stop()  # a second stop after the test's own is a no-op join
        logging.getLogger().handlers.clear()

    def messages(self) -> list[str]:
        return [r["msg"].splitlines()[0] for r in self.store["launcher"]]

    def test_every_record_from_any_call_context_or_thread_lands_whole(self):
        call = ContextVar("test_call", default=None)
        with patch("modal.current_function_call_id", side_effect=call.get):
            logging.getLogger("leasebook").debug("container boot")
            call.set("fc-request-a")
            logging.getLogger("leasebook").info("first request")

            def background():
                try:
                    raise ValueError("background boom")
                except ValueError:
                    logging.getLogger("leasebook").exception("background failed")

            thread = threading.Thread(target=background)
            thread.start()
            thread.join()
        self.stop()
        self.assertEqual(self.messages(), ["container boot", "first request", "background failed"])
        rows = self.store["launcher"]
        self.assertEqual(rows[0]["level"], "DEBUG")
        self.assertIn("ValueError: background boom", rows[-1]["msg"])
        self.assertEqual(set(rows[-1]), {"ts", "level", "logger", "msg"})

    def test_a_publish_that_fails_is_a_warning_the_next_one_carries(self):
        self.stop()
        failures = [RuntimeError("Dict unavailable")]

        def put(key, value):
            if failures:
                raise failures.pop()
            self.store[key] = value

        with patch.object(logs, "HEARTBEAT_SECONDS", 0.01), patch.object(self.store, "put", put):
            self.stop = logs.start_launcher_logging()
            logging.getLogger("leasebook").info("still serving")
            deadline = time.time() + 2
            while not self.store.get("launcher") and time.time() < deadline:
                time.sleep(0.01)
            self.stop()
        self.assertEqual(sorted(self.messages()), ["launcher log not published (Dict unavailable)", "still serving"])

    def test_a_new_container_continues_the_longer_of_the_last_list_and_the_file(self):
        self.stop()
        for launcher, volume in (([row("a"), row("b")], [row("a")]), ([row("c")], [row("a"), row("b")])):
            with self.subTest(launcher=launcher):
                self.store.update({"launcher": launcher, "launcher:volume": volume})
                self.stop = logs.start_launcher_logging()
                logging.getLogger("leasebook").info("started")
                self.stop()
                self.assertEqual(self.messages(), ["a", "b", "started"])

    def test_launcher_log_reaches_the_call_channel_and_the_launcher_log(self):
        launcher_log("fc-a", "granted lease")
        self.stop()
        self.assertEqual([r["msg"] for r in self.store["fc-a:launcher"]], ["granted lease"])
        self.assertEqual(self.messages(), ["granted lease"])


class SnapshotTests(unittest.TestCase):
    ARTIFACT = "runs/toy/pretraining"

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.call_logs, self.call_history = Dict(), Dict()
        self.volume = Volume()
        self.patches = [
            patch.object(logs, "call_logs", self.call_logs),
            patch.object(logs, "call_history", self.call_history),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.directory.cleanup()

    def file(self, call_id: str) -> Path:
        return self.root / self.ARTIFACT / "logs" / f"{call_id}.jsonl"

    def grant(self, call_id: str, ts: float) -> None:
        self.call_history.put(self.ARTIFACT, [*(self.call_history.get(self.ARTIFACT) or []), {"call_id": call_id, "granted_ts": ts, "artifact_type": "Pretraining"}])

    def test_a_pass_appends_only_what_is_new_and_the_volume_channel_follows(self):
        self.grant("fc-a", 10.0)
        launcher_log("fc-a", "granted")
        self.call_logs.put("fc-a:container", [row("boot"), row("step 1")])
        save_snapshot_to_volume(self.root, self.volume, {})
        self.call_logs.put("fc-a:container", [row("boot"), row("step 1"), row("done")])
        # each pass rebuilds the cursor from the files, as persist_logs does
        persisted = load_snapshot_from_volume(self.root)
        self.assertEqual(persisted, {"fc-a:launcher": 1, "fc-a:container": 2, "launcher": 0})
        save_snapshot_to_volume(self.root, self.volume, persisted)

        rows = [json.loads(line) for line in self.file("fc-a").read_text().splitlines()]
        self.assertEqual([(r["source"], r["msg"]) for r in rows], [("launcher", "granted"), ("container", "boot"), ("container", "step 1"), ("container", "done")])
        self.assertEqual(self.call_logs["fc-a:volume"], rows)
        self.assertEqual(json.loads((self.root / "call_history.json").read_text()), self.call_history)
        self.assertEqual(self.volume.commits, 2)

    def test_a_call_with_nothing_new_gets_no_file(self):
        self.grant("fc-a", 10.0)
        save_snapshot_to_volume(self.root, self.volume, {})
        self.assertFalse(self.file("fc-a").exists())

    def test_a_channel_that_came_back_shorter_moves_nothing(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:container", [row("boot"), row("done")])
        save_snapshot_to_volume(self.root, self.volume, {})
        self.call_logs.put("fc-a:container", [])  # a wiped Dict, republished empty
        persisted = load_snapshot_from_volume(self.root)
        self.assertEqual(persisted["fc-a:container"], 2)
        save_snapshot_to_volume(self.root, self.volume, persisted)
        self.assertEqual(len(self.file("fc-a").read_text().splitlines()), 2)

    def test_startup_reads_the_files_back_and_seeds_the_cursor(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:launcher", [row("granted", 10.0)])
        self.call_logs.put("fc-a:container", [row("boot", 11.0)])
        save_snapshot_to_volume(self.root, self.volume, {})
        volume_rows = self.call_logs["fc-a:volume"]

        # A fresh leasebook with the Dicts wiped, and one file no grant names.
        self.call_logs.clear()
        self.call_history.clear()
        (self.root / "sources/tiny/logs").mkdir(parents=True)
        (self.root / "sources/tiny/logs/fc-old.jsonl").write_text(json.dumps(row("hi")) + "\n" + '{"ts": 2.0, "lev')
        persisted = load_snapshot_from_volume(self.root)

        self.assertEqual(persisted, {"launcher": 0, "fc-a:launcher": 1, "fc-a:container": 1, "fc-old:launcher": 0, "fc-old:container": 0})
        self.assertEqual(self.call_logs["fc-a:volume"], volume_rows)
        self.assertEqual(self.call_logs["fc-old:volume"], [row("hi")])
        self.assertEqual(self.call_history[self.ARTIFACT], [{"call_id": "fc-a", "granted_ts": 10.0, "artifact_type": "Pretraining"}])
        self.assertEqual(self.call_history["sources/tiny"], [{"call_id": "fc-old", "granted_ts": None, "artifact_type": None}])

    def test_startup_merges_the_file_with_a_dict_that_is_ahead(self):
        (self.root / "call_history.json").write_text(json.dumps({self.ARTIFACT: [{"call_id": "fc-a", "granted_ts": 10.0, "artifact_type": "Pretraining"}]}))
        self.grant("fc-b", 20.0)  # granted after the last persist pass
        load_snapshot_from_volume(self.root)
        self.assertEqual([g["call_id"] for g in self.call_history[self.ARTIFACT]], ["fc-a", "fc-b"])

    def test_an_empty_volume_is_empty(self):
        self.assertEqual(load_snapshot_from_volume(self.root), {"launcher": 0})
        self.assertEqual(self.call_logs, {"launcher:volume": []})
        self.assertEqual(self.call_history, {})

    def test_the_launcher_log_is_filed_at_the_root_and_read_back_as_its_volume_channel(self):
        """Its file is under `logs/` like a call's, one level up, and is not
        taken for a call's on the way back in."""
        self.call_logs.put("launcher", [row("started", 1.0), row("launch a: requested", 2.0)])
        save_snapshot_to_volume(self.root, self.volume, load_snapshot_from_volume(self.root))
        self.call_logs.put("launcher", [row("started", 1.0), row("launch a: requested", 2.0), row("granted", 3.0)])
        save_snapshot_to_volume(self.root, self.volume, load_snapshot_from_volume(self.root))

        file = self.root / "logs" / "launcher.jsonl"
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        self.assertEqual([r["msg"] for r in rows], ["started", "launch a: requested", "granted"])
        self.assertNotIn("source", rows[0])
        self.assertEqual(self.call_logs["launcher:volume"], rows)
        self.assertEqual(load_snapshot_from_volume(self.root)["launcher"], 3)

        self.call_logs.clear()
        self.call_history.clear()
        self.assertEqual(load_snapshot_from_volume(self.root)["launcher"], 3)
        self.assertEqual(self.call_logs, {"launcher:volume": rows})
        self.assertEqual(self.call_history, {})

    def test_a_pass_is_stateless_across_containers(self):
        """A second pass with a cursor rebuilt from the files, as the scheduled
        function does each time, appends only what the first left out."""
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:container", [row("boot")])
        save_snapshot_to_volume(self.root, self.volume, load_snapshot_from_volume(self.root))
        self.call_logs.put("fc-a:container", [row("boot"), row("done")])
        save_snapshot_to_volume(self.root, self.volume, load_snapshot_from_volume(self.root))
        self.assertEqual([json.loads(line)["msg"] for line in self.file("fc-a").read_text().splitlines()], ["boot", "done"])


if __name__ == "__main__":
    unittest.main()
