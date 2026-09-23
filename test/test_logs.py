"""system.logs: the shape every row carries, the channel its stamp files it
under, and the pass that moves those rows between the Dicts and the volume."""

import json
import logging
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from system import logs
from system.logs import (
    AMBIENT,
    LAUNCHER,
    WORKER,
    load_snapshot_from_volume,
    persist_snapshot,
)


class Dict(dict):
    put = dict.__setitem__


class Volume:
    commits = 0

    def commit(self) -> None:
        self.commits += 1


def row(msg: str, ts: float = 1.0, call_id: str = "fc-a", source: str = WORKER) -> dict:
    return {"ts": ts, "call_id": call_id, "source": source, "level": "INFO", "logger": "job", "msg": msg}


class LoggingTests(unittest.TestCase):
    """`start_logging`: every record filed under the call and source its
    stamp names, an unstamped one under the container's own call id."""

    def setUp(self):
        self.store = Dict()
        self.enterContext(patch.object(logs, "call_logs", self.store))
        self.enterContext(patch.object(logs, "HEARTBEAT_SECONDS", 3600))  # only the stop publishes
        self.addCleanup(logging.getLogger().handlers.clear)

    def start(self, call_id: str):
        stop = logs.start_logging(call_id)
        self.addCleanup(stop)  # a second stop after the test's own is a no-op join
        return stop

    def test_a_call_and_the_container_around_it_share_one_shape(self):
        stop = self.start("fc-a")
        logs.call_logger("job", WORKER, "fc-a").info("started %d", 3)
        logging.getLogger("a_library").warning("a library's own")
        stop()

        [entry] = self.store["fc-a:livedict:worker"]
        self.assertEqual(set(entry), {"ts", "call_id", "source", "level", "logger", "msg"})
        self.assertEqual(
            {key: entry[key] for key in ("call_id", "source", "level", "logger", "msg")},
            {"call_id": "fc-a", "source": WORKER, "level": "INFO", "logger": "job", "msg": "started 3"},
        )
        [noise] = self.store["fc-a:livedict:ambient"]
        self.assertEqual((noise["call_id"], noise["source"], noise["msg"]), ("fc-a", AMBIENT, "a library's own"))

    def test_a_traceback_is_folded_into_the_row(self):
        stop = self.start("fc-a")
        try:
            raise ValueError("boom")
        except ValueError:
            logs.call_logger("job", WORKER, "fc-a").exception("failed")
        stop()

        [entry] = self.store["fc-a:livedict:worker"]
        self.assertEqual(entry["msg"].splitlines()[0], "failed")
        self.assertIn("ValueError: boom", entry["msg"])

    def test_a_thread_files_under_the_call_that_stamped_the_record(self):
        """The id rides on the record, so no thread has to inherit anything."""
        stop = self.start("fc-a")
        logger = logs.call_logger("job", WORKER, "fc-a")
        thread = threading.Thread(target=lambda: logger.info("from a thread"))
        thread.start()
        thread.join()
        stop()

        self.assertEqual([r["msg"] for r in self.store["fc-a:livedict:worker"]], ["from a thread"])

    def test_a_row_about_a_call_is_the_launchers_own_as_well(self):
        """The one way in: the call id the logger was asked for is what files
        the row, never a second function next to the first. A row about a call
        is filed twice, so the call's page has it and the launcher's log stays
        the whole account of what the container did."""
        stop = self.start(LAUNCHER)
        logs.launcher_logger().info("container started")
        logs.launcher_logger("fc-b").info("granted")
        logging.getLogger("uvicorn").warning("noise")
        stop()

        self.assertEqual(
            [(r["call_id"], r["msg"]) for r in self.store["launcher:livedict:launcher"]],
            [("launcher", "container started"), ("fc-b", "granted")],
        )
        self.assertEqual([r["msg"] for r in self.store["fc-b:livedict:launcher"]], ["granted"])
        self.assertEqual([r["msg"] for r in self.store["launcher:livedict:ambient"]], ["noise"])

    def test_a_worker_files_its_rows_once(self):
        """A worker stamps no call but its own, so the two channels the
        launcher would write are one here."""
        stop = self.start("fc-a")
        logs.call_logger("job", WORKER, "fc-a").info("counting")
        stop()

        self.assertEqual(list(self.store), ["fc-a:livedict:worker"])

    def test_a_new_container_continues_the_longer_of_the_live_channel_and_the_file(self):
        for live, filed in (([row("a"), row("b")], [row("a")]), ([row("c")], [row("a"), row("b")])):
            with self.subTest(live=live):
                self.store.update({"launcher:livedict:launcher": live, "launcher:volume:launcher": filed})
                stop = self.start(LAUNCHER)
                logs.launcher_logger().info("started")
                stop()
                self.assertEqual(
                    [r["msg"] for r in self.store["launcher:livedict:launcher"]],
                    ["a", "b", "started"],
                )

    def test_a_publish_that_fails_is_a_warning_the_next_one_carries(self):
        failures = [RuntimeError("Dict unavailable")]

        def put(key, value):
            if failures:
                raise failures.pop()
            self.store[key] = value

        with patch.object(logs, "HEARTBEAT_SECONDS", 0.01), patch.object(self.store, "put", put):
            stop = self.start("fc-a")
            logs.call_logger("job", WORKER, "fc-a").info("still working")
            deadline = time.time() + 2
            while not self.store.get("fc-a:livedict:ambient") and time.time() < deadline:
                time.sleep(0.01)
            stop()

        self.assertEqual([r["msg"] for r in self.store["fc-a:livedict:worker"]], ["still working"])
        self.assertEqual(
            [r["msg"] for r in self.store["fc-a:livedict:ambient"]],
            ["fc-a:livedict:worker not published (Dict unavailable)"],
        )


class SnapshotTests(unittest.TestCase):
    ARTIFACT = "runs/toy/pretraining"

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.call_logs, self.call_history = Dict(), Dict()
        self.volume = Volume()
        self.enterContext(patch.object(logs, "call_logs", self.call_logs))
        self.enterContext(patch.object(logs, "call_history", self.call_history))
        self.addCleanup(self.directory.cleanup)

    def file(self, call_id: str) -> Path:
        return self.root / self.ARTIFACT / "logs" / f"{call_id}.jsonl"

    def rows(self, file: Path) -> list[dict]:
        return [json.loads(line) for line in file.read_text().splitlines()]

    def grant(self, call_id: str, ts: float) -> None:
        grants = self.call_history.get(self.ARTIFACT) or []
        self.call_history.put(self.ARTIFACT, [*grants, {"call_id": call_id, "granted_ts": ts, "artifact_type": "Pretraining"}])

    def test_a_pass_appends_only_what_is_new_and_the_volume_channel_follows(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:livedict:launcher", [row("granted", source=LAUNCHER)])
        self.call_logs.put("fc-a:livedict:worker", [row("boot"), row("step 1")])
        persist_snapshot(self.root, self.volume)
        self.call_logs.put("fc-a:livedict:worker", [row("boot"), row("step 1"), row("done")])
        persist_snapshot(self.root, self.volume)

        rows = self.rows(self.file("fc-a"))
        self.assertEqual(
            [(r["source"], r["msg"]) for r in rows],
            [(WORKER, "boot"), (WORKER, "step 1"), (LAUNCHER, "granted"), (WORKER, "done")],
        )
        self.assertEqual(self.call_logs["fc-a:volume:worker"], [r for r in rows if r["source"] == WORKER])
        self.assertEqual(self.call_logs["fc-a:volume:launcher"], [row("granted", source=LAUNCHER)])
        self.assertEqual(json.loads((self.root / "call_history.json").read_text()), self.call_history)
        self.assertEqual(self.volume.commits, 2)

    def test_a_call_with_nothing_new_gets_no_file(self):
        self.grant("fc-a", 10.0)
        persist_snapshot(self.root, self.volume)
        self.assertFalse(self.file("fc-a").exists())

    def test_a_channel_that_came_back_shorter_moves_nothing(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:livedict:worker", [row("boot"), row("done")])
        persist_snapshot(self.root, self.volume)
        self.call_logs.put("fc-a:livedict:worker", [])  # a wiped Dict, republished empty
        persist_snapshot(self.root, self.volume)
        self.assertEqual([r["msg"] for r in self.rows(self.file("fc-a"))], ["boot", "done"])

    def test_startup_reads_the_files_back_and_lists_what_no_grant_names(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:livedict:launcher", [row("granted", 10.0, source=LAUNCHER)])
        self.call_logs.put("fc-a:livedict:worker", [row("boot", 11.0)])
        persist_snapshot(self.root, self.volume)

        # A fresh leasebook with the Dicts wiped, and one file no grant names.
        self.call_logs.clear()
        self.call_history.clear()
        (self.root / "sources/tiny/logs").mkdir(parents=True)
        (self.root / "sources/tiny/logs/fc-old.jsonl").write_text(
            json.dumps(row("hi", call_id="fc-old")) + "\n" + '{"ts": 2.0, "lev'
        )
        filed = load_snapshot_from_volume(self.root)

        self.assertEqual(filed, {
            "fc-a:volume:worker": [row("boot", 11.0)],
            "fc-a:volume:launcher": [row("granted", 10.0, source=LAUNCHER)],
            "fc-old:volume:worker": [row("hi", call_id="fc-old")],
        })
        self.assertEqual(self.call_logs, filed)
        self.assertEqual(self.call_history[self.ARTIFACT], [{"call_id": "fc-a", "granted_ts": 10.0, "artifact_type": "Pretraining"}])
        self.assertEqual(self.call_history["sources/tiny"], [{"call_id": "fc-old", "granted_ts": None, "artifact_type": None}])

    def test_startup_merges_the_file_with_a_dict_that_is_ahead(self):
        (self.root / "call_history.json").write_text(json.dumps({self.ARTIFACT: [{"call_id": "fc-a", "granted_ts": 10.0, "artifact_type": "Pretraining"}]}))
        self.grant("fc-b", 20.0)  # granted after the last persist pass
        load_snapshot_from_volume(self.root)
        self.assertEqual([g["call_id"] for g in self.call_history[self.ARTIFACT]], ["fc-a", "fc-b"])

    def test_an_empty_volume_is_empty(self):
        self.assertEqual(load_snapshot_from_volume(self.root), {})
        self.assertEqual(self.call_logs, {})
        self.assertEqual(self.call_history, {})

    def test_the_launcher_log_is_filed_at_the_root_and_read_back_as_its_own_call(self):
        """Its file is under `logs/` like a call's, one level up, and is not
        taken for a call's on the way back in."""
        launcher_row = lambda msg, ts: row(msg, ts, call_id=LAUNCHER, source=LAUNCHER)
        self.call_logs.put("launcher:livedict:launcher", [launcher_row("started", 1.0), launcher_row("granted", 2.0)])
        persist_snapshot(self.root, self.volume)
        self.call_logs.put("launcher:livedict:launcher", [launcher_row("started", 1.0), launcher_row("granted", 2.0), launcher_row("stopped", 3.0)])
        persist_snapshot(self.root, self.volume)

        file = self.root / "logs" / "launcher.jsonl"
        rows = self.rows(file)
        self.assertEqual([r["msg"] for r in rows], ["started", "granted", "stopped"])
        self.assertEqual(self.call_logs["launcher:volume:launcher"], rows)

        self.call_logs.clear()
        self.call_history.clear()
        self.assertEqual(load_snapshot_from_volume(self.root), {"launcher:volume:launcher": rows})
        self.assertEqual(self.call_history, {})

    def test_a_pass_is_stateless_across_containers(self):
        """Each pass rebuilds the count it appends from out of the files, so a
        second pass in a container that never saw the first appends only what
        the first left out."""
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:livedict:worker", [row("boot")])
        persist_snapshot(self.root, self.volume)
        self.call_logs.put("fc-a:livedict:worker", [row("boot"), row("done")])
        persist_snapshot(self.root, self.volume)
        self.assertEqual([r["msg"] for r in self.rows(self.file("fc-a"))], ["boot", "done"])


if __name__ == "__main__":
    unittest.main()
