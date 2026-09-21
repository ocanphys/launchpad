"""system.logs: the buffer that keeps one call's records, and the two passes
that move log rows between the Dicts and the volume."""

import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from system import logs
from system.logs import (
    BufferHandler,
    launcher_log,
    load_snapshot_from_volume,
    save_snapshot_to_volume,
    sync_volume_train,
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


class SnapshotTests(unittest.TestCase):
    ARTIFACT = "runs/toy/pretraining"

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.call_logs, self.call_history, self.train = Dict(), Dict(), Dict()
        self.volume = Volume()
        self.patches = [
            patch.object(logs, "call_logs", self.call_logs),
            patch.object(logs, "call_history", self.call_history),
            patch.object(logs, "train", self.train),
            patch.object(logs, "dirty", set()),
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
        persisted = {}
        save_snapshot_to_volume(self.root, self.volume, persisted, set())  # fc-a is dirty, so filed
        self.call_logs.put("fc-a:container", [row("boot"), row("step 1"), row("done")])
        save_snapshot_to_volume(self.root, self.volume, persisted, {"fc-a"})

        rows = [json.loads(line) for line in self.file("fc-a").read_text().splitlines()]
        self.assertEqual([(r["source"], r["msg"]) for r in rows], [("launcher", "granted"), ("container", "boot"), ("container", "step 1"), ("container", "done")])
        self.assertEqual(self.call_logs["fc-a:volume"], rows)
        self.assertEqual(persisted, {("fc-a", "launcher"): 1, ("fc-a", "container"): 3})
        self.assertEqual(json.loads((self.root / "call_history.json").read_text()), self.call_history)
        self.assertEqual(self.volume.commits, 2)
        self.assertEqual(logs.dirty, set())

    def test_a_call_not_in_beating_or_dirty_is_left_alone(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:container", [row("boot")])
        save_snapshot_to_volume(self.root, self.volume, {}, set())
        self.assertFalse(self.file("fc-a").exists())

    def test_a_channel_that_came_back_shorter_moves_nothing(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:container", [row("boot"), row("done")])
        persisted = {}
        save_snapshot_to_volume(self.root, self.volume, persisted, {"fc-a"})
        self.call_logs.put("fc-a:container", [])  # a wiped Dict, republished empty
        save_snapshot_to_volume(self.root, self.volume, persisted, {"fc-a"})
        self.assertEqual(len(self.file("fc-a").read_text().splitlines()), 2)
        self.assertEqual(persisted[("fc-a", "container")], 2)

    def test_startup_reads_the_files_back_and_seeds_the_cursor(self):
        self.grant("fc-a", 10.0)
        self.call_logs.put("fc-a:launcher", [row("granted", 10.0)])
        self.call_logs.put("fc-a:container", [row("boot", 11.0)])
        save_snapshot_to_volume(self.root, self.volume, {}, {"fc-a"})
        volume_rows = self.call_logs["fc-a:volume"]

        # A fresh leasebook with the Dicts wiped, and one file no grant names.
        self.call_logs.clear()
        self.call_history.clear()
        (self.root / "sources/tiny/logs").mkdir(parents=True)
        (self.root / "sources/tiny/logs/fc-old.jsonl").write_text(json.dumps(row("hi")) + "\n" + '{"ts": 2.0, "lev')
        persisted = load_snapshot_from_volume(self.root)

        self.assertEqual(persisted, {("fc-a", "launcher"): 1, ("fc-a", "container"): 1, ("fc-old", "launcher"): 0, ("fc-old", "container"): 0})
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
        self.assertEqual(load_snapshot_from_volume(self.root), {})
        self.assertEqual(self.call_history, {})

    def test_every_step_log_is_published_under_its_artifact(self):
        step = json.dumps({"step": 1, "attempt": 1, "loss": 2.0, "grad_norm": 1.0, "learning_rate": 0.5})
        (self.root / self.ARTIFACT).mkdir(parents=True)
        (self.root / self.ARTIFACT / "train.jsonl").write_text(step + "\n" + step[:10])
        self.assertEqual(sync_volume_train(self.root), 1)
        self.assertEqual(self.train, {f"{self.ARTIFACT}:volume": [json.loads(step)]})


if __name__ == "__main__":
    unittest.main()
