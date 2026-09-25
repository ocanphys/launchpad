"""PretrainJob checkpointing against a temporary root: failsafes, resuming a
crashed attempt, and one leg continuing from another's final model."""

import json
import unittest
from array import array
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch

from artifacts.core.artifact import MANIFEST, Artifact
from artifacts.core.SGD.lr_schedule import lr_cosine_schedule
from artifacts.core.SGD.training import (
    LoopConfig,
    LRSchedule,
    OptimizerParameters,
    TrainingParameters,
)
from artifacts.dataset import DataSet
from artifacts.dataset.jobs import DataSetJob
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.stages.pretraining.jobs import PretrainJob
from artifacts.tokenizers.bpe import Tokenizer
from models.transformer import ModelParameters
from system.lease_protocol import LeaseLost
from system.runtime import Worker

SOURCE = SourceURL(name="first", url="https://example.org/first.txt")
TOKENIZER = Tokenizer(vocab_size=64, special_tokens=("<pad>",), sources=(SOURCE,))
MODEL = ModelParameters(
    vocab_size=64, sequence_length=8, num_layers=1, d_model=16, d_ff=32,
    num_heads=2, rope_theta=10000, device="cpu",
)
TRAINING = TrainingParameters(
    total_steps=4, batch_size=4, max_norm=1,
    lr_schedule=LRSchedule(1e-3, 1e-4, 1, 100),
    optimizer="torch.optim.AdamW",
    optimizer_parameters=OptimizerParameters(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
)
LOOP = LoopConfig(val_every=2, gpu_check_every=0, checkpoint_every=2)


def worker() -> Worker:
    """A real `Worker` with the lease no-opped: a job writes every file
    through `worker.publishing`, so the stand-in has to be one that publishes.
    Its log and confirm are Mocks, for tests that read either back."""
    return Worker(artifact_path="", call_id="local", log=Mock(), confirm_lease=Mock(), progress={})


def logged(worker: Worker) -> list[str]:
    return [call.args[0] for call in worker.log.info.call_args_list]


def declare(artifact: Artifact, root: Path) -> None:
    """Write one manifest, the way declaration will -- what bind reads back."""
    path = root / artifact.artifact_path / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_manifest()))


def build_dataset(root: Path) -> DataSet:
    """A built, declared DataSet of 2000 random tokens below MODEL.vocab_size,
    same split for train and valid."""
    dataset = DataSet.from_sources(TOKENIZER, (SOURCE,), (SOURCE,))
    declare(dataset, root)
    tokens = dataset.train_set[0].paths(root)["tokens"]
    tokens.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    tokens.write_bytes(array("H", rng.integers(0, 64, 2000).tolist()).tobytes())
    DataSetJob(dataset).run(root, worker())
    return dataset


def leg(dataset: DataSet, starting_checkpoint: Pretraining | None = None) -> Pretraining:
    return Pretraining(
        run_id="r", dataset=dataset, tokenizer=TOKENIZER, training_parameters=TRAINING,
        loop_config=LOOP, model="models.transformer", model_parameters=MODEL,
        starting_checkpoint=starting_checkpoint,
    )


def crash_after_first_failsafe():
    """A patch on PretrainJob.failsafe that writes the failsafe and then dies,
    the way a container would after its last durable write."""
    original = PretrainJob.failsafe

    def failsafe(self, *args):
        original(self, *args)
        raise RuntimeError("container died")

    return patch.object(PretrainJob, "failsafe", failsafe)


class PretrainJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.dataset = build_dataset(self.root)
        self.leg = leg(self.dataset)
        self.folder = self.root / self.leg.artifact_path

    def tearDown(self):
        self.directory.cleanup()

    def final(self, leg: Pretraining) -> dict:
        return torch.load(leg.paths(self.root)["model"])

    def test_a_full_run_records_the_step_of_every_checkpoint(self):
        PretrainJob(self.leg).run(self.root, worker())
        self.assertEqual(self.final(self.leg)["step"], 4)
        for name in ("2.pt", "4.pt"):
            self.assertEqual(torch.load(self.folder / "checkpoints" / name)["step"], int(name[0]))

    def test_a_lease_is_read_once_per_file_published_and_never_per_step(self):
        """The fence is at the rename, not in the loop. A four-step leg costs
        four lease reads because four renames happen in it, not because four
        steps do -- the two in between are the failsafes and the strip that
        drops the optimizer out of the earlier one."""
        running = worker()
        PretrainJob(self.leg).run(self.root, running)
        self.assertEqual(
            [call.args[0] for call in running.confirm_lease.call_args_list],
            [
                "before publishing 2.pt",
                "before publishing 4.pt",
                "before publishing 2.pt",
                "before publishing model.pt",
            ],
        )

    def test_a_leg_that_loses_its_lease_publishes_no_model(self):
        """Every file a training job publishes is renamed into place under a
        confirmed lease, the last of them `model.pt`. A leg that trains to its
        end and loses the lease on that last confirm leaves the failsafes it
        already published and no finished artifact."""
        def stop_at_the_end(label: str) -> None:
            if label == "before publishing model.pt":
                raise LeaseLost("this call was cancelled")

        stopped = worker()
        stopped.confirm_lease.side_effect = stop_at_the_end
        with self.assertRaises(LeaseLost):
            PretrainJob(self.leg).run(self.root, stopped)
        self.assertFalse(self.leg.paths(self.root)["model"].exists())
        self.assertEqual(sorted(p.name for p in (self.folder / "checkpoints").glob("*.pt")), ["2.pt", "4.pt"])

        # and the next attempt resumes off the last of them and finishes
        resumed = worker()
        PretrainJob(self.leg).run(self.root, resumed)
        self.assertIn("found failsafe 4.pt, starting from step 4", logged(resumed))
        self.assertEqual(self.final(self.leg)["step"], 4)

    def test_a_cancelled_leg_stops_at_its_next_checkpoint(self):
        """The confirm is inside `worker.publishing`, so a failsafe is a fence
        too: a call that lost the lease mid-run stops there rather than
        training on into a folder someone else now owns."""
        stopped = worker()
        stopped.confirm_lease.side_effect = LeaseLost("this call was cancelled")
        with self.assertRaises(LeaseLost):
            PretrainJob(self.leg).run(self.root, stopped)
        self.assertEqual(list((self.folder / "checkpoints").glob("*.pt")), [])

    def test_only_the_latest_failsafe_keeps_its_optimizer(self):
        PretrainJob(self.leg).run(self.root, worker())
        self.assertNotIn("optimizer", torch.load(self.folder / "checkpoints" / "2.pt"))
        self.assertIn("optimizer", torch.load(self.folder / "checkpoints" / "4.pt"))
        self.assertIn("optimizer", self.final(self.leg))

    def test_a_crashed_attempt_resumes_from_its_failsafe_and_matches_an_uninterrupted_run(self):
        with crash_after_first_failsafe(), self.assertRaisesRegex(RuntimeError, "died"):
            PretrainJob(self.leg).run(self.root, worker())
        self.assertFalse(self.leg.paths(self.root)["model"].exists())
        self.assertEqual(sorted(p.name for p in (self.folder / "checkpoints").iterdir()), ["2.pt"])

        resumed = worker()
        PretrainJob(self.leg).run(self.root, resumed)
        self.assertIn("found failsafe 2.pt, starting from step 2", logged(resumed))
        self.assertEqual(self.final(self.leg)["step"], 4)

        # the same leg run straight through lands on identical weights: the
        # batches are a function of (seed, step), and the optimizer state
        # round-tripped exactly
        with TemporaryDirectory() as other:
            other_root = Path(other)
            straight = leg(build_dataset(other_root))
            PretrainJob(straight).run(other_root, worker())
            for key, tensor in torch.load(straight.paths(other_root)["model"])["model"].items():
                self.assertTrue(torch.equal(tensor, self.final(self.leg)["model"][key]), key)

    def test_durable_progress_is_the_furthest_checkpoint(self):
        progress = lambda: self.leg.durable_progress(self.root)
        self.assertEqual(progress(), {"phase": "step", "done": 0, "total": 4})
        with crash_after_first_failsafe(), self.assertRaises(RuntimeError):
            PretrainJob(self.leg).run(self.root, worker())
        self.assertEqual(progress(), {"phase": "step", "done": 2, "total": 4})
        PretrainJob(self.leg).run(self.root, worker())
        for name in ("2.pt", "4.pt"):
            (self.folder / "checkpoints" / name).unlink()
        self.assertEqual(progress(), {"phase": "step", "done": 4, "total": 4})

    def test_rerunning_a_finished_leg_leaves_its_model_untouched(self):
        PretrainJob(self.leg).run(self.root, worker())
        path = self.leg.paths(self.root)["model"]
        before = path.read_bytes()
        # even with the failsafes gone, nothing is retrained or rewritten
        for name in ("2.pt", "4.pt"):
            (self.folder / "checkpoints" / name).unlink()
        rerun = worker()
        PretrainJob(self.leg).run(self.root, rerun)
        self.assertEqual(logged(rerun), ["model.pt exists, leg already complete at step 4"])
        self.assertEqual(path.read_bytes(), before)

    def test_a_crash_while_writing_the_final_model_is_recovered_bit_for_bit(self):
        original = torch.save

        def dies_on_the_final_model(state, out, *args, **kwargs):
            if Path(out.name).name.startswith("model.pt."):  # the file `publishing` opened for it
                raise RuntimeError("container died")
            original(state, out, *args, **kwargs)

        # the last failsafe sits on end_step, so the second attempt trains
        # zero steps and model.pt must still come out as an uninterrupted run's
        with patch.object(torch, "save", dies_on_the_final_model), self.assertRaisesRegex(RuntimeError, "died"):
            PretrainJob(self.leg).run(self.root, worker())
        model = self.leg.paths(self.root)["model"]
        self.assertFalse(model.exists())
        self.assertEqual(list(self.folder.glob("*.tmp")), [], "the failed write left its tmp behind")
        resumed = worker()
        PretrainJob(self.leg).run(self.root, resumed)
        self.assertIn("found failsafe 4.pt, starting from step 4", logged(resumed))
        with TemporaryDirectory() as other:
            other_root = Path(other)
            straight = leg(build_dataset(other_root))
            PretrainJob(straight).run(other_root, worker())
            expected, actual = torch.load(straight.paths(other_root)["model"]), self.final(self.leg)
            for key, tensor in expected["model"].items():
                self.assertTrue(torch.equal(tensor, actual["model"][key]), key)
            self.assertEqual(expected["optimizer"]["param_groups"], actual["optimizer"]["param_groups"])

    def test_a_crash_after_the_last_failsafe_only_writes_the_final_model(self):
        PretrainJob(self.leg).run(self.root, worker())
        self.leg.paths(self.root)["model"].unlink()
        resumed = worker()
        PretrainJob(self.leg).run(self.root, resumed)
        self.assertIn("found failsafe 4.pt, starting from step 4", logged(resumed))
        self.assertEqual(self.final(self.leg)["step"], 4)
        self.assertNotIn("step 6/4", " ".join(logged(resumed)))

    def test_a_failsafe_without_an_optimizer_is_not_resumed_from(self):
        PretrainJob(self.leg).run(self.root, worker())
        (self.folder / "checkpoints" / "4.pt").unlink()
        self.leg.paths(self.root)["model"].unlink()
        resumed = worker()
        PretrainJob(self.leg).run(self.root, resumed)
        self.assertIn("no checkpoint, initializing model, starting from step 0", logged(resumed))

    def test_a_leg_continues_from_the_previous_legs_final_model(self):
        PretrainJob(self.leg).run(self.root, worker())
        second = leg(self.dataset, starting_checkpoint=self.leg)
        self.assertEqual((second.start_step, second.end_step), (4, 8))
        continued = worker()
        PretrainJob(second).run(self.root, continued)
        self.assertIn(f"loaded {self.leg.uid}, starting from step 4", logged(continued))
        self.assertEqual(self.final(second)["step"], 8)
        self.assertEqual(
            sorted(p.name for p in (self.root / second.artifact_path / "checkpoints").iterdir()),
            ["6.pt", "8.pt"],
        )
        # the first leg's folder is untouched
        self.assertEqual(sorted(p.name for p in (self.folder / "checkpoints").iterdir()), ["2.pt", "4.pt"])

    def test_a_starting_checkpoint_at_the_wrong_step_is_refused(self):
        PretrainJob(self.leg).run(self.root, worker())
        path = self.leg.paths(self.root)["model"]
        state = torch.load(path)
        state["step"] = 3
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "at step 3"):
            PretrainJob(leg(self.dataset, starting_checkpoint=self.leg)).run(self.root, worker())

    def test_a_checkpoint_holds_the_rate_its_step_was_taken_with(self):
        PretrainJob(self.leg).run(self.root, worker())
        rate_of_step_4 = lr_cosine_schedule(4, **asdict(TRAINING.lr_schedule))
        self.assertNotEqual(rate_of_step_4, TRAINING.optimizer_parameters.lr)
        for state in (torch.load(self.folder / "checkpoints" / "4.pt"), self.final(self.leg)):
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], rate_of_step_4)

    def test_the_step_log_keeps_every_attempts_rows(self):
        with crash_after_first_failsafe(), self.assertRaisesRegex(RuntimeError, "died"):
            PretrainJob(self.leg).run(self.root, worker())
        PretrainJob(self.leg).run(self.root, worker())
        rows = [json.loads(line) for line in (self.folder / "train.jsonl").read_text().splitlines()]
        self.assertEqual([(r["attempt"], r["step"]) for r in rows], [(1, 1), (1, 2), (2, 3), (2, 4)])
        self.assertEqual(set(rows[0]), {"step", "attempt", "loss", "val_loss", "grad_norm", "learning_rate"})
        self.assertTrue(all(isinstance(r["loss"], float) for r in rows))
        # validation runs every val_every steps; the rows in between carry no val_loss
        self.assertEqual(
            [r["val_loss"] is None for r in rows],
            [(r["step"] % LOOP.val_every) != 0 for r in rows],
        )

    def test_a_model_vocab_that_disagrees_with_its_tokenizer_is_refused(self):
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            Pretraining(
                run_id="r", dataset=self.dataset, tokenizer=TOKENIZER, training_parameters=TRAINING,
                loop_config=LOOP, model="models.transformer", model_parameters=replace(MODEL, vocab_size=65),
            )

    def test_a_manifest_rebuilds_model_parameters_as_the_model_packages_dataclass(self):
        second = leg(self.dataset, starting_checkpoint=self.leg)
        rebuilt = Artifact.from_manifest(second.to_manifest())
        self.assertEqual(rebuilt, second)
        self.assertEqual(rebuilt.model, "models.transformer")
        self.assertIsInstance(rebuilt.model_parameters, ModelParameters)
        self.assertIsInstance(rebuilt.starting_checkpoint.model_parameters, ModelParameters)

    def test_a_leg_keeps_its_starting_checkpoints_tokenizer(self):
        other = Tokenizer(vocab_size=64, special_tokens=("<eos>",), sources=(SOURCE,))
        with self.assertRaisesRegex(ValueError, "starting_checkpoint"):
            Pretraining(
                run_id="r", dataset=DataSet.from_sources(other, (SOURCE,), (SOURCE,)), tokenizer=other,
                training_parameters=TRAINING, loop_config=LOOP, starting_checkpoint=self.leg,
            )
        with self.assertRaisesRegex(ValueError, "not tokenized with"):
            leg(DataSet.from_sources(other, (SOURCE,), (SOURCE,)))

    def test_a_starting_checkpoint_without_an_optimizer_gets_a_fresh_one(self):
        PretrainJob(self.leg).run(self.root, worker())
        path = self.leg.paths(self.root)["model"]
        state = torch.load(path)
        del state["optimizer"]
        torch.save(state, path)
        continued = worker()
        PretrainJob(leg(self.dataset, starting_checkpoint=self.leg)).run(self.root, continued)
        continued.log.warning.assert_called_once()
        self.assertIn("no optimizer", continued.log.warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
