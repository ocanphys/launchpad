"""PretrainJob checkpointing against a temporary root: failsafes, resuming a
crashed attempt, and one leg continuing from another's final model."""

import json
import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
from pydantic import ValidationError

from artifacts.core.artifact import Artifact
from artifacts.core.storage import write_manifest
from artifacts.dataset import DataSet
from artifacts.dataset.jobs import DataSetJob
from artifacts.sources import SourceURL
from artifacts.stages.pretraining import Pretraining
from artifacts.stages.pretraining.evaluations import BlockGradNorm, TokenLoss
from artifacts.stages.pretraining.jobs import PretrainJob
from artifacts.stages.pretraining.samplers import Epochs, Sweep
from artifacts.tokenized import TokenizedSource
from artifacts.tokenizers.bpe import Tokenizer
from models.mamba import Mamba
from models.transformer import Transformer
from models.transformerFA import TransformerFA
from system.lease_protocol import LeaseLost
from system.runtime import Worker
from training.leg import (
    Evaluation,
    LoopConfig,
    TrainingParameters,
)
from training.lr_schedule import WarmupCosine
from training.optimizers import AdamW
from training.steplog import StepLog

SOURCE = SourceURL(name="first", url="https://example.org/first.txt")
TOKENIZER = Tokenizer(vocab_size=64, special_tokens=("<pad>",), sources=(SOURCE,))
MODEL = Transformer(
    vocab_size=64, sequence_length=8, num_layers=1, d_model=16, d_ff=32,
    num_heads=2, rope_theta=10000,
)
TRAINING = TrainingParameters(
    total_steps=4, batch_size=4, max_norm=1,
    lr_schedule=WarmupCosine(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=1, cosine_cycle_iters=100),
    optimizer=AdamW(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8),
)
LOOP = LoopConfig(evaluations=(TokenLoss(eval_every=2, batch_size=4),), gpu_check_every=2, checkpoint_every=2)


class Taken(Evaluation):
    """Measures val_taken, 1 at every step it is taken."""

    def measure(self, job, model, dataset, worker):
        return {"val_taken": torch.tensor(1.0)}


def rebuilt(artifact: Artifact, **changes) -> Artifact:
    """Returns `artifact` constructed again with `changes`, so its validators run and its
    cached identity is recomputed, as model_copy would not."""
    return type(artifact)(**{name: getattr(artifact, name) for name in type(artifact).model_fields} | changes)


def worker() -> Worker:
    """A real `Worker` with the lease no-opped: a job writes every file
    through `worker.publishing`, so the stand-in has to be one that publishes.
    Its log and confirm are Mocks, for tests that read either back."""
    return Worker(artifact_path="", call_id="local", log=Mock(), confirm_lease=Mock(), progress={})


def logged(worker: Worker) -> list[str]:
    return [call.args[0] for call in worker.log.info.call_args_list]


def declare(artifact: Artifact, root: Path) -> None:
    """Write one manifest, the way declaration will -- what bind reads back."""
    write_manifest(root, artifact.artifact_path, artifact.to_manifest())


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
        run_id="r", dataset=dataset, training_parameters=TRAINING,
        loop_config=LOOP, model=MODEL,
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

    def test_a_tied_leg_resumes_still_tied_and_matches_an_uninterrupted_run(self):
        tied = rebuilt(self.leg, model=MODEL.model_copy(update={"tie_embeddings": True}))
        self.assertNotEqual(tied.lineage_hash, self.leg.lineage_hash)
        with crash_after_first_failsafe(), self.assertRaisesRegex(RuntimeError, "died"):
            PretrainJob(tied).run(self.root, worker())
        resumed = worker()
        PretrainJob(tied).run(self.root, resumed)
        self.assertIn("found failsafe 2.pt, starting from step 2", logged(resumed))
        weights = self.final(tied)["model"]
        self.assertTrue(torch.equal(weights["embedding.weight"], weights["linear.W"]))

        with TemporaryDirectory() as other:
            other_root = Path(other)
            straight = rebuilt(leg(build_dataset(other_root)), model=tied.model)
            PretrainJob(straight).run(other_root, worker())
            for key, tensor in torch.load(straight.paths(other_root)["model"])["model"].items():
                self.assertTrue(torch.equal(tensor, weights[key]), key)

    def test_a_compiled_leg_is_the_same_leg_and_resumes_onto_the_same_weights(self):
        compiled = rebuilt(self.leg, loop_config=LOOP.model_copy(update={"compile": "max-autotune"}))
        self.assertEqual(compiled.artifact_path, self.leg.artifact_path)  # loop_config is not identity
        modes = []
        original = torch.nn.Module.compile

        def compile(module, mode):
            modes.append(mode)
            original(module, backend="eager")  # dynamo traces and wraps in place, without building kernels

        with patch.object(torch.nn.Module, "compile", compile):
            with crash_after_first_failsafe(), self.assertRaisesRegex(RuntimeError, "died"):
                PretrainJob(compiled).run(self.root, worker())
            PretrainJob(compiled).run(self.root, worker())
        self.assertEqual(modes, ["max-autotune", "max-autotune"])  # each attempt compiles what it resumed

        # the checkpoint keys are the module's own, and the weights an eager run's
        with TemporaryDirectory() as other:
            other_root = Path(other)
            eager = leg(build_dataset(other_root))
            PretrainJob(eager).run(other_root, worker())
            expected = torch.load(eager.paths(other_root)["model"])["model"]
            weights = self.final(compiled)["model"]
            self.assertEqual(weights.keys(), expected.keys())
            for key, tensor in expected.items():
                self.assertTrue(torch.equal(tensor, weights[key]), key)

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
        rate_of_step_4 = TRAINING.lr_schedule.rate(4)
        self.assertNotEqual(rate_of_step_4, 1e-3)  # torch's default, the rate the optimizer is built with
        for state in (torch.load(self.folder / "checkpoints" / "4.pt"), self.final(self.leg)):
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], rate_of_step_4)

    def test_the_step_log_keeps_every_attempts_rows(self):
        with crash_after_first_failsafe(), self.assertRaisesRegex(RuntimeError, "died"):
            PretrainJob(self.leg).run(self.root, worker())
        PretrainJob(self.leg).run(self.root, worker())
        rows = [json.loads(line) for line in (self.folder / "train.jsonl").read_text().splitlines()]
        self.assertEqual([(r["attempt"], r["step"]) for r in rows], [(1, 1), (1, 2), (2, 3), (2, 4)])
        self.assertTrue(all(isinstance(r["loss"], float) for r in rows))
        # TokenLoss runs every 2 steps; the rows in between carry no val_loss
        for r in rows:
            validated = {"val_loss", "val_loss_var"} if r["step"] % 2 == 0 else set()
            self.assertEqual(set(r), {"step", "attempt", "loss", "grad_norm", "learning_rate"} | validated)

    def test_a_crash_between_validations_leaves_no_gap_and_the_last_step_validates(self):
        sparse = rebuilt(self.leg, loop_config=LOOP.model_copy(update={"evaluations": (TokenLoss(eval_every=3, batch_size=4),)}))
        folder = self.root / sparse.artifact_path
        with crash_after_first_failsafe(), self.assertRaisesRegex(RuntimeError, "died"):
            PretrainJob(sparse).run(self.root, worker())
        PretrainJob(sparse).run(self.root, worker())
        rows = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        # the failsafe at step 2 flushed steps 1-2 before the crash
        self.assertEqual([(r["attempt"], r["step"]) for r in rows], [(1, 1), (1, 2), (2, 3), (2, 4)])
        # step 4 is not a multiple of eval_every, but it is end_step
        self.assertEqual([r["step"] for r in rows if "val_loss" in r], [3, 4])

    def test_each_evaluation_is_taken_on_its_own_period_and_all_at_the_last_step(self):
        both = (TokenLoss(eval_every=2, batch_size=4), Taken(eval_every=3))
        leg = rebuilt(self.leg, loop_config=LOOP.model_copy(update={"evaluations": both}))
        PretrainJob(leg).run(self.root, worker())
        rows = [json.loads(line) for line in (self.root / leg.artifact_path / "train.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows if "val_loss" in r], [2, 4])
        self.assertEqual([r["step"] for r in rows if "val_taken" in r], [3, 4])

    def test_a_leg_with_no_evaluations_records_only_training_metrics(self):
        leg = rebuilt(self.leg, loop_config=LOOP.model_copy(update={"evaluations": ()}))
        PretrainJob(leg).run(self.root, worker())
        rows = [json.loads(line) for line in (self.root / leg.artifact_path / "train.jsonl").read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [1, 2, 3, 4])
        self.assertFalse(any(name.startswith("val_") for r in rows for name in r))

    def test_the_step_log_writes_a_diverged_value_as_null(self):
        steplog = StepLog(self.root, self.leg.artifact_path.as_posix())
        self.folder.mkdir(parents=True, exist_ok=True)
        steplog.record(1, {"loss": torch.tensor(float("nan")), "grad_norm": torch.tensor(float("inf")), "learning_rate": 0.5})
        steplog.flush()
        # strict JSON, which is what the dashboard's response has to be
        row = json.loads((self.folder / "train.jsonl").read_text(), parse_constant=lambda name: self.fail(name))
        self.assertEqual(row, {"step": 1, "attempt": 1, "loss": None, "grad_norm": None, "learning_rate": 0.5})

    def test_an_epoch_lasts_as_many_steps_as_epochs_reads_before_reshuffling(self):
        steps = Epochs().steps_per_epoch(self.leg, self.root)
        self.assertEqual(steps, 62)  # 2000 tokens: 249 sequences of 8, 62 batches of 4
        train = self.dataset.bind(self.root).train
        one_epoch = Epochs(max_epochs=1)
        starts = one_epoch.starts(train, 4, 8, seed=0, end_step=steps)  # the sampler's own epoch holds them all
        first = np.concatenate([starts(step) for step in range(1, steps + 1)])
        self.assertEqual(len(set(first.tolist())), 4 * steps)  # no sequence twice within the epoch
        with self.assertRaises(ValueError):
            one_epoch.starts(train, 4, 8, seed=0, end_step=steps + 1)  # and not one step more

    def test_a_model_vocab_that_disagrees_with_its_tokenizer_is_refused(self):
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            Pretraining(
                run_id="r", dataset=self.dataset, training_parameters=TRAINING,
                loop_config=LOOP, model=Transformer.model_validate({**MODEL.model_dump(), "vocab_size": 65}),
            )

    def test_a_manifest_names_the_model_class_beside_its_fields(self):
        second = leg(self.dataset, starting_checkpoint=self.leg)
        model = second.to_manifest()["parameters"]["model"]
        self.assertEqual(model, {"class": "models.transformer.Transformer", **MODEL.model_dump()})
        rebuilt = Artifact.from_manifest(second.to_manifest())
        self.assertEqual(rebuilt, second)
        self.assertIsInstance(rebuilt.model, Transformer)
        self.assertIsInstance(rebuilt.starting_checkpoint.model, Transformer)

    def test_a_manifest_names_the_sampler_class_and_identity_follows_it(self):
        self.assertEqual(
            self.leg.to_manifest()["parameters"]["sampler"],
            {"class": "artifacts.stages.pretraining.samplers.RandomWindows"},
        )
        epochs = rebuilt(self.leg, sampler=Epochs(max_epochs=3))
        self.assertEqual(
            epochs.to_manifest()["parameters"]["sampler"],
            {"class": "artifacts.stages.pretraining.samplers.Epochs", "max_epochs": 3},
        )
        self.assertNotEqual(epochs.lineage_hash, self.leg.lineage_hash)
        self.assertEqual(Artifact.from_manifest(epochs.to_manifest()).sampler, Epochs(max_epochs=3))

    def test_a_leg_its_model_cannot_run_is_refused_at_declaration(self):
        flash = TransformerFA(**MODEL.model_dump())
        with self.assertRaisesRegex(ValidationError, "flash attention takes fp16 or bf16, not torch.float32"):
            rebuilt(self.leg, model=flash)
        self.assertEqual(rebuilt(self.leg, model=flash, autocast="torch.bfloat16").precision, "torch.bfloat16")

    def test_an_autocast_leg_is_its_own_trajectory_and_trains_in_bf16(self):
        self.assertIsNone(self.leg.to_manifest()["parameters"]["autocast"])
        mixed = rebuilt(self.leg, autocast="torch.bfloat16")
        self.assertEqual(mixed.to_manifest()["parameters"]["autocast"], "torch.bfloat16")
        self.assertNotEqual(mixed.lineage_hash, self.leg.lineage_hash)
        with self.assertRaisesRegex(ValidationError, "autocast"):
            rebuilt(self.leg, autocast="torch.float16")

        # every forward, training and evaluation alike, runs under bf16 autocast; without it, under none
        for leg, expected in ((mixed, torch.bfloat16), (self.leg, None)):
            seen = set()

            def batch_loss(job, model, inputs, targets):
                seen.add(torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else None)
                return original(job, model, inputs, targets)

            original = PretrainJob.batch_loss
            with patch.object(PretrainJob, "batch_loss", batch_loss):
                PretrainJob(leg).run(self.root, worker())
            self.assertEqual(seen, {expected})
            self.assertEqual(self.final(leg)["step"], 4)

    def test_a_sweep_leg_trains_on_the_split_in_order_and_one_past_it_refuses(self):
        # 2000 tokens hold 249 sequences of 8: 62 steps of 4, each the next 4 in order
        swept = rebuilt(self.leg, sampler=Sweep())
        self.assertEqual(swept.to_manifest()["parameters"]["sampler"], {"class": "artifacts.stages.pretraining.samplers.Sweep"})
        firsts = []

        def batch_loss(job, model, inputs, targets):
            if model.training:
                firsts.append(inputs[:, 0].tolist())
            return original(job, model, inputs, targets)

        original = PretrainJob.batch_loss
        with patch.object(PretrainJob, "batch_loss", batch_loss):
            PretrainJob(swept).run(self.root, worker())
        train = self.dataset.bind(self.root).train.tokens
        self.assertEqual(firsts, [[int(train[s * 8]) for s in range(step * 4, step * 4 + 4)] for step in range(4)])
        long = rebuilt(swept, training_parameters=TRAINING.model_copy(update={"total_steps": 63}))
        with self.assertRaisesRegex(ValueError, "62 batches of 4, fewer than end_step 63"):
            PretrainJob(long).run(self.root, worker())

    def test_an_epochs_leg_trains_and_one_past_its_epochs_refuses_before_its_first_step(self):
        # 2000 tokens hold 249 sequences of 8: 62 steps an epoch in batches of 4
        epochs = rebuilt(self.leg, sampler=Epochs(max_epochs=1))
        PretrainJob(epochs).run(self.root, worker())
        self.assertEqual(self.final(epochs)["step"], 4)
        long = rebuilt(epochs, training_parameters=TRAINING.model_copy(update={"total_steps": 63}))
        with self.assertRaisesRegex(ValueError, "end_step 63 is past 1 epochs of 62 steps"):
            PretrainJob(long).run(self.root, worker())
        self.assertEqual(list((self.root / long.artifact_path / "checkpoints").iterdir()), [])

    def test_a_leg_trains_any_model_and_continues_only_its_own(self):
        inputs = {"run_id": "r", "dataset": self.dataset, "training_parameters": TRAINING, "loop_config": LOOP}
        mamba = Mamba(vocab_size=MODEL.vocab_size, sequence_length=8, num_layers=1, d_model=16)
        leg = Pretraining(model=mamba, **inputs)
        self.assertEqual(leg.to_manifest()["parameters"]["model"]["class"], "models.mamba.Mamba")
        self.assertEqual(Artifact.from_manifest(leg.to_manifest()).model, mamba)
        with self.assertRaisesRegex(ValueError, "model must match starting_checkpoint.model"):
            Pretraining(model=mamba, starting_checkpoint=self.leg, **inputs)

    def test_a_leg_states_its_model_or_inherits_it_from_its_starting_checkpoint(self):
        inputs = {"run_id": "r", "dataset": self.dataset, "training_parameters": TRAINING, "loop_config": LOOP}
        self.assertEqual(Pretraining(starting_checkpoint=self.leg, **inputs).model, self.leg.model)
        with self.assertRaisesRegex(ValidationError, r"model\n  Field required"):
            Pretraining(**inputs)

    def test_validation_scores_a_split_shorter_than_one_batch(self):
        # 2000 tokens hold 249 sequences of 8, all in one batch of up to 300
        wide = (TokenLoss(eval_every=2, batch_size=300),)
        metrics = PretrainJob(self.leg).evaluate(MODEL.build(torch.device("cpu")), self.dataset.bind(self.root), worker(), wide)
        self.assertTrue(torch.isfinite(metrics["val_loss"]))
        self.assertEqual(metrics["val_loss_var"].item(), 0.0)  # one batch: no spread, and never NaN

    def test_two_evaluations_measuring_one_metric_are_refused(self):
        twice = (TokenLoss(eval_every=2, batch_size=4), TokenLoss(eval_every=2, batch_size=8))
        with self.assertRaisesRegex(ValueError, r"TokenLoss\(eval_every=2, batch_size=8\) measures \['val_loss', 'val_loss_var'\]"):
            PretrainJob(self.leg).evaluate(MODEL.build(torch.device("cpu")), self.dataset.bind(self.root), worker(), twice)

    def test_block_grad_norm_reads_each_blocks_clipped_gradient(self):
        both = (TokenLoss(eval_every=2, batch_size=4), BlockGradNorm(eval_every=2))
        leg = rebuilt(self.leg, loop_config=LOOP.model_copy(update={"evaluations": both}))
        PretrainJob(leg).run(self.root, worker())
        rows = [json.loads(line) for line in (self.root / leg.artifact_path / "train.jsonl").read_text().splitlines()]
        names = {f"grad_norm_tblocks_{i}" for i in range(leg.model.num_layers)}
        for r in rows:
            self.assertEqual(names & set(r), names if r["step"] % 2 == 0 else set())
            if r["step"] % 2 == 0:
                # the blocks hold some of the parameters grad_norm is taken over, clipped to max_norm
                blocks = sum(r[name] ** 2 for name in names) ** 0.5
                self.assertTrue(0 < blocks <= min(r["grad_norm"], TRAINING.max_norm) * (1 + 1e-5))

    def test_live_progress_is_the_step_alone_off_cuda(self):
        running = worker()
        PretrainJob(self.leg).run(self.root, running)
        self.assertEqual(set(running.progress), {"step"})

    def test_a_leg_keeps_its_starting_checkpoints_tokenizer(self):
        other = Tokenizer(vocab_size=64, special_tokens=("<eos>",), sources=(SOURCE,))
        with self.assertRaisesRegex(ValueError, "starting_checkpoint"):
            Pretraining(
                run_id="r", dataset=DataSet.from_sources(other, (SOURCE,), (SOURCE,)),
                training_parameters=TRAINING, loop_config=LOOP, starting_checkpoint=self.leg,
            )

    def test_a_dataset_speaks_one_tokenizer_and_a_leg_reads_it_there(self):
        other = Tokenizer(vocab_size=64, special_tokens=("<eos>",), sources=(SOURCE,))
        with self.assertRaisesRegex(ValueError, "mixes tokenizers"):
            DataSet(
                train_set=(TokenizedSource(tokenizer=TOKENIZER, source=SOURCE),),
                valid_set=(TokenizedSource(tokenizer=other, source=SOURCE),),
            )
        self.assertEqual(self.leg.tokenizer, TOKENIZER)

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
