"""Resolver regressions using local manifests."""

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from artifacts.core.artifact import Artifact
from artifacts.core.resolve import Node, conflict_diff, declare, resolve


@dataclass(frozen=True)
class Example(Artifact):
    producer = "unused.Job"
    value: object

    @property
    def uid(self):
        return "example"

    @property
    def artifact_path(self):
        return Path(self.uid)

    @property
    def files(self):
        return {"first": "first.txt", "second": "second.txt"}


class ResolveTests(unittest.TestCase):
    def test_empty_containers_are_visible_in_conflicts(self):
        node = Node(
            Example([], commit="new"),
            (),
            "conflict",
            recorded=Example({}, commit="old"),
        )
        self.assertEqual(
            conflict_diff(node), ["parameters.value: on disk {}, requested []"]
        )

    def test_conflict_retains_commit_drift(self):
        with TemporaryDirectory() as directory:
            target = Path(directory)
            declare(resolve(Example("old", commit="old"), target=target))
            dag = resolve(Example("new", commit="new"), target=target)
            node = dag["example"]
            self.assertEqual(node.status, "conflict")
            self.assertTrue(node.drift)
            self.assertEqual(node.recorded_commit, "old")
            self.assertFalse(dag.ok)

    def test_conflicting_requests_at_same_path_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different artifacts requested"):
            resolve(Example("a", commit="same"), Example("b", commit="same"))
        self.assertEqual(
            len(resolve(Example("a", commit="same"), Example("a", commit="same"))), 1
        )

    def test_declaration_and_completion_statuses(self):
        with TemporaryDirectory() as directory:
            target = Path(directory)
            artifact = Example("a", commit="same")
            dag = resolve(artifact, target=target)
            self.assertEqual(dag["example"].status, "new")
            self.assertEqual(len(declare(dag)), 1)
            self.assertEqual(declare(dag), [])
            self.assertEqual(
                resolve(artifact, target=target)["example"].status, "declared"
            )
            (target / "example/first.txt").touch()
            self.assertEqual(
                resolve(artifact, target=target)["example"].status, "partial"
            )
            (target / "example/second.txt").touch()
            self.assertEqual(resolve(artifact, target=target)["example"].status, "done")
