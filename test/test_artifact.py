"""The definition contract: normalization, manifests, load, status, bind.

Every case here runs against a TemporaryDirectory passed as `root`, which is
the same mechanism the volume is reached by -- an absolute path, and nothing
that distinguishes one absolute path from another (spec.md section 4).
"""

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

from artifacts.core.artifact import MANIFEST, Artifact, Resources
from artifacts.core.manifest import manifest_json
from artifacts.sources import Source, SourceURL
from artifacts.tokenizers.bpe import Tokenizer


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


def declare(artifact: Artifact, root: Path) -> None:
    """Write one manifest, the way declaration will (spec.md section 6)."""
    path = root / artifact.artifact_path / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_json(artifact.to_manifest()))


def build(artifact: Artifact, root: Path) -> None:
    """Put every one of an artifact's own files in place."""
    for path in artifact.paths(root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


ODYSSEY = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
ILIAD = SourceURL(name="iliad", url="https://example.org/iliad.txt")


def tokenizer(sources) -> Tokenizer:
    return Tokenizer(vocab_size=1000, special_tokens=("<pad>",), sources=sources)


class DefinitionTests(unittest.TestCase):
    def test_reversed_sources_are_one_definition(self):
        forwards, backwards = tokenizer((ODYSSEY, ILIAD)), tokenizer((ILIAD, ODYSSEY))
        self.assertEqual(forwards, backwards)
        self.assertEqual(hash(forwards), hash(backwards))
        self.assertEqual(forwards.artifact_path, backwards.artifact_path)
        self.assertEqual(forwards.to_manifest(), backwards.to_manifest())
        self.assertEqual(
            [d.artifact_path for d in forwards.deps()],
            [d.artifact_path for d in backwards.deps()],
        )

    def test_a_set_holds_one_of_each_source(self):
        self.assertEqual(tokenizer((ODYSSEY, ODYSSEY)), tokenizer((ODYSSEY,)))

    def test_two_definitions_at_one_path_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "more than one definition"):
            tokenizer((ODYSSEY, SourceURL(name="odyssey", url="elsewhere")))

    def test_an_empty_set_is_still_a_dependency(self):
        manifest = tokenizer(()).to_manifest()
        self.assertEqual(manifest["dependencies"], {"sources": []})
        self.assertEqual(sorted(manifest["parameters"]), ["special_tokens", "vocab_size"])


class ManifestTests(unittest.TestCase):
    def test_round_trip_keeps_the_definition_and_what_was_recorded(self):
        original = Tokenizer(
            vocab_size=1000,
            special_tokens=("<pad>",),
            sources=(ODYSSEY, ILIAD),
            commit="old",
            allocated_resources=Resources(gpu_type="A100"),
        )
        rebuilt = Artifact.from_manifest(original.to_manifest())
        self.assertEqual(rebuilt, original)
        self.assertEqual(rebuilt.commit, "old")
        self.assertEqual(rebuilt.allocated_resources, Resources(gpu_type="A100"))
        self.assertFalse(rebuilt.bound)

    def test_a_shared_memo_decodes_a_shared_subtree_once(self):
        memo = {}
        first = Artifact.from_manifest(tokenizer((ODYSSEY, ILIAD)).to_manifest(), memo)
        second = Artifact.from_manifest(tokenizer((ODYSSEY,)).to_manifest(), memo)
        [odyssey] = second.sources
        self.assertIs(odyssey, next(s for s in first.sources if s == ODYSSEY))
        self.assertEqual(len(memo), 4)  # two tokenizers, two sources
        # Without one, every call decodes its own tree.
        self.assertIsNot(Artifact.from_manifest(ODYSSEY.to_manifest()), Artifact.from_manifest(ODYSSEY.to_manifest()))

    def test_bytes_are_canonical(self):
        text = manifest_json(ODYSSEY.to_manifest())
        self.assertTrue(text.endswith("}\n"))
        self.assertIn('\n  "commit": ', text)  # two-space indent
        self.assertEqual(
            list(json.loads(text)),
            ["artifact", "commit", "allocated_resources", "parameters", "dependencies"],
        )
        # Resources declares cpu, gpu_type, gpu_count; the file sorts them, so
        # reordering the dataclass cannot rewrite every manifest that has one.
        self.assertEqual(
            list(json.loads(text)["allocated_resources"]),
            ["cpu", "gpu_count", "gpu_type"],
        )

    def test_non_finite_numbers_are_refused(self):
        with self.assertRaises(ValueError):
            manifest_json({"parameters": {"lr": float("nan")}})


class LoadTests(unittest.TestCase):
    def test_load_comes_back_unbound_even_when_built(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tok = tokenizer((ODYSSEY,))
            declare(tok, root)
            tok.paths(root)["tokenizer"].write_text(
                json.dumps(
                    {
                        "vocab": {},
                        "merges": [],
                        "special_tokens": ["<pad>"],
                        "vocab_size": 1000,
                    }
                )
            )
            loaded = Artifact.load(tok.artifact_path, root)
            self.assertEqual(loaded, tok)
            self.assertFalse(loaded.bound)
            self.assertTrue(loaded.bind(root).bound)

    def test_a_manifest_in_the_wrong_folder_is_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declare(ODYSSEY, root)
            moved = root / "sources" / "elsewhere"
            moved.mkdir()
            (moved / MANIFEST).write_text(manifest_json(ODYSSEY.to_manifest()))
            with self.assertRaisesRegex(ValueError, "describes sources/odyssey"):
                Artifact.load("sources/elsewhere", root)

    def test_a_memo_hands_back_the_same_instance_for_the_same_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declare(ODYSSEY, root)
            memo = {}
            first = Artifact.load("sources/odyssey", root, memo)
            self.assertIs(Artifact.load("sources/odyssey", root, memo), first)
            self.assertIsNot(Artifact.load("sources/odyssey", root), first)
            # The same manifest at the wrong folder is still refused.
            moved = root / "sources" / "elsewhere"
            moved.mkdir()
            (moved / MANIFEST).write_text(manifest_json(ODYSSEY.to_manifest()))
            with self.assertRaisesRegex(ValueError, "describes sources/odyssey"):
                Artifact.load("sources/elsewhere", root, memo)

    def test_a_path_cannot_escape_its_root(self):
        with self.assertRaisesRegex(ValueError, "must stay under root"):
            Artifact.load("../secrets", Path("/storage"))

    def test_a_relative_root_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be an absolute path"):
            Artifact.load("sources/odyssey", Path("storage"))

    def test_a_string_root_is_refused(self):
        with self.assertRaisesRegex(TypeError, "must be a Path"):
            Artifact.load("sources/odyssey", "/storage")
        with self.assertRaisesRegex(TypeError, "must be a Path"):
            ODYSSEY.status("/storage")

    def test_nothing_declared_says_so(self):
        with TemporaryDirectory() as directory:
            missing = self.assertRaisesRegex(FileNotFoundError, "nothing declared")
            with missing:
                Artifact.load("sources/odyssey", Path(directory))


class StatusTests(unittest.TestCase):
    def test_the_footprint_follows_the_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            footprint = ODYSSEY.status(root)
            self.assertFalse(footprint.manifest)
            self.assertEqual(footprint.outputs, {"raw text": False})

            declare(ODYSSEY, root)
            self.assertTrue(ODYSSEY.status(root).manifest)
            self.assertEqual(ODYSSEY.durable_progress(root), {"phase": "files", "done": 0, "total": 1})

            build(ODYSSEY, root)
            self.assertEqual(ODYSSEY.status(root).outputs, {"raw text": True})
            self.assertEqual(ODYSSEY.durable_progress(root), {"phase": "files", "done": 1, "total": 1})

    def test_a_virtual_artifact_owns_nothing_and_borrows_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            virtual = Virtual(sources=(ODYSSEY,))
            build(ODYSSEY, root)
            footprint = virtual.status(root)
            # complete, yet undeclared and owning nothing: the two questions
            # are separate, which is what keeps this readable as `new`.
            self.assertEqual(footprint.outputs, {})
            self.assertEqual(list(footprint.completion.values()), [True])
            self.assertFalse(footprint.manifest)

    def test_an_artifact_nothing_produces_has_no_job(self):
        with self.assertRaisesRegex(ValueError, "declares no producer"):
            Virtual(sources=()).job()


class BindTests(unittest.TestCase):
    def test_bind_returns_the_stored_declaration_and_leaves_the_original(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stored = SourceURL(name="odyssey", url=ODYSSEY.url, commit="old")
            declare(stored, root)
            build(stored, root)

            bound = ODYSSEY.bind(root)
            self.assertEqual(bound, ODYSSEY)
            self.assertIsNot(bound, ODYSSEY)
            self.assertEqual(bound.commit, "old")  # the manifest is the authority
            self.assertNotEqual(ODYSSEY.commit, "old")

    def test_bind_refuses_a_definition_the_path_does_not_hold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declare(ODYSSEY, root)
            build(ODYSSEY, root)
            with self.assertRaisesRegex(ValueError, "different definition"):
                SourceURL(name="odyssey", url="elsewhere").bind(root)

    def test_bind_refuses_an_undeclared_or_unbuilt_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "nothing declared"):
                ODYSSEY.bind(root)
            declare(ODYSSEY, root)
            with self.assertRaisesRegex(FileNotFoundError, "not built"):
                ODYSSEY.bind(root)

    def test_nothing_to_complete_binds_as_soon_as_it_is_declared(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            virtual = Virtual(sources=())
            declare(virtual, root)
            self.assertEqual(virtual.bind(root), virtual)
            self.assertIsNone(virtual.durable_progress(root))


if __name__ == "__main__":
    unittest.main()
