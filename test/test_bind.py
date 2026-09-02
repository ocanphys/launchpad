"""Does bind() ever mutate the object it was called on? For every artifact
family in the repo: no. `bind` always returns a distinct, equal copy, and the
recipe you called it on stays exactly as unbound as it was -- see
Artifact.bind's docstring (dag/artifact.py) for the contract this checks.

Plain functions named test_*, plain assert -- no test framework required, so
this runs with `python test/test_bind.py` today and is already pytest-shaped
if pytest is ever added. `run_all()` at the bottom drives every test_* and
prints a pass/fail summary.
"""

import logging
import sys
import tempfile
from array import array
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dag.artifact import Artifact
from dag.resolve import Declaration, producer_for, status
from datasets.artifact import DataSet
from mappeddatasets.artifact import MappedDataSet
from models.mock.artifact import ModelParameters, Pretraining, PretrainingConfig
from sources.artifact import Source
from tokenizers import template  # noqa: F401 -- registers TemplateTokenizer
from tokenizers.bpe import TokenizedSource, Tokenizer
from tokenizers.template import TemplateTokenizer

logging.basicConfig(level=logging.WARNING, format="%(message)s")
WORKER = SimpleNamespace(log=logging.getLogger("test"))

CORPUS = (
    "But soft, what light through yonder window breaks? It is the east, and "
    "Juliet is the sun. Arise, fair sun, and kill the envious moon, who is "
    "already sick and pale with grief that thou her maid art far more fair "
    "than she.<|endoftext|>"
) * 20


def _root() -> Path:
    return Path(tempfile.mkdtemp(prefix="test_bind_"))


def _write_source(source: Source, root: Path, text: str = CORPUS) -> None:
    source.paths(root)["raw text"].write_text(text)


def build(artifact, root: Path) -> None:
    """Declare, then run every job the artifact's plan still needs. Safe to
    call again after writing more source files -- already-done jobs (a
    source with no text on disk yet still counts as its own job, not "done")
    are skipped."""
    from dag.resolve import job_list, resolve

    Declaration(artifact, root).write()
    for job in job_list(resolve(artifact)):
        if status(job.artifact, root) == "done":
            continue
        job.run(root, WORKER)


# -- default _load: Source, TokenizedSource, DataSet, Pretraining ----------
#
# None of these override _load, so bind()'s only job for them is the file
# check plus a copy -- there's no private state to attach. The point of
# testing them anyway is that bind still must never mutate the original.


def test_bind_never_mutates_source():
    root = _root()
    source = Source(name="a", url="https://example.invalid/a.txt")
    Declaration(source, root).write()

    try:
        source.bind(root)
        raise AssertionError("expected FileNotFoundError before the file exists")
    except FileNotFoundError:
        pass

    _write_source(source, root)
    bound = source.bind(root)

    assert bound is not source, "bind must not return the same object"
    assert bound == source, "bind must return an equal artifact"
    assert hash(bound) == hash(source)
    assert bound.paths(root) == source.paths(root)

    bound_again = source.bind(root)
    assert bound_again is not bound, "two binds must give two distinct copies"
    assert bound_again == bound


def test_bind_never_mutates_tokenized_source():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a,)
    )
    tokenized = TokenizedSource(tokenizer=tokenizer, source=a)
    Declaration(tokenized, root).write()  # declare only -- no network call yet
    _write_source(a, root)  # stands in for SourceJob's actual download
    build(tokenized, root)  # sources/a already "done", so only the rest runs

    assert status(tokenized, root) == "done"
    bound = tokenized.bind(root)
    assert bound is not tokenized
    assert bound == tokenized
    assert bound.paths(root)["tokens"].exists()


def test_bind_never_mutates_dataset():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    b = Source(name="b", url="https://example.invalid/b.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a, b)
    )
    dataset = DataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a, b], valid_sources=[a]
    )
    Declaration(dataset, root).write()
    _write_source(a, root)
    _write_source(b, root, CORPUS[::-1].replace("\n", " "))
    build(dataset, root)

    assert status(dataset, root) == "done"
    bound = dataset.bind(root)
    assert bound is not dataset
    assert bound == dataset
    assert bound.paths(root)["training set"].stat().st_size > 0


def test_dataset_is_shared_not_run_scoped():
    """DataSet has no run_id: identity is a digest over train_set/valid_set,
    same pattern as MappedDataSet/Tokenizer, so a second request for the
    same tokenizer and sources lands on the same folder and reuses it --
    no run has to rebuild train.bin/valid.bin its own copy already has."""
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    b = Source(name="b", url="https://example.invalid/b.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a, b)
    )
    dataset = DataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a, b], valid_sources=[a]
    )
    assert not hasattr(dataset, "run_id")

    Declaration(dataset, root).write()
    _write_source(a, root)
    _write_source(b, root, CORPUS[::-1].replace("\n", " "))
    build(dataset, root)
    assert status(dataset, root) == "done"

    same_again = DataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a, b], valid_sources=[a]
    )
    assert same_again.uid == dataset.uid
    assert same_again.artifact_path == dataset.artifact_path
    assert status(same_again, root) == "done", "a second request reuses the same folder"

    # order is identifying, same as MappedDataSet: a different concatenation
    # order is a different train.bin, so it must not share a folder
    reordered = DataSet.from_sources(
        tokenizer=tokenizer, train_sources=[b, a], valid_sources=[a]
    )
    assert reordered.uid != dataset.uid


def test_bind_never_mutates_pretraining():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    b = Source(name="b", url="https://example.invalid/b.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a, b)
    )
    dataset = DataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a], valid_sources=[b]
    )
    pretraining = Pretraining(
        run_id="t2",
        dataset=dataset,
        tokenizer=tokenizer,
        model_parameters=ModelParameters(),
        config=PretrainingConfig(total_steps=5, batch_size=4, checkpoint_every=5),
    )
    Declaration(pretraining, root).write()
    _write_source(a, root)
    _write_source(b, root, CORPUS[::-1].replace("\n", " "))
    build(pretraining, root)

    assert status(pretraining, root) == "done"
    bound = pretraining.bind(root)
    assert bound is not pretraining
    assert bound == pretraining


def test_pretraining_manifest_round_trips_with_either_dataset_kind():
    """Pretraining.dataset accepts DataSet | MappedDataSet. A field typed as
    a union of several Artifact subclasses isn't actually ambiguous to
    decode -- every manifest already names its own concrete type -- but
    dag.artifact._decode used to treat any union with more than one arm as
    unreadable. This locks in the fix: manifest() -> from_manifest() must
    reproduce the right concrete type on both sides of the union."""
    a = Source(name="a", url="https://example.invalid/a.txt")
    tokenizer = Tokenizer(vocab_size=280, special_tokens=(), sources=(a,))
    for dataset in (
        DataSet.from_sources(tokenizer=tokenizer, train_sources=[a], valid_sources=[a]),
        MappedDataSet.from_sources(
            tokenizer=tokenizer, train_sources=[a], valid_sources=[a]
        ),
    ):
        pretraining = Pretraining(
            run_id="t3",
            dataset=dataset,
            tokenizer=tokenizer,
            model_parameters=ModelParameters(),
            config=PretrainingConfig(total_steps=1, batch_size=1),
        )
        rebuilt = Artifact.from_manifest(pretraining.manifest())
        assert rebuilt == pretraining
        assert type(rebuilt.dataset) is type(dataset)


# -- custom _load: Tokenizer, TemplateTokenizer, MappedDataSet -------------
#
# Here bind attaches real, private state (vocab/merges, a codec, a
# TokenStream). The thing to prove is stronger: the ORIGINAL must stay
# unusable (still raises) after bind() runs, and only the returned copy
# gains a working interface.


def test_bind_never_mutates_tokenizer():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a,)
    )
    tokenized = TokenizedSource(tokenizer=tokenizer, source=a)
    Declaration(tokenized, root).write()
    _write_source(a, root)
    build(tokenized, root)

    assert not tokenizer.bound
    try:
        tokenizer.encode("x")
        raise AssertionError("unbound tokenizer must refuse to encode")
    except RuntimeError:
        pass

    bound = tokenizer.bind(root)

    assert bound is not tokenizer, "bind must not return the same object"
    assert bound == tokenizer, "bind must return an equal artifact"
    assert bound.bound, "the returned copy must be usable"
    assert not tokenizer.bound, "the original must still be unbound"

    ids = bound.encode("what light<|endoftext|>through")
    assert bound.decode(ids) == "what light<|endoftext|>through"

    # the original is still refusing, proving _load never touched it
    try:
        tokenizer.encode("x")
        raise AssertionError("original tokenizer must still refuse to encode")
    except RuntimeError:
        pass
    try:
        _ = tokenizer.vocab
        raise AssertionError("original tokenizer must still have no vocab")
    except RuntimeError:
        pass

    # binding twice gives two independent, equally-usable copies
    bound2 = tokenizer.bind(root)
    assert bound2 is not bound
    assert bound2.decode(bound2.encode("the sun")) == "the sun"


def test_bind_never_mutates_template_tokenizer():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    t = TemplateTokenizer(vocab_size=8, special_tokens=("<|endoftext|>",), sources=(a,))
    Declaration(t, root).write()
    _write_source(a, root)
    producer_for(t).run(root, WORKER)

    assert not t.bound
    try:
        t.encode("the")
        raise AssertionError("unbound TemplateTokenizer must refuse to encode")
    except RuntimeError:
        pass

    bound = t.bind(root)
    assert bound is not t
    assert bound == t
    assert bound.bound
    assert not t.bound
    assert bound.decode(bound.encode("the and")) == "the and"
    assert bound.decode(bound.encode("zebra")) == "<unk>"

    try:
        t.encode("the")
        raise AssertionError("original TemplateTokenizer must still refuse to encode")
    except RuntimeError:
        pass


def test_bind_never_mutates_mapped_dataset():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    b = Source(name="b", url="https://example.invalid/b.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a, b)
    )
    mapped = MappedDataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a, b], valid_sources=[a]
    )
    Declaration(mapped, root).write()
    _write_source(a, root)
    _write_source(b, root, CORPUS[::-1].replace("\n", " "))
    build(mapped, root)

    assert status(mapped, root) == "done"
    # its own folder holds nothing but the manifest -- MappedDataSetJob's
    # run() never had anything to write, and completion follows the sources
    manifest_folder = root / mapped.artifact_path
    assert [p.name for p in manifest_folder.iterdir()] == ["manifest.json"]
    assert not mapped.bound
    try:
        _ = mapped.train_tokens
        raise AssertionError("unbound MappedDataSet must refuse to read train_tokens")
    except RuntimeError:
        pass

    bound = mapped.bind(root)
    assert bound is not mapped, "bind must not return the same object"
    assert bound == mapped, "bind must return an equal artifact"
    assert bound.bound
    assert not mapped.bound, "the original must still be unbound"

    total = len(bound.train_tokens)
    assert total > 0
    a_tokens = array("H")
    a_tokens.frombytes(
        (root / tokenizer.artifact_path / "bin" / "a" / "tokens.bin").read_bytes()
    )
    assert list(bound.train_tokens[: len(a_tokens)]) == list(a_tokens)

    try:
        _ = mapped.train_tokens
        raise AssertionError(
            "original MappedDataSet must still refuse to read train_tokens"
        )
    except RuntimeError:
        pass


def test_mapped_dataset_reads_new_not_undeclared_over_reused_sources():
    """A fresh MappedDataSet built over sources that are already tokenized
    -- the exact reuse case it exists for -- must read `new`, not
    `undeclared`. `undeclared` means "something wrote outputs nobody asked
    for", which pre-declaration has to be judged by this artifact's own
    (always empty) folder, never by its dependencies' completion_paths --
    otherwise every reuse would look like an anomaly."""
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a,)
    )
    tokenized = TokenizedSource(tokenizer=tokenizer, source=a)
    Declaration(tokenized, root).write()
    _write_source(a, root)
    build(tokenized, root)
    assert status(tokenized, root) == "done"

    # never declared, and its dependency is already fully done
    mapped = MappedDataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a], valid_sources=[a]
    )
    assert status(mapped, root) == "new"

    Declaration(mapped, root).write()
    assert status(mapped, root) == "done", "completion follows the sources immediately"


def test_mapped_dataset_partial_when_only_some_sources_are_done():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    b = Source(name="b", url="https://example.invalid/b.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a, b)
    )
    mapped = MappedDataSet.from_sources(
        tokenizer=tokenizer, train_sources=[a, b], valid_sources=[a]
    )
    Declaration(mapped, root).write()
    assert status(mapped, root) == "declared"

    # training the tokenizer needs both sources' raw text (tokenizer.sources
    # is (a, b)), so both must exist even though only a's TokenizedSource is
    # in this build's plan -- b's own tokens.bin still doesn't get produced
    _write_source(a, root)
    _write_source(b, root, CORPUS[::-1].replace("\n", " "))
    build(TokenizedSource(tokenizer, a), root)
    assert status(mapped, root) == "partial", (
        "a is done, b's tokens.bin still isn't there"
    )

    _write_source(b, root, CORPUS[::-1].replace("\n", " "))
    build(mapped, root)
    assert status(mapped, root) == "done"


# -- Artifact.at: the other way to get a bound artifact --------------------


def test_artifact_at_returns_bound_copy():
    root = _root()
    a = Source(name="a", url="https://example.invalid/a.txt")
    tokenizer = Tokenizer(
        vocab_size=280, special_tokens=("<|endoftext|>",), sources=(a,)
    )
    tokenized = TokenizedSource(tokenizer=tokenizer, source=a)
    Declaration(tokenized, root).write()
    _write_source(a, root)
    build(tokenized, root)

    at_path = root / tokenizer.artifact_path
    from_at = Artifact.at(at_path)

    assert from_at is not tokenizer, (
        "at() must not return the constructor's object either"
    )
    assert from_at == tokenizer
    assert from_at.bound, "at() must come back bound when the files exist"
    assert from_at.decode(from_at.encode("the sun")) == "the sun"

    # a freshly declared, never-built recipe comes back plain, not bound
    b = Source(name="b", url="https://example.invalid/b.txt")
    unbuilt = Tokenizer(vocab_size=280, special_tokens=(), sources=(b,))
    Declaration(TokenizedSource(unbuilt, b), root).write()
    from_at_unbuilt = Artifact.at(root / unbuilt.artifact_path)
    assert not from_at_unbuilt.bound


def run_all() -> None:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001 -- a test-runner report, not app code
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
