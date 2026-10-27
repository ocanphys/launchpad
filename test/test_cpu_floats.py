"""The cpu-floats migration against a temporary root holding the golden
manifests: it plans exactly the files an int cpu makes differ, restores their
bytes, and refuses a root with any other difference."""

import pytest
from test_golden import CASES

from artifacts.core.manifest import manifest_json
from artifacts.core.storage import MANIFEST
from migrations.cpu_floats import plan
from migrations.scan import scan


@pytest.fixture
def root(tmp_path):
    for artifact in CASES.values():
        folder = tmp_path / artifact.artifact_path
        folder.mkdir(parents=True)
        (folder / MANIFEST).write_text(manifest_json(artifact.to_manifest()))
    return tmp_path


def as_old_code_wrote(root):
    """Rewrites every float cpu under `root` as an int; returns the original text per path."""
    originals = {}
    for path in root.rglob(MANIFEST):
        text = path.read_text()
        if '"cpu": 4.0' in text:
            originals[path] = text
            path.write_text(text.replace('"cpu": 4.0', '"cpu": 4'))
    return originals


def test_a_root_the_code_wrote_plans_nothing(root):
    assert plan(root) == {}


def test_int_cpus_are_planned_back_to_todays_bytes(root):
    originals = as_old_code_wrote(root)
    assert len(originals) > 1  # the tokenizer, and everything that embeds it
    assert plan(root) == originals
    for path, text in plan(root).items():
        path.write_text(text)
    assert scan(root) == (len(CASES), [])


def test_any_other_difference_refuses_the_whole_plan(root):
    as_old_code_wrote(root)
    source = root / "sources/odyssey" / MANIFEST
    source.write_text(source.read_text().replace('"name": "odyssey"', '"name":  "odyssey"'))
    with pytest.raises(ValueError, match=r"sources/odyssey: differs in more than an int cpu"):
        plan(root)


def test_a_missing_final_newline_is_another_difference(root):
    source = root / "sources/odyssey" / MANIFEST
    source.write_text(source.read_text().rstrip("\n"))
    with pytest.raises(ValueError, match=r"sources/odyssey"):
        plan(root)
