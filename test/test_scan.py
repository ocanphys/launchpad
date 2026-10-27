"""The volume scan against a temporary root: a clean root reports nothing, and
each kind of manifest that would not survive a rewrite is named by folder."""

import json
import shutil

import pytest
from test_golden import CASES, ODYSSEY

from artifacts.core.manifest import manifest_json
from artifacts.core.storage import MANIFEST
from migrations.scan import scan


@pytest.fixture
def root(tmp_path):
    for artifact in CASES.values():
        folder = tmp_path / artifact.artifact_path
        folder.mkdir(parents=True)
        (folder / MANIFEST).write_text(manifest_json(artifact.to_manifest()))
    return tmp_path


def test_a_root_the_code_wrote_reports_nothing(root):
    assert scan(root) == (len(CASES), [])


def test_unreadable_json_is_named(root):
    (root / "sources/odyssey" / MANIFEST).write_text("{not json")
    total, problems = scan(root)
    assert total == len(CASES)
    assert [line.split(":")[0] for line in problems] == ["sources/odyssey"]


def test_a_manifest_in_the_wrong_folder_is_named(root):
    shutil.copytree(root / "sources/odyssey", root / "sources/moved")
    _, problems = scan(root)
    assert len(problems) == 1 and problems[0].startswith("sources/moved:")


def test_a_manifest_written_in_other_bytes_names_its_first_differing_line(root):
    path = root / "sources/odyssey" / MANIFEST
    path.write_text(path.read_text().replace('"name": "odyssey"', '"name":  "odyssey"'))
    [problem] = scan(root)[1]
    assert problem == "sources/odyssey: line 10: '    \"name\":  \"odyssey\",' -> '    \"name\": \"odyssey\",'"


def test_a_missing_trailing_newline_is_a_difference(root):
    path = root / "sources/odyssey" / MANIFEST
    path.write_text(json.dumps(ODYSSEY.to_manifest()))
    assert scan(root)[1][0].startswith("sources/odyssey: line 1:")
