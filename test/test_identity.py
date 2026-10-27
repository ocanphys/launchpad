"""The identity migration against a temporary root holding the old-format
golden manifests at their old paths: it moves exactly the legs, and leaves
every manifest as the new goldens are, refuses anything it cannot place, and
plans nothing on a second run."""

import json
from pathlib import Path

import pytest
from test_golden import CASES, GOLDEN

from artifacts.core.storage import MANIFEST
from config import CALL_HISTORY
from migrations.identity import apply, moves, plan
from migrations.scan import scan

OLD = Path(__file__).parent / "fixtures" / "old_format"
OLD_PATHS = json.loads((OLD / "paths.json").read_text())
NEW_PATHS = json.loads((GOLDEN / "paths.json").read_text())
FIRST_LEG = OLD_PATHS["pretraining_first_leg"]


@pytest.fixture
def root(tmp_path):
    for name, path in OLD_PATHS.items():
        (tmp_path / path).mkdir(parents=True)
        (tmp_path / path / MANIFEST).write_text((OLD / f"{name}.json").read_text())
    return tmp_path


def test_only_the_legs_move_and_every_manifest_is_planned(root):
    planned = plan(root)
    assert len(planned) == len(CASES)
    assert moves(planned) == {OLD_PATHS[n]: NEW_PATHS[n] for n in CASES if n.startswith("pretraining")}


def test_applied_the_root_holds_the_new_goldens_and_plans_nothing(root):
    apply(root, plan(root))
    for name in CASES:
        assert (root / NEW_PATHS[name] / MANIFEST).read_text() == (GOLDEN / f"{name}.json").read_text()
    assert scan(root) == (len(CASES), [])
    assert plan(root) == {}


def test_a_moved_leg_takes_its_files_its_notebook_copy_and_its_call_history(root):
    folder = root / FIRST_LEG
    (folder / "model.pt").touch()
    (folder / "checkpoints").mkdir()
    (folder / "checkpoints" / "500.pt").touch()
    (folder / "logs").mkdir()
    (folder / "logs" / "fc-1.jsonl").touch()
    old_copy = root / "runs/GOLDEN" / f"declare-GOLDEN-pretraining-{Path(FIRST_LEG).name}.ipynb"
    old_copy.write_text("{}")
    history = {FIRST_LEG: [{"call_id": "fc-1"}], "sources/odyssey": [{"call_id": "fc-2"}]}
    (root / CALL_HISTORY).write_text(json.dumps(history))

    apply(root, plan(root))
    new = root / NEW_PATHS["pretraining_first_leg"]
    assert not folder.exists()
    for name in ("model.pt", "checkpoints/500.pt", "logs/fc-1.jsonl"):
        assert (new / name).is_file()
    assert (root / "runs/GOLDEN" / f"declare-{CASES['pretraining_first_leg'].uid}.ipynb").read_text() == "{}"
    assert not old_copy.exists()
    assert json.loads((root / CALL_HISTORY).read_text()) == {
        NEW_PATHS["pretraining_first_leg"]: [{"call_id": "fc-1"}],
        "sources/odyssey": [{"call_id": "fc-2"}],
    }


def test_an_interrupted_apply_is_finished_by_running_it_again(root):
    planned = plan(root)
    first = next(folder for folder in planned if folder.as_posix() == FIRST_LEG)
    (root / first).rename(root / planned[first].artifact_path)  # moved, manifest not yet rewritten
    apply(root, plan(root))
    assert scan(root) == (len(CASES), [])


def test_legs_that_differed_only_in_device_are_refused_not_merged(root):
    data = json.loads((root / FIRST_LEG / MANIFEST).read_text())
    data["parameters"]["model"]["device"] = "cpu"
    twin = root / Path(FIRST_LEG).with_name("0-1000-cpu0000000")
    twin.mkdir()
    (twin / MANIFEST).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="as another folder does"):
        plan(root)


def test_a_new_path_another_folder_holds_is_refused(root):
    (root / NEW_PATHS["pretraining_first_leg"]).mkdir()
    with pytest.raises(ValueError, match="which is already there"):
        plan(root)


def test_a_manifest_that_does_not_decode_refuses_everything(root):
    (root / OLD_PATHS["source"] / MANIFEST).write_text("{not json")
    with pytest.raises(ValueError, match="sources/odyssey: does not decode"):
        plan(root)
