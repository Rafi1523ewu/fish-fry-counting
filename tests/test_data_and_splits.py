import pandas as pd
import pytest
import yaml
from conftest import FRAMES, SESSIONS, build_zip

from fishfry import splits
from fishfry.data import load_index, parse_name, prepare
from fishfry.io import load_config

CFG = load_config()


def test_parse_name():
    assert parse_name("60_yellow_high_5dead_12.jpg") == {
        "session": "60_yellow_high_5dead",
        "density": 60,
        "background": "yellow",
        "level": "high",
        "n_dead_planted": 5,
        "frame": 12,
    }
    with pytest.raises(ValueError, match="photo.jpg"):
        parse_name("photo.jpg")


@pytest.mark.parametrize("wrap", [True, False])
def test_prepare_indexes_every_frame(tmp_path, wrap):
    path = tmp_path / "ffd.zip"
    totals = build_zip(path, wrap=wrap)
    index = prepare(path, tmp_path / "ffd", totals)
    assert len(index) == totals["frames"] and index["session"].nunique() == 16
    assert set(index["published_split"]) == {"train", "val", "test"}
    row = index[index["file"] == "60_white_low_3dead_1.jpg"].iloc[0]
    assert (row["n_fish"], row["n_dead"], row["density"]) == (6, 3, 60)
    _, classes = load_index(tmp_path / "ffd")
    assert classes == {0: "fish", 1: "dead_fish"}


def test_prepare_rejects_missing_label_and_wrong_totals(tmp_path):
    path = tmp_path / "ffd.zip"
    totals = build_zip(path, drop_label="20_white_high_1dead_2")
    with pytest.raises(ValueError, match="no label file"):
        prepare(path, tmp_path / "a", totals)
    build_zip(path)
    with pytest.raises(ValueError, match="frames"):
        prepare(path, tmp_path / "b", {**totals, "frames": 1})


@pytest.fixture
def index(fake_zip, tmp_path):
    path, totals = fake_zip
    return prepare(path, tmp_path / "ffd", totals)


def test_published_split_leaks_every_session(index):
    parts = splits.published(index)
    assert [len(parts[p]) for p in ("train", "val", "test")] == [16, 16, 16]
    sessions = [set(splits.sessions_of(parts[p])) for p in ("train", "val", "test")]
    assert sessions[0] == sessions[1] == sessions[2]  # the flaw the study measures


@pytest.mark.parametrize("fold", range(4))
def test_session_cv_holds_out_whole_sessions(index, fold):
    parts = splits.session_cv(index, CFG["splits"], fold, seed=0)
    sessions = {p: set(splits.sessions_of(f)) for p, f in parts.items()}
    assert not (sessions["train"] & sessions["val"] or sessions["train"] & sessions["test"])
    assert not sessions["val"] & sessions["test"]
    assert len(sessions["test"]) == 4 and len(sessions["val"]) == 2
    tested = index[index["session"].isin(sessions["test"])].drop_duplicates("session")
    assert sorted(tested["density"]) == [20, 40, 60, 80]
    assert len(set(zip(tested["background"], tested["level"], strict=True))) == 4
    val = index[index["session"].isin(sessions["val"])].drop_duplicates("session")
    assert val["density"].nunique() == 2
    assert sum(len(f) for f in parts.values()) == len(SESSIONS) * FRAMES


def test_every_session_is_tested_exactly_once(index):
    tested = [
        s
        for fold in range(4)
        for s in splits.sessions_of(splits.session_cv(index, CFG["splits"], fold, 0)["test"])
    ]
    assert sorted(tested) == sorted(index["session"].unique())


def test_session_cv_is_deterministic(index):
    a = splits.session_cv(index, CFG["splits"], 1, seed=0)
    assert a == splits.session_cv(index, CFG["splits"], 1, seed=0)


def test_materialise_writes_a_yolo_dataset(index, tmp_path):
    parts = splits.session_cv(index, CFG["splits"], 0, seed=0)
    data_yaml = splits.materialise(
        parts, tmp_path / "ffd", tmp_path / "split", {0: "fish", 1: "dead_fish"}
    )
    spec = yaml.safe_load(data_yaml.read_text())
    assert spec["names"] == {0: "fish", 1: "dead_fish"} and spec["test"] == "images/test"
    for part, files in parts.items():
        assert len(list((tmp_path / "split" / "images" / part).glob("*.jpg"))) == len(files)
        assert len(list((tmp_path / "split" / "labels" / part).glob("*.txt"))) == len(files)
    assert isinstance(pd.read_csv(tmp_path / "ffd" / "index.csv"), pd.DataFrame)
