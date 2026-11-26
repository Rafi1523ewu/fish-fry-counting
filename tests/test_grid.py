"""The grid end to end, without training: job list, sharding, resuming, summary and figures."""

import json

import pandas as pd

from fishfry import grid, plots
from fishfry.data import prepare
from fishfry.io import load_config

CFG = load_config()


def test_job_list_matches_the_design():
    todo = grid.jobs(CFG, ["published", "session_cv"])
    assert len(todo) == 1 + 4 * len(CFG["grid"])  # one published reproduction + 4 folds each
    assert sum(p == "published" for p, _, _ in todo) == 1


def test_shards_cover_every_job_once_and_runs_resume(fake_zip, tmp_path):
    path, totals = fake_zip
    data = tmp_path / "ffd"
    prepare(path, data, totals)
    runs = tmp_path / "runs"
    for shard in range(2):
        grid.run(CFG, data, runs, ["published", "session_cv"], (shard, 2), dry_run=True)
    done = sorted(runs.glob("*/*/fold*/metrics.json"))
    assert len(done) == len(grid.jobs(CFG, ["published", "session_cv"]))

    marker = done[0]
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "kept": True}))
    grid.run(CFG, data, runs, ["published", "session_cv"], dry_run=True)
    assert json.loads(marker.read_text())["kept"]  # finished runs are not redone

    manifest = json.loads((data / "splits" / "session_cv" / "fold0" / "manifest.json").read_text())
    assert len(manifest["test"]["sessions"]) == 4


def test_summary_and_figures(fake_zip, tmp_path):
    path, totals = fake_zip
    data, runs, results = tmp_path / "ffd", tmp_path / "runs", tmp_path / "results"
    prepare(path, data, totals)
    grid.run(CFG, data, runs, ["published", "session_cv"], dry_run=True)
    summary = grid.summarize(runs, results)
    assert summary.loc[("session_cv", "yolo11n_640"), "folds"] == 4
    assert summary.loc[("published", "yolo11n_640"), "folds"] == 1
    runs_df = pd.read_csv(results / "runs.csv")
    assert set(runs_df["protocol"]) == {"published", "session_cv"}
    plots.draw_all(results, "yolo11n_640")
    assert (results / "figures" / "leak_gap.png").is_file()
    assert (results / "figures" / "configs.png").is_file()
