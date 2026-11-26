"""Run every (protocol, configuration, fold) job, resumably and sharded across GPUs."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fishfry import splits
from fishfry.data import load_index
from fishfry.io import write_json

log = logging.getLogger("fishfry")


def jobs(cfg: dict, protocols: list[str]) -> list[tuple[str, dict, int]]:
    by_name = {e["name"]: e for e in cfg["grid"]}
    out = []
    for protocol in protocols:
        names = cfg["protocols"][protocol]
        entries = list(by_name.values()) if names == "all" else [by_name[n] for n in names]
        folds = [0] if protocol == "published" else range(len(cfg["splits"]["latin_square"]))
        out += [(protocol, entry, fold) for entry in entries for fold in folds]
    return out


def split_dir(data_dir: Path, protocol: str, fold: int, cfg: dict) -> Path:
    out = data_dir / "splits" / protocol / f"fold{fold}"
    if not (out / "data.yaml").is_file():
        index, classes = load_index(data_dir)
        if protocol == "published":
            parts = splits.published(index)
        else:
            parts = splits.session_cv(index, cfg["splits"], fold, cfg["seed"])
        splits.materialise(parts, data_dir, out, classes)
    return out


def run(
    cfg: dict,
    data_dir: Path,
    runs: Path,
    protocols: list[str],
    shard: tuple[int, int] = (0, 1),
    dry_run: bool = False,
) -> None:
    todo = jobs(cfg, protocols)
    index, count = shard
    for n, (protocol, entry, fold) in enumerate(todo):
        if n % count != index:
            continue
        out = runs / protocol / entry["name"] / f"fold{fold}"
        if (out / "metrics.json").is_file():
            log.info("[%d/%d] skip %s (done)", n + 1, len(todo), out)
            continue
        log.info("[%d/%d] %s", n + 1, len(todo), out)
        sdir = split_dir(data_dir, protocol, fold, cfg)
        if dry_run:
            write_json(_fake_metrics(entry, fold), out / "metrics.json")
            continue
        from fishfry.train import train_and_eval

        train_and_eval(data_dir, sdir, entry, cfg, out)


def launch(cfg_path: str, data_dir: Path, runs: Path, protocols: list[str], gpus: int) -> None:
    """One grid shard per GPU; fail loudly if any shard fails."""
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fishfry.cli",
                "--config",
                cfg_path,
                "grid",
                "--data",
                str(data_dir),
                "--runs",
                str(runs),
                "--protocols",
                ",".join(protocols),
                "--shard",
                f"{i}/{gpus}",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)},
        )
        for i in range(gpus)
    ]
    failed = [w.args for w in workers if w.wait() != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} of {gpus} shard(s) failed")


METRICS = {
    "val_map": ("detection", "val", "map"),
    "test_map50": ("detection", "test", "map50"),
    "test_map": ("detection", "test", "map"),
    "test_dead_map": ("detection", "test", "per_class", "dead_fish", "map"),
    "threshold": ("counting", "threshold"),
    "live_mae": ("counting", "test", "live_mae"),
    "live_mape": ("counting", "test", "live_mape"),
    "dead_mae": ("counting", "test", "dead_mae"),
    "mortality_error_pp": ("counting", "test", "mortality_error_pp"),
    "alarm_recall": ("counting", "test", "alarm_recall"),
    "session_live_mae": ("counting", "test", "session_live_mae"),
}


def _get(d: dict, path: tuple):
    for key in path:
        d = d.get(key) if isinstance(d, dict) else None
    return d


def summarize(runs: Path, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(runs.glob("*/*/fold*/metrics.json")):
        protocol, config, fold = path.parts[-4], path.parts[-3], int(path.parts[-2][4:])
        m = json.loads(path.read_text())
        rows.append(
            {
                "protocol": protocol,
                "config": config,
                "fold": fold,
                **{k: _get(m, p) for k, p in METRICS.items()},
            }
        )
        frames = path.parent / "frames.csv"
        if frames.is_file():
            df = pd.read_csv(frames).assign(protocol=protocol, config=config, fold=fold)
            df.to_csv(
                out / "frames.csv", mode="a", index=False, header=not (out / "frames.csv").exists()
            )
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "runs.csv", index=False)
    summary = per_run.groupby(["protocol", "config"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "folds", per_run.groupby(["protocol", "config"]).size())
    summary.round(4).to_csv(out / "summary.csv")
    log.info("summarised %d runs into %s", len(per_run), out)
    return summary


def _fake_metrics(entry: dict, fold: int) -> dict:
    """Shape-correct metrics for --dry-run, used by the tests and to check a grid end to end."""
    score = 0.5 + 0.01 * fold
    test = {
        "live_mae": 1.0 + fold,
        "live_mape": 0.02,
        "dead_mae": 0.5,
        "mortality_error_pp": 1.0,
        "alarm_recall": 0.9,
        "session_live_mae": 0.5,
    }
    return {
        "config": entry,
        "detection": {
            p: {"map50": score, "map": score, "per_class": {"dead_fish": {"map": score}}}
            for p in ("val", "test")
        },
        "counting": {"threshold": 0.3, "val": test, "test": test},
    }
