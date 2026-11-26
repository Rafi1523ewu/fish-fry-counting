"""Train one configuration on one split, then evaluate detection and counting."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fishfry import counting
from fishfry.data import load_index
from fishfry.io import write_json

log = logging.getLogger("fishfry")


def _detection(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    per_class = {
        names[c]: {"map50": float(box.ap50[i]), "map": float(box.ap[i])}
        for i, c in enumerate(box.ap_class_index)
    }
    return {
        "map50": float(box.map50),
        "map": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
    }


def train_and_eval(data_dir: Path, split_dir: Path, entry: dict, cfg: dict, out: Path) -> dict:
    from ultralytics import YOLO  # heavy import kept out of the CLI's startup path

    started = time.time()
    # Ultralytics nests relative project paths under its own runs/ folder, so pass absolutes.
    out = out.resolve()
    index, classes = load_index(data_dir)
    model = YOLO(entry["weights"])
    model.train(
        data=str(split_dir / "data.yaml"),
        imgsz=entry["imgsz"],
        batch=entry["batch"],
        epochs=cfg["train"]["epochs"],
        patience=cfg["train"]["patience"],
        workers=cfg["train"]["workers"],
        seed=cfg["seed"],
        deterministic=True,
        project=str(out),
        name="train",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    weights = Path(model.trainer.save_dir) / "weights"
    best = YOLO(str(weights / "best.pt"))
    names = {int(k): v for k, v in best.names.items()}
    fish_id = next(k for k, v in names.items() if v == "fish")
    dead_id = next(k for k, v in names.items() if v == "dead_fish")
    if {fish_id: "fish", dead_id: "dead_fish"} != {k: classes[k] for k in (fish_id, dead_id)}:
        raise ValueError(f"model classes {names} disagree with the dataset {classes}")

    def evaluate(part: str) -> dict:
        res = best.val(
            data=str(split_dir / "data.yaml"),
            split=part,
            imgsz=entry["imgsz"],
            batch=entry["batch"],
            plots=False,
            verbose=False,
            project=str(out),
            name=f"val_{part}",
            exist_ok=True,
        )
        return _detection(res, names)

    count_cfg = cfg["counting"]

    def preds(part: str):
        images = sorted((split_dir / "images" / part).glob("*.jpg"))
        return counting.predict(
            best,
            images,
            entry["imgsz"],
            min(count_cfg["conf_grid"]),
            count_cfg["iou"],
            count_cfg["max_det"],
        )

    counts, frames = counting.evaluate(
        preds("val"), preds("test"), index, count_cfg["conf_grid"], fish_id, dead_id
    )
    frames.to_csv(out / "frames.csv", index=False)
    (weights / "last.pt").unlink(missing_ok=True)

    metrics = {
        "config": entry,
        "split": json.loads((split_dir / "manifest.json").read_text()),
        "detection": {"val": evaluate("val"), "test": evaluate("test")},
        "counting": counts,
        "seconds": round(time.time() - started),
    }
    write_json(metrics, out / "metrics.json")
    log.info(
        "%s: test mAP50-95 %.3f, live-count MAE %.2f",
        entry["name"],
        metrics["detection"]["test"]["map"],
        counts["test"]["live_mae"],
    )
    return metrics
