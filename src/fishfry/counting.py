"""What a farm needs from a detector: how many live fry, how many dead, and the mortality rate.

Detections are collected once at the lowest confidence of the grid, so every threshold is just
a filter. The threshold is chosen on validation sessions and frozen before the test sessions
are scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Predictions = dict[str, np.ndarray]  # file -> (n_boxes, 2) array of [class_id, confidence]


def predict(
    model, image_paths: list, imgsz: int, conf: float, iou: float, max_det: int
) -> Predictions:
    out = {}
    for path in image_paths:
        result = model.predict(
            str(path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False
        )[0]
        boxes = result.boxes
        out[path.name] = np.column_stack(
            [boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()]
        ).reshape(-1, 2)
    return out


def counts_at(preds: Predictions, threshold: float, fish_id: int, dead_id: int) -> pd.DataFrame:
    rows = []
    for file, boxes in preds.items():
        kept = boxes[boxes[:, 1] >= threshold, 0]
        rows.append(
            {
                "file": file,
                "pred_fish": int((kept == fish_id).sum()),
                "pred_dead": int((kept == dead_id).sum()),
            }
        )
    return pd.DataFrame(rows, columns=["file", "pred_fish", "pred_dead"])


def _frames(preds, threshold, fish_id, dead_id, index: pd.DataFrame) -> pd.DataFrame:
    counts = counts_at(preds, threshold, fish_id, dead_id)
    return index.merge(counts, on="file", how="inner")


def choose_threshold(preds, index, grid, fish_id, dead_id) -> float:
    """The confidence threshold that makes validation counts most accurate."""

    def error(t: float) -> float:
        f = _frames(preds, t, fish_id, dead_id, index)
        return float(
            (f["pred_fish"] - f["n_fish"]).abs().mean()
            + (f["pred_dead"] - f["n_dead"]).abs().mean()
        )

    return min(grid, key=error)


def mortality(dead, total) -> np.ndarray:
    dead, total = np.asarray(dead, float), np.asarray(total, float)
    return np.divide(dead, total, out=np.zeros_like(dead), where=total > 0)


def summarize(frames: pd.DataFrame) -> dict:
    """Count errors per frame, per session (median over its frames) and per density."""
    live_err = frames["pred_fish"] - frames["n_fish"]
    dead_err = frames["pred_dead"] - frames["n_dead"]
    true_rate = mortality(frames["n_dead"], frames["n_fish"] + frames["n_dead"])
    pred_rate = mortality(frames["pred_dead"], frames["pred_fish"] + frames["pred_dead"])
    has_dead, no_dead = frames["n_dead"] > 0, frames["n_dead"] == 0

    sessions = frames.groupby("session").agg(
        density=("density", "first"),
        n_fish=("n_fish", "median"),
        pred_fish=("pred_fish", "median"),
        n_dead=("n_dead", "median"),
        pred_dead=("pred_dead", "median"),
    )
    by_density = {}
    for density, g in frames.groupby("density"):
        by_density[str(density)] = {
            "frames": int(len(g)),
            "live_mae": float((g["pred_fish"] - g["n_fish"]).abs().mean()),
            "live_mape": float(((g["pred_fish"] - g["n_fish"]).abs() / g["n_fish"]).mean()),
            "dead_mae": float((g["pred_dead"] - g["n_dead"]).abs().mean()),
        }
    return {
        "frames": int(len(frames)),
        "live_mae": float(live_err.abs().mean()),
        "live_mape": float((live_err.abs() / frames["n_fish"]).mean()),
        "live_bias": float(live_err.mean()),
        "dead_mae": float(dead_err.abs().mean()),
        "dead_bias": float(dead_err.mean()),
        "mortality_error_pp": float(100 * np.abs(pred_rate - true_rate).mean()),
        "alarm_recall": float((frames.loc[has_dead, "pred_dead"] > 0).mean())
        if has_dead.any()
        else None,
        "false_alarm_rate": float((frames.loc[no_dead, "pred_dead"] > 0).mean())
        if no_dead.any()
        else None,
        "session_live_mae": float((sessions["pred_fish"] - sessions["n_fish"]).abs().mean()),
        "session_dead_mae": float((sessions["pred_dead"] - sessions["n_dead"]).abs().mean()),
        "by_density": by_density,
    }


def evaluate(val_preds, test_preds, index, grid, fish_id, dead_id) -> tuple[dict, pd.DataFrame]:
    threshold = choose_threshold(val_preds, index, grid, fish_id, dead_id)
    frames = _frames(test_preds, threshold, fish_id, dead_id, index)
    val_frames = _frames(val_preds, threshold, fish_id, dead_id, index)
    return {
        "threshold": threshold,
        "val": summarize(val_frames),
        "test": summarize(frames),
    }, frames
