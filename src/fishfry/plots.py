"""README figures, drawn from results/summary.csv, runs.csv and frames.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DENSITY_COLORS = {20: "#1b9e77", 40: "#7570b3", 60: "#d95f02", 80: "#e7298a"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leak_gap(runs: pd.DataFrame, config: str, path: Path) -> None:
    sub = runs[runs["config"] == config]
    metrics = [
        ("test_map", "mAP50-95"),
        ("test_dead_map", "dead-fry mAP50-95"),
        ("live_mae", "live-count MAE"),
        ("dead_mae", "dead-count MAE"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.4))
    for ax, (col, label) in zip(axes, metrics, strict=True):
        groups = [sub.loc[sub["protocol"] == p, col].dropna() for p in ("published", "session_cv")]
        ax.bar(
            [0, 1],
            [g.mean() for g in groups],
            yerr=[g.std() if len(g) > 1 else 0 for g in groups],
            color=["#bbbbbb", "#2c3e50"],
            capsize=4,
            tick_label=["published\nsplit", "new\nsession"],
        )
        ax.set_title(label)
    fig.suptitle(f"{config}: frames from seen tank sessions vs unseen ones")
    _save(fig, path)


def configs(runs: pd.DataFrame, path: Path) -> None:
    sub = runs[runs["protocol"] == "session_cv"]
    stats = sub.groupby("config")[["val_map", "test_map", "live_mae"]].agg(["mean", "std"])
    names = list(stats.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, label in (
        (a, "test_map", "test mAP50-95"),
        (b, "live_mae", "test live-count MAE"),
    ):
        ax.bar(
            range(len(names)),
            stats[(col, "mean")],
            yerr=stats[(col, "std")],
            color="#2c3e50",
            capsize=4,
            tick_label=names,
        )
        ax.set_title(f"{label} (mean ± std, 4 folds)")
        ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def count_scatter(frames: pd.DataFrame, config: str, path: Path) -> None:
    sub = frames[(frames["protocol"] == "session_cv") & (frames["config"] == config)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for density, g in sub.groupby("density"):
        color = DENSITY_COLORS.get(int(density), "gray")
        a.scatter(g["n_fish"], g["pred_fish"], s=8, alpha=0.5, color=color, label=f"{density} fry")
        b.scatter(g["n_dead"], g["pred_dead"], s=8, alpha=0.5, color=color)
    for ax, title in ((a, "live fry per frame"), (b, "dead fry per frame")):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set_xlabel("annotated")
        ax.set_ylabel("predicted")
        ax.set_title(f"{title} (held-out sessions)")
    a.legend()
    _save(fig, path)


def error_by_density(frames: pd.DataFrame, config: str, path: Path) -> None:
    sub = frames[(frames["protocol"] == "session_cv") & (frames["config"] == config)].copy()
    sub["live"] = (sub["pred_fish"] - sub["n_fish"]).abs()
    sub["dead"] = (sub["pred_dead"] - sub["n_dead"]).abs()
    stats = sub.groupby("density")[["live", "dead"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(stats))
    ax.bar([i - 0.2 for i in x], stats["live"], width=0.4, label="live", color="#2c3e50")
    ax.bar([i + 0.2 for i in x], stats["dead"], width=0.4, label="dead", color="#c0392b")
    ax.set_xticks(list(x), [f"{d} fry" for d in stats.index])
    ax.set_ylabel("mean absolute count error per frame")
    ax.set_title("Counting gets harder as the tank gets crowded")
    ax.legend()
    _save(fig, path)


def draw_all(results: Path, best_config: str) -> None:
    runs = pd.read_csv(results / "runs.csv")
    figures = results / "figures"
    if {"published", "session_cv"} <= set(runs["protocol"]):
        leak_gap(runs, "yolo11n_640", figures / "leak_gap.png")
    configs(runs, figures / "configs.png")
    if (results / "frames.csv").is_file():
        frames = pd.read_csv(results / "frames.csv")
        count_scatter(frames, best_config, figures / "count_scatter.png")
        error_by_density(frames, best_config, figures / "error_by_density.png")
