"""Two evaluation protocols over the same frames.

`published` is the dataset's own split. All 16 recording sessions appear in its train,
validation and test parts, so test frames have near-identical neighbours in training.
`session_cv` holds out whole sessions: a fold's test set is four sessions the model never saw.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import pandas as pd
import yaml

PARTS = ("train", "val", "test")


def published(index: pd.DataFrame) -> dict[str, list[str]]:
    return {p: sorted(index.loc[index["published_split"] == p, "file"]) for p in PARTS}


def session_cv(index: pd.DataFrame, cfg: dict, fold: int, seed: int) -> dict[str, list[str]]:
    square = cfg["latin_square"]
    if not 0 <= fold < len(square):
        raise ValueError(f"fold must be in [0, {len(square)}), got {fold}")
    sessions = index.drop_duplicates("session").set_index(["density", "background", "level"])
    sessions = sessions["session"]

    test = []
    for density, combo in zip(cfg["densities"], square[fold], strict=True):
        background, level = combo.split("-")
        test.append(sessions.loc[(density, background, level)])

    # Validation: whole sessions too, at different densities, chosen reproducibly per fold.
    rest = index[~index["session"].isin(test)].drop_duplicates("session")
    rng = random.Random(seed + fold)
    densities = rng.sample(sorted(rest["density"].unique()), cfg["val_sessions_per_fold"])
    val = [rng.choice(sorted(rest.loc[rest["density"] == d, "session"])) for d in densities]

    part = index["session"].map(lambda s: "test" if s in test else "val" if s in val else "train")
    return {p: sorted(index.loc[part == p, "file"]) for p in PARTS}


def sessions_of(files: list[str]) -> list[str]:
    return sorted({f.rsplit("_", 1)[0] for f in files})


def materialise(
    split: dict[str, list[str]], data_dir: str | Path, out: str | Path, classes: dict[int, str]
) -> Path:
    """Write an Ultralytics dataset folder (hard links, so no image is stored twice)."""
    data_dir, out = Path(data_dir), Path(out)
    for part, files in split.items():
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            target = out / kind / part
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                src = data_dir / kind / (f[:-4] + ext)
                dst = target / src.name
                if dst.exists():
                    continue
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    data_yaml = {
        "path": str(out.resolve()),
        **{p: f"images/{p}" for p in PARTS},
        "names": {k: classes[k] for k in sorted(classes)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    manifest = {p: {"frames": len(f), "sessions": sessions_of(f)} for p, f in split.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out / "data.yaml"
