"""Get the Fish Fry Dataset, unpack it and index every frame by its recording session.

Each file name encodes the tank set-up it was filmed in, e.g. `60_yellow_high_5dead_12.jpg` is
frame 12 of the session with 60 fry on a yellow background, level "high", with 5 dead fry
placed. Frames of one session are near-duplicates, so the session is the unit for splitting.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

log = logging.getLogger("fishfry")

NAME = re.compile(r"^(\d+)_(white|yellow)_(high|low)_(\d+)dead_(\d+)\.jpg$")
FOLDERS = {"train": "train", "validation": "val", "test": "test"}
CLASS_NAMES = ("fish", "dead_fish")


def parse_name(filename: str) -> dict:
    match = NAME.match(filename)
    if not match:
        raise ValueError(f"unexpected file name {filename!r}")
    density, background, level, dead, frame = match.groups()
    return {
        "session": f"{density}_{background}_{level}_{dead}dead",
        "density": int(density),
        "background": background,
        "level": level,
        "n_dead_planted": int(dead),
        "frame": int(frame),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, expected_sha256: str, dest: Path) -> Path:
    """Download the official archive once and refuse it unless the checksum matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        log.info("downloading %s", url)
        with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
            shutil.copyfileobj(response, f)
    actual = sha256(dest)
    if actual != expected_sha256:
        raise ValueError(f"{dest} has sha256 {actual}, expected {expected_sha256}")
    return dest


def _open_ffd(source: Path) -> zipfile.ZipFile:
    """Accept FFD.zip itself or an archive that wraps it (as some re-downloads do)."""
    outer = zipfile.ZipFile(source)
    if any(n.lower().endswith(".jpg") for n in outer.namelist()):
        return outer
    inner = [n for n in outer.namelist() if n.lower().endswith(".zip")]
    if len(inner) != 1:
        raise ValueError(f"{source} holds neither images nor exactly one inner zip: {inner}")
    return zipfile.ZipFile(io.BytesIO(outer.read(inner[0])))


def _yolo_class_map(archive: zipfile.ZipFile, label_counts: Counter) -> dict[int, str]:
    """Map YOLO class ids to names by matching box totals with the COCO annotations.

    The two formats number classes independently; matching on counts makes a swapped order
    impossible to miss.
    """
    coco_counts = Counter()
    for name in archive.namelist():
        if name.endswith(".json") and "/annotations/" in name:
            coco = json.loads(archive.read(name))
            names = {c["id"]: c["name"] for c in coco["categories"]}
            coco_counts.update(names[a["category_id"]] for a in coco["annotations"])
    mapping = {}
    for class_id, count in label_counts.items():
        matches = [n for n, c in coco_counts.items() if c == count]
        if len(matches) != 1:
            raise ValueError(f"cannot match YOLO class {class_id} ({count} boxes) to {coco_counts}")
        mapping[class_id] = matches[0]
    if sorted(mapping.values()) != sorted(CLASS_NAMES):
        raise ValueError(f"unexpected classes {mapping}")
    return mapping


def prepare(source: str | Path, out: str | Path, expected: dict) -> pd.DataFrame:
    """Unpack to `out/images` + `out/labels` (flat), write `out/index.csv`, verify totals."""
    out = Path(out)
    archive = _open_ffd(Path(source))
    images, labels = out / "images", out / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    rows, label_counts, box_counts = [], {}, Counter()
    members = {Path(n).name: n for n in archive.namelist() if not n.endswith("/")}
    for name, member in members.items():
        if not name.lower().endswith(".jpg"):
            continue
        folder = Path(member).parent.name
        if folder not in FOLDERS:
            raise ValueError(f"{member} is not in one of {sorted(FOLDERS)}")
        label_name = name[:-4] + ".txt"
        if label_name not in members:
            raise ValueError(f"no label file for {member}")
        text = archive.read(members[label_name]).decode()
        classes = [int(line.split()[0]) for line in text.splitlines() if line.strip()]
        box_counts.update(classes)
        label_counts[name] = Counter(classes)
        (images / name).write_bytes(archive.read(member))
        (labels / label_name).write_text(text)
        rows.append({"file": name, **parse_name(name), "published_split": FOLDERS[folder]})

    class_map = _yolo_class_map(archive, box_counts)
    fish_id = next(k for k, v in class_map.items() if v == "fish")
    dead_id = next(k for k, v in class_map.items() if v == "dead_fish")
    for row in rows:
        row["n_fish"] = label_counts[row["file"]][fish_id]
        row["n_dead"] = label_counts[row["file"]][dead_id]

    index = pd.DataFrame(rows).sort_values(["session", "frame"]).reset_index(drop=True)
    totals = {
        "frames": len(index),
        "fish": int(index["n_fish"].sum()),
        "dead_fish": int(index["n_dead"].sum()),
        "sessions": int(index["session"].nunique()),
    }
    mismatched = {k: (totals[k], v) for k, v in expected.items() if totals[k] != v}
    if mismatched:
        raise ValueError(f"dataset totals differ from expected (got, expected): {mismatched}")
    index.to_csv(out / "index.csv", index=False)
    (out / "classes.json").write_text(json.dumps({str(k): v for k, v in class_map.items()}))
    log.info("prepared %s", totals)
    return index


def load_index(data_dir: str | Path) -> tuple[pd.DataFrame, dict[int, str]]:
    data_dir = Path(data_dir)
    classes = json.loads((data_dir / "classes.json").read_text())
    return pd.read_csv(data_dir / "index.csv"), {int(k): v for k, v in classes.items()}
