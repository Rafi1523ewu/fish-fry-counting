"""A tiny stand-in for the Fish Fry Dataset with the same structure and naming."""

import io
import json
import zipfile

import pytest
from PIL import Image

# The 16 real sessions (density, background, level, dead fry planted).
SESSIONS = [
    (20, "white", "high", 1), (20, "white", "low", 2), (20, "yellow", "high", 1),
    (20, "yellow", "low", 2), (40, "white", "high", 4), (40, "white", "low", 2),
    (40, "yellow", "high", 3), (40, "yellow", "low", 2), (60, "white", "high", 4),
    (60, "white", "low", 3), (60, "yellow", "high", 5), (60, "yellow", "low", 3),
    (80, "white", "high", 5), (80, "white", "low", 6), (80, "yellow", "high", 5),
    (80, "yellow", "low", 6),
]  # fmt: skip
FRAMES = 3  # per session: one in each published folder, like the real (leaky) split
FOLDERS = {"train": "train", "validation": "val", "test": "test"}


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    return buf.getvalue()


def build_zip(path, drop_label: str | None = None, wrap: bool = True) -> dict:
    """Write a fake FFD.zip (optionally wrapped in an outer zip); return expected totals."""
    root = "Fish Fry Dataset"
    categories = [{"id": 1, "name": "fish"}, {"id": 2, "name": "dead_fish"}]
    coco = {f: {"images": [], "annotations": [], "categories": categories} for f in FOLDERS}
    totals = {"frames": 0, "fish": 0, "dead_fish": 0, "sessions": len(SESSIONS)}
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for density, bg, level, dead in SESSIONS:
            for frame in range(1, FRAMES + 1):
                folder = list(FOLDERS)[frame % 3]
                name = f"{density}_{bg}_{level}_{dead}dead_{frame}"
                live = density // 10
                z.writestr(f"{root}/{folder}/{name}.jpg", _jpg())
                lines = ["0 0.5 0.5 0.1 0.1"] * live + ["1 0.2 0.2 0.1 0.1"] * dead
                if name != drop_label:
                    z.writestr(f"{root}/{folder}/{name}.txt", "\n".join(lines) + "\n")
                coco[folder]["annotations"] += [{"category_id": 1}] * live
                coco[folder]["annotations"] += [{"category_id": 2}] * dead
                totals["frames"] += 1
                totals["fish"] += live
                totals["dead_fish"] += dead
        for folder, data in coco.items():
            z.writestr(f"{root}/annotations/instances_{FOLDERS[folder]}.json", json.dumps(data))
    if wrap:
        with zipfile.ZipFile(path, "w") as outer:
            outer.writestr("Fish Fry Dataset (FFD)/FFD.zip", inner.getvalue())
    else:
        path.write_bytes(inner.getvalue())
    return totals


@pytest.fixture
def fake_zip(tmp_path):
    path = tmp_path / "Fish Fry Dataset.zip"
    return path, build_zip(path)
