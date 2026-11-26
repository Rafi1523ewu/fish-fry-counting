"""Show how a detector did on one frame: correct, missed and false boxes, and where it looked.

Grad-CAM needs a single class score to differentiate, which an object detector does not have,
so the heat map is EigenCAM (Muhammad & Yeasin, 2020): the first principal component of the
feature map the detection head reads. It needs no gradients and works for any YOLO model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLORS = {"tp": "#2ecc71", "fp": "#f39c12", "fn": "#e74c3c"}


def read_labels(path: Path, width: int, height: int) -> np.ndarray:
    """YOLO label file -> (n, 5) array of [class, x1, y1, x2, y2] in pixels."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows, dtype=float)
    cx, cy, w, h = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    return np.column_stack([a[:, 0], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of boxes [x1, y1, x2, y2]: (n, 4) x (m, 4) -> (n, m)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> dict:
    """Greedy same-class matching, highest confidence first.

    `pred` rows are [class, x1, y1, x2, y2, conf]; `truth` rows are [class, x1, y1, x2, y2].
    Returns boolean masks: `tp` (per prediction), `fp` (per prediction), `fn` (per truth).
    """
    tp = np.zeros(len(pred), bool)
    found = np.zeros(len(truth), bool)
    if len(pred) and len(truth):
        overlaps = iou(pred[:, 1:5], truth[:, 1:5])
        overlaps[pred[:, :1] != truth[None, :, 0]] = 0  # a dead fry boxed as live is a miss
        for i in np.argsort(-pred[:, 5]):
            candidates = np.where(~found & (overlaps[i] >= threshold))[0]
            if len(candidates):
                j = candidates[np.argmax(overlaps[i, candidates])]
                tp[i] = found[j] = True
    return {"tp": tp, "fp": ~tp, "fn": ~found}


def eigencam(activation: np.ndarray) -> np.ndarray:
    """(C, H, W) feature map -> (H, W) map in [0, 1] from its first principal component."""
    c, h, w = activation.shape
    flat = activation.reshape(c, h * w).T  # one row per location
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    cam = (flat @ vt[0]).reshape(h, w)
    if cam.max() < -cam.min():  # the component's sign is arbitrary; keep it pointing at objects
        cam = -cam
    cam = np.clip(cam, 0, None)
    return cam / cam.max() if cam.max() > 0 else cam


def _uncrop(cam: np.ndarray, input_hw: tuple[int, int], image_hw: tuple[int, int]) -> np.ndarray:
    """Undo the letterbox: crop the padding off the heat map and resize to the image."""
    from PIL import Image

    (ih, iw), (h0, w0) = input_hw, image_hw
    gain = min(ih / h0, iw / w0)
    pad_x, pad_y = (iw - w0 * gain) / 2, (ih - h0 * gain) / 2
    big = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((iw, ih)))
    crop = big[round(pad_y) : round(ih - pad_y), round(pad_x) : round(iw - pad_x)]
    return np.array(Image.fromarray(crop).resize((w0, h0))) / 255.0


def explain(model, image_path: Path, label_path: Path, imgsz: int, conf: float) -> dict:
    """Run the model once, capturing the feature map the head uses for the smallest objects."""
    from PIL import Image

    if model.predictor is None:  # the predictor builds its own (fused) copy of the network
        model.predict(str(image_path), imgsz=imgsz, verbose=False)
    net = model.predictor.model.model.model
    layer = net[net[-1].f[0]]  # Detect's highest-resolution input (stride 8)
    captured = {}
    hooks = [
        layer.register_forward_hook(lambda m, i, o: captured.__setitem__("act", o)),
        net[0].register_forward_pre_hook(lambda m, i: captured.__setitem__("input", i[0].shape)),
    ]
    try:
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    finally:
        for h in hooks:
            h.remove()

    image = np.array(Image.open(image_path).convert("RGB"))
    h0, w0 = image.shape[:2]
    boxes = result.boxes
    pred = np.column_stack(
        [boxes.cls.cpu().numpy(), boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()]
    ).reshape(-1, 6)
    truth = read_labels(label_path, w0, h0)
    act = captured["act"][0].detach().float().cpu().numpy()
    cam = _uncrop(eigencam(act), tuple(captured["input"][2:]), (h0, w0))
    return {"image": image, "pred": pred, "truth": truth, "match": match(pred, truth), "cam": cam}


def render(result: dict, names: dict[int, str], title: str, path: Path) -> dict:
    """Two panels: boxes coloured by outcome, and the EigenCAM heat map. Returns the counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pred, truth, m = result["pred"], result["truth"], result["match"]
    counts = {k: int(v.sum()) for k, v in m.items()}
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 8.2))
    a.imshow(result["image"])
    for kind, rows, mask in (("tp", pred, m["tp"]), ("fp", pred, m["fp"]), ("fn", truth, m["fn"])):
        for row in rows[mask]:
            dead = names[int(row[0])] == "dead_fish"
            a.add_patch(
                Rectangle(
                    (row[1], row[2]),
                    row[3] - row[1],
                    row[4] - row[2],
                    fill=False,
                    lw=2.2 if dead else 1.2,
                    ls="--" if dead else "-",
                    color=COLORS[kind],
                )
            )
    a.set_title(
        f"correct {counts['tp']} · false {counts['fp']} · missed {counts['fn']}\n"
        f"precision {precision:.2f} · recall {recall:.2f} (dashed = dead fry)"
    )
    b.imshow(result["image"])
    b.imshow(result["cam"], cmap="jet", alpha=0.45)
    b.set_title("EigenCAM: where the detector looks")
    for ax in (a, b):
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, fill=False, color=c, lw=2) for c in COLORS.values()]
    fig.legend(handles, ["correct", "false detection", "missed"], loc="lower center", ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return {**counts, "precision": precision, "recall": recall}
