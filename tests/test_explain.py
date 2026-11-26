import numpy as np
import pytest

from fishfry.explain import eigencam, iou, match, read_labels


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert iou(a, b)[0] == pytest.approx([1.0, 50 / 150, 0.0])


def test_match_counts_correct_false_and_missed():
    truth = np.array([[0, 0, 0, 10, 10], [0, 20, 0, 30, 10], [1, 40, 0, 50, 10]], float)
    pred = np.array(
        [
            [0, 0, 0, 10, 10, 0.9],  # correct
            [0, 1, 0, 11, 10, 0.8],  # duplicate of the same fry: false
            [0, 40, 0, 50, 10, 0.7],  # right place, wrong class (dead fry called live): false
        ],
        float,
    )
    m = match(pred, truth)
    assert m["tp"].tolist() == [True, False, False]
    assert m["fp"].tolist() == [False, True, True]
    assert m["fn"].tolist() == [False, True, True]


def test_match_with_nothing():
    m = match(np.zeros((0, 6)), np.zeros((0, 5)))
    assert len(m["tp"]) == len(m["fn"]) == 0


def test_read_labels_converts_to_pixels(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("1 0.5 0.5 0.2 0.1\n")
    assert read_labels(path, 100, 200).tolist() == [[1, 40, 90, 60, 110]]


def test_eigencam_highlights_the_active_region():
    act = np.random.default_rng(0).normal(0, 0.01, (8, 6, 6))
    act[:, 2:4, 2:4] += np.arange(1, 9)[:, None, None]  # one strongly active patch
    cam = eigencam(act)
    assert cam.shape == (6, 6) and cam.max() == pytest.approx(1)
    assert cam[2:4, 2:4].min() > 0.9 and cam[0, 0] < 0.1
