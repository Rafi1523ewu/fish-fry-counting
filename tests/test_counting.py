import numpy as np
import pandas as pd
import pytest

from fishfry import counting

FISH, DEAD = 0, 1
GRID = [0.1, 0.3, 0.5, 0.7]


def frames(n_fish=(20, 40, 60, 80), n_dead=(1, 2, 3, 0)):
    return pd.DataFrame(
        {
            "file": [f"f{i}.jpg" for i in range(len(n_fish))],
            "session": [f"{f}_white_high_{d}dead" for f, d in zip(n_fish, n_dead, strict=True)],
            "density": list(n_fish),
            "n_fish": list(n_fish),
            "n_dead": list(n_dead),
        }
    )


def predictions(index, true_conf=0.6, noise=0, noise_conf=0.2, drop_dead_in=None):
    """Correct boxes at `true_conf`, plus `noise` false live boxes at `noise_conf`."""
    out = {}
    for _, r in index.iterrows():
        dead = r["n_dead"] - (1 if r["file"] == drop_dead_in else 0)
        boxes = [[FISH, true_conf]] * r["n_fish"] + [[DEAD, true_conf]] * dead
        boxes += [[FISH, noise_conf]] * noise
        out[r["file"]] = np.array(boxes, dtype=float).reshape(-1, 2)
    return out


def test_perfect_predictions_have_zero_error():
    index = frames()
    preds = predictions(index)
    result, _ = counting.evaluate(preds, preds, index, GRID, FISH, DEAD)
    test = result["test"]
    assert test["live_mae"] == test["dead_mae"] == test["mortality_error_pp"] == 0
    assert test["alarm_recall"] == 1 and test["false_alarm_rate"] == 0


def test_threshold_is_chosen_to_drop_low_confidence_noise():
    index = frames()
    preds = predictions(index, noise=5)
    assert counting.choose_threshold(preds, index, GRID, FISH, DEAD) == 0.3


def test_missing_one_dead_fry():
    index = frames()
    result, per_frame = counting.evaluate(
        predictions(index), predictions(index, drop_dead_in="f0.jpg"), index, GRID, FISH, DEAD
    )
    test = result["test"]
    assert test["dead_mae"] == pytest.approx(1 / 4)
    assert test["alarm_recall"] == pytest.approx(2 / 3)  # f0 had one dead fry, now none found
    assert per_frame.loc[per_frame["file"] == "f0.jpg", "pred_dead"].item() == 0


def test_no_predictions_does_not_crash():
    index = frames()
    empty = {f: np.zeros((0, 2)) for f in index["file"]}
    result, _ = counting.evaluate(empty, empty, index, GRID, FISH, DEAD)
    assert result["test"]["live_mae"] == index["n_fish"].mean()
    assert np.isfinite(result["test"]["mortality_error_pp"])


def test_breakdown_by_density():
    index = frames()
    result, _ = counting.evaluate(
        predictions(index), predictions(index, noise=2, noise_conf=0.9), index, GRID, FISH, DEAD
    )
    assert set(result["test"]["by_density"]) == {"20", "40", "60", "80"}
    assert result["test"]["by_density"]["80"]["live_mae"] == 2
