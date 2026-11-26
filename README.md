# Counting Fish Fry in a Tank the Model Has Never Seen

[![CI](https://github.com/Rafi1523ewu/fish-fry-counting/actions/workflows/ci.yml/badge.svg)](https://github.com/Rafi1523ewu/fish-fry-counting/actions/workflows/ci.yml)

Hatcheries count fry to set stocking density and watch for dead fry to catch disease early.
Object detectors trained on the public Fish Fry Dataset report about 0.97 mAP50, but that
number is measured on frames from the same tank sessions the model was trained on. This
project asks the question a farm actually faces: **on a new tank, how accurately does a
detector count live fry, count dead fry, and estimate mortality?**

## The dataset, and why its split overstates accuracy

[Fish Fry Dataset (FFD)](https://data.mendeley.com/datasets/y52ffd3xdc) (Zhai, Xu & Wu, Mendeley
Data, 2024, CC BY 4.0): 1,101 frames (1080×1920) of largemouth bass fry with 51,119 live and 3,586
dead fry annotated.

The file names encode how each frame was filmed. `60_yellow_high_5dead_12.jpg` is frame 12 of
the session with 60 fry, a yellow background, level "high" (a recording condition the dataset
does not explain), and 5 dead fry placed in the tank. That gives **16 sessions** (4 densities ×
2 backgrounds × 2 levels) of 33–50 frames each.

The dataset's own train / validation / test split (656 / 220 / 225 frames) draws frames from
**all 16 sessions in every part**. Neighbouring frames of one session are nearly identical, so a
test frame almost always has a twin in training. Scores on that split measure recall of a seen
tank, not generalisation to a new one.

## Protocol

- **`published`**: the dataset's split, used only to reproduce the original results and measure
  how much the leak inflates them.
- **`session_cv` (primary)**: 4-fold cross-validation over whole sessions. Each fold tests four
  sessions the model never saw, one per density, arranged so every background/level combination
  appears once per fold and every session is tested exactly once:

  | fold | 20 fry | 40 fry | 60 fry | 80 fry |
  |---|---|---|---|---|
  | 0 | white-high | white-low | yellow-high | yellow-low |
  | 1 | white-low | yellow-high | yellow-low | white-high |
  | 2 | yellow-high | yellow-low | white-high | white-low |
  | 3 | yellow-low | white-high | white-low | yellow-high |

  Two further whole sessions per fold form the validation set (early stopping and the counting
  threshold); the remaining ten are for training.
- **Models**, fixed in advance and selected by validation mAP50-95 only: YOLO11n at 640 px (the
  original setting), YOLO11n at 960 px (the fry are small), YOLO11s at 640 px and YOLO12n at
  640 px, each for 50 epochs. The grid is sized to fit one weekly Kaggle GPU quota.
- **Metrics**: mAP50 and mAP50-95 per class; per-frame live-count and dead-count error;
  mortality-rate error in percentage points; dead-fry alarm recall (frames with a dead fry
  where at least one is found); session-level count error (median over a session's frames); all
  broken down by stocking density. The confidence threshold for counting is chosen on the
  validation sessions, then frozen.

## Results

The original coursework reported, on the published split:

| Model | test mAP50 | test mAP50-95 | dead-fry mAP50-95 |
|---|---|---|---|
| YOLO12n, 640 px, 50 epochs | 0.968 | 0.689 | 0.644 |
| YOLO11n, 640 px, 50 epochs | 0.965 | 0.681 | 0.632 |

**Results on held-out sessions are pending the GPU run** ([`kaggle/run.ipynb`](kaggle/run.ipynb)).

## Usage

```bash
git clone https://github.com/Rafi1523ewu/fish-fry-counting.git
cd fish-fry-counting
pip install -e ".[dev]"

fishfry prepare --out data/ffd            # downloads FFD.zip from Mendeley and checks its SHA-256
fishfry grid --data data/ffd --runs runs  # trains and evaluates every job; finished jobs are skipped
fishfry summarize --runs runs --out results
```

`fishfry prepare --zip path/to/FFD.zip` uses a local copy instead. On a machine with several
GPUs, `fishfry grid --gpus N` runs one shard per GPU. Every setting lives in
[`configs/default.yaml`](configs/default.yaml).

## Project structure

```
configs/default.yaml     data checksum, session Latin square, model grid, counting thresholds
src/fishfry/data.py      download, unpack, index frames by session, verify totals
src/fishfry/splits.py    published and session-held-out splits, Ultralytics dataset folders
src/fishfry/train.py     train one configuration, evaluate detection and counting
src/fishfry/counting.py  live/dead counts, threshold chosen on validation, farm metrics
src/fishfry/grid.py      resumable grid, sharded across GPUs; summary tables
src/fishfry/plots.py     figures for this README
kaggle/run.ipynb         the full run on a Kaggle GPU notebook
tests/                   data, split, counting and grid tests on a synthetic dataset
```

## Testing

```bash
pytest -q
```

The tests build a miniature dataset with the real session structure and check that the
published split leaks every session, that session folds never share a session and test each
one exactly once, that counting metrics are exact on known predictions, and that the grid
shards and resumes correctly. They run on CPU without downloading any model weights.

## License

Code: MIT. Dataset: CC BY 4.0, by Zhai, Xu & Wu (2024); cite
[doi:10.17632/y52ffd3xdc.1](https://doi.org/10.17632/y52ffd3xdc.1).
