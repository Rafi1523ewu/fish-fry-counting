"""fishfry: prepare the data, run the experiment grid, summarise and plot the results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fishfry import data, grid, plots
from fishfry.io import DEFAULT_CONFIG, load_config

log = logging.getLogger("fishfry")


def best_config(results: Path) -> str:
    """The configuration with the best mean *validation* mAP50-95 under session CV."""
    runs = pd.read_csv(results / "runs.csv")
    cv = runs[runs["protocol"] == "session_cv"]
    return cv.groupby("config")["val_map"].mean().idxmax()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download (or use a local zip), unpack, index and verify")
    p.add_argument("--zip", help="local FFD.zip, or an archive that wraps it")
    p.add_argument("--out", default="data/ffd")

    p = sub.add_parser("grid", help="train and evaluate every job (resumable)")
    p.add_argument("--data", default="data/ffd")
    p.add_argument("--runs", default="runs")
    p.add_argument("--protocols", default="published,session_cv")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    p.add_argument("--gpus", type=int, default=1, help="> 1 launches one shard per GPU")
    p.add_argument("--dry-run", action="store_true", help="write placeholder metrics, no training")

    p = sub.add_parser("summarize", help="aggregate runs into results/ and draw the figures")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)

    if args.command == "prepare":
        source = (
            Path(args.zip)
            if args.zip
            else data.download(
                cfg["data"]["url"], cfg["data"]["sha256"], Path(args.out).parent / "FFD.zip"
            )
        )
        data.prepare(source, args.out, cfg["data"]["expected"])
    elif args.command == "grid":
        protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
        unknown = set(protocols) - set(cfg["protocols"])
        if unknown:
            parser.error(
                f"unknown protocols {sorted(unknown)}; choose from {list(cfg['protocols'])}"
            )
        if args.gpus > 1:
            grid.launch(args.config, Path(args.data), Path(args.runs), protocols, args.gpus)
        else:
            i, n = (int(x) for x in args.shard.split("/"))
            grid.run(cfg, Path(args.data), Path(args.runs), protocols, (i, n), args.dry_run)
    elif args.command == "summarize":
        out = Path(args.out)
        (out / "frames.csv").unlink(missing_ok=True)
        grid.summarize(Path(args.runs), out)
        chosen = best_config(out)
        log.info("selected %s by validation mAP50-95", chosen)
        (out / "selected_config.txt").write_text(chosen + "\n")
        plots.draw_all(out, chosen)


if __name__ == "__main__":
    main()
