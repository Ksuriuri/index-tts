"""Backfill a wandb run from a finished SFT training log.

The SFT trainer prints lines like::

    [Train] epoch=3 step=2500 loss=4.7943 lr=9.90e-10
    [Val] epoch=3 step=2400 mel_loss=4.6776

This script parses those lines from a saved log file and logs them to wandb as a
single run, so runs launched with ``--no-wandb`` can still be visualized.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TRAIN_RE = re.compile(
    r"\[Train\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+"
    r"lr=(?P<lr>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
VAL_RE = re.compile(
    r"\[Val\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"mel_loss=(?P<loss>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)


def parse_log(log_path: Path):
    by_step: dict[int, dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            for marker, regex in (("[Train]", TRAIN_RE), ("[Val]", VAL_RE)):
                idx = raw_line.find(marker)
                if idx == -1:
                    continue
                segment = raw_line[idx:].strip()
                match = regex.search(segment)
                if not match:
                    continue
                step = int(match["step"])
                metrics = by_step.setdefault(step, {})
                if marker == "[Train]":
                    metrics.update(
                        {
                            "train/mel_loss": float(match["loss"]),
                            "train/lr": float(match["lr"]),
                            "train/epoch": int(match["epoch"]),
                        }
                    )
                else:
                    metrics.update(
                        {
                            "val/mel_loss": float(match["loss"]),
                            "val/epoch": int(match["epoch"]),
                        }
                    )

    return sorted(by_step.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Training log file")
    parser.add_argument("--project", default="IndexTTS2-SFT")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Parse only; do not upload.")
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(args.log)

    entries = parse_log(args.log)
    if not entries:
        raise RuntimeError(f"No SFT metrics found in {args.log}")

    print(f"Parsed {len(entries)} steps from {args.log}")
    print(f"First: step={entries[0][0]} metrics={entries[0][1]}")
    print(f"Last : step={entries[-1][0]} metrics={entries[-1][1]}")

    if args.dry_run:
        return

    import wandb

    wandb.init(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
        config={
            "source_log": str(args.log),
            "num_logged_steps": len(entries),
            "first_step": entries[0][0],
            "last_step": entries[-1][0],
            "note": "Backfilled from finished SFT training log; metrics only.",
        },
        resume="never",
    )
    try:
        for step, metrics in entries:
            wandb.log(metrics, step=step)
    finally:
        wandb.finish()

    print("Upload complete.")


if __name__ == "__main__":
    main()
