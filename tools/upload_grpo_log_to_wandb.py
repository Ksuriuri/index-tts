"""Backfill a wandb run from a finished GRPO training log.

The trainer prints lines like::

    [GRPO] e=1 s=10 loss=0.0017 pg=0.0017 kl=0.0006 ratio=1.0047 n=3 lr=2.00e-06
    [RefCache] entries=13 hit_rate=0.0% (hits=0 misses=13) mem=2.1MB

This script parses such lines from a saved log file and re-logs them to wandb
as a single run, so previously completed trainings can still be visualised.

Example::

    python tools/upload_grpo_log_to_wandb.py \
        --log logs/train_grpo_20260525_132020.log \
        --project IndexTTS2-GRPO \
        --run-name noiz_v2_multigen_grpo
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import wandb


GRPO_RE = re.compile(
    r"\[GRPO\]\s+e=(?P<epoch>\d+)\s+s=(?P<step>\d+)\s+"
    r"loss=(?P<loss>-?\d+\.\d+(?:e[+-]?\d+)?)\s+"
    r"pg=(?P<pg>-?\d+\.\d+(?:e[+-]?\d+)?)\s+"
    r"kl=(?P<kl>-?\d+\.\d+(?:e[+-]?\d+)?)\s+"
    r"ratio=(?P<ratio>-?\d+\.\d+(?:e[+-]?\d+)?)\s+"
    r"n=(?P<n>\d+)\s+"
    r"lr=(?P<lr>-?\d+\.\d+(?:e[+-]?\d+)?)"
)

REFCACHE_RE = re.compile(
    r"\[RefCache\]\s+entries=(?P<entries>\d+)\s+"
    r"hit_rate=(?P<hit_rate>\d+\.\d+)%\s+"
    r"\(hits=(?P<hits>\d+)\s+misses=(?P<misses>\d+)\)\s+"
    r"mem=(?P<mem>\d+\.\d+)MB"
)


def parse_log(log_path: Path):
    """Yield (global_step, metric_dict) per [GRPO] entry in the log."""
    pending_grpo = None
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            # Progress bars rewrite the same terminal line; split on '[GRPO]'
            # and '[RefCache]' so we still see those entries even when prefixed
            # by tqdm noise.
            for marker in ("[GRPO]", "[RefCache]"):
                idx = raw_line.find(marker)
                if idx == -1:
                    continue
                segment = raw_line[idx:].strip()
                if marker == "[GRPO]":
                    m = GRPO_RE.search(segment)
                    if not m:
                        continue
                    if pending_grpo is not None:
                        yield pending_grpo
                    step = int(m["step"])
                    pending_grpo = (
                        step,
                        {
                            "train/loss": float(m["loss"]),
                            "train/policy_loss": float(m["pg"]),
                            "train/kl": float(m["kl"]),
                            "train/ratio_mean": float(m["ratio"]),
                            "train/valid_samples": int(m["n"]),
                            "train/lr": float(m["lr"]),
                            "train/epoch": int(m["epoch"]),
                        },
                    )
                else:
                    m = REFCACHE_RE.search(segment)
                    if not m or pending_grpo is None:
                        continue
                    pending_grpo[1].update(
                        {
                            "refcache/entries": int(m["entries"]),
                            "refcache/hit_rate": float(m["hit_rate"]) / 100.0,
                            "refcache/hits": int(m["hits"]),
                            "refcache/misses": int(m["misses"]),
                            "refcache/mem_mb": float(m["mem"]),
                        }
                    )
    if pending_grpo is not None:
        yield pending_grpo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Training log file")
    parser.add_argument("--project", default="IndexTTS2-GRPO")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only; print first/last entries without uploading.",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(args.log)

    entries = list(parse_log(args.log))
    if not entries:
        raise RuntimeError(f"No [GRPO] lines found in {args.log}")

    print(f"Parsed {len(entries)} steps from {args.log}")
    print(f"First: step={entries[0][0]} metrics={entries[0][1]}")
    print(f"Last : step={entries[-1][0]} metrics={entries[-1][1]}")

    if args.dry_run:
        return

    config = {
        "source_log": str(args.log),
        "num_logged_steps": len(entries),
        "first_step": entries[0][0],
        "last_step": entries[-1][0],
        "note": "Backfilled from finished training log; metrics only.",
    }
    wandb.init(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
        config=config,
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
