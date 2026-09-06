#!/usr/bin/env python3
"""Create the area-normalized sample and 70/15/15 region-stratified splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_splits import build_splits
from src.experiment import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/metadata.csv")
    parser.add_argument("--regions", default="configs/la_county_regions.json")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = build_splits(
        metadata_path=resolve_path(REPO_ROOT, args.metadata),
        output_dir=resolve_path(REPO_ROOT, args.output_dir),
        config_path=resolve_path(REPO_ROOT, args.regions),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        "Created area-normalized splits: "
        + ", ".join(f"{name}={len(split)}" for name, split in splits.items())
    )


if __name__ == "__main__":
    main()
