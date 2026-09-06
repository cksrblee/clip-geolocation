#!/usr/bin/env python3
"""Generate the controlled cue, structure, and layout interventions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import require_image_files
from src.experiment import resolve_path
from src.interventions.generator import build_interventions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/val_metadata.csv")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--output-dir", default="interventions/val")
    parser.add_argument(
        "--bundles", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"]
    )
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    metadata_path = resolve_path(REPO_ROOT, args.metadata)
    image_dir = resolve_path(REPO_ROOT, args.image_dir)
    metadata = pd.read_csv(metadata_path)
    require_image_files(metadata, image_dir)
    requested_weights = resolve_path(REPO_ROOT, args.yolo_weights)
    yolo_weights = (
        str(requested_weights) if requested_weights.is_file() else args.yolo_weights
    )
    build_interventions(
        metadata=metadata,
        image_dir=image_dir,
        output_dir=resolve_path(REPO_ROOT, args.output_dir),
        bundles=args.bundles,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        yolo_weights=yolo_weights,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
