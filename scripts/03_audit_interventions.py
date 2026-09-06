#!/usr/bin/env python3
"""Audit completeness, readability, and mask coverage for all interventions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiment import INTERVENTIONS, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/val_metadata.csv")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--interventions-dir", default="interventions/val")
    parser.add_argument("--output", default="results/intervention_audit.csv")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    metadata = pd.read_csv(resolve_path(REPO_ROOT, args.metadata))
    image_dir = resolve_path(REPO_ROOT, args.image_dir)
    intervention_dir = resolve_path(REPO_ROOT, args.interventions_dir)
    expected = metadata["filename"].astype(str).tolist()
    rows: list[dict] = []

    for spec in INTERVENTIONS:
        condition_dir = intervention_dir / spec.name
        missing = 0
        corrupted = 0
        unexpected_size = 0
        for filename in tqdm(expected, desc=spec.name, leave=False):
            path = condition_dir / filename
            if not path.is_file():
                missing += 1
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                if spec.bundle in {"A", "A-control", "B"}:
                    with Image.open(path) as intervened, Image.open(
                        image_dir / filename
                    ) as original:
                        if intervened.size != original.size:
                            unexpected_size += 1
            except Exception:
                corrupted += 1
        rows.append(
            {
                "Intervention": spec.name,
                "Bundle": spec.bundle,
                "Expected": len(expected),
                "Present": len(expected) - missing,
                "Missing": missing,
                "Corrupted": corrupted,
                "Unexpected_Size": unexpected_size,
                "Coverage_pct": 100.0 * (len(expected) - missing) / len(expected),
            }
        )

    report = pd.DataFrame(rows)
    coverage_path = intervention_dir / "mask_coverage.csv"
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path)
        means = coverage.groupby("condition")["coverage_pct"].mean()
        report["Mean_Mask_Coverage_pct"] = report["Intervention"].map(means)

    output_path = resolve_path(REPO_ROOT, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(report.to_string(index=False))

    invalid = report[["Missing", "Corrupted", "Unexpected_Size"]].to_numpy().sum()
    if invalid and not args.allow_incomplete:
        raise SystemExit(f"Intervention audit failed with {int(invalid)} invalid files")


if __name__ == "__main__":
    main()
