#!/usr/bin/env python3
"""Compute regional cue coverage and one-way ANOVA for four visual cues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.cue_statistics import (
    cue_anova,
    prepare_cue_coverage,
    summarize_cue_coverage,
)
from src.experiment import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/val_metadata.csv")
    parser.add_argument(
        "--mask-coverage", default="interventions/val/mask_coverage.csv"
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    metadata = pd.read_csv(resolve_path(REPO_ROOT, args.metadata))
    coverage = pd.read_csv(resolve_path(REPO_ROOT, args.mask_coverage))
    records = prepare_cue_coverage(coverage, metadata)
    summary = summarize_cue_coverage(records)
    anova = cue_anova(records)

    results_dir = resolve_path(REPO_ROOT, args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(results_dir / "visual_cue_coverage.csv", index=False)
    summary.to_csv(results_dir / "visual_cue_regional_summary.csv", index=False)
    anova.to_csv(results_dir / "visual_cue_anova.csv", index=False)
    print(summary.to_string(index=False))
    print("\nOne-way ANOVA")
    print(anova.to_string(index=False))


if __name__ == "__main__":
    main()
