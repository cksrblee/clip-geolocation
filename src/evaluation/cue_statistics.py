"""Visual-cue coverage summaries and one-way ANOVA."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


CUE_CONDITIONS = {
    "text_masked": "text",
    "vehicle_masked": "vehicle",
    "sky_masked": "sky",
    "vegetation_masked": "vegetation",
}


def prepare_cue_coverage(
    coverage: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = {"filename", "condition", "coverage_pct"}
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Mask coverage is missing: {', '.join(sorted(missing))}")

    records = coverage[coverage["condition"].isin(CUE_CONDITIONS)].copy()
    records["cue"] = records["condition"].map(CUE_CONDITIONS)
    records = records.drop(columns=["region_id"], errors="ignore").merge(
        metadata[["filename", "region_id"]],
        on="filename",
        how="inner",
        validate="many_to_one",
    )
    return records[["filename", "region_id", "cue", "coverage_pct"]]


def summarize_cue_coverage(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby(["region_id", "cue"])["coverage_pct"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "mean_coverage_pct",
                "std": "std_coverage_pct",
            }
        )
    )


def cue_anova(records: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import f_oneway

    rows: list[dict] = []
    for cue, cue_records in records.groupby("cue", sort=True):
        groups = [
            region["coverage_pct"].dropna().to_numpy()
            for _, region in cue_records.groupby("region_id", sort=True)
        ]
        groups = [group for group in groups if len(group)]
        if len(groups) < 2:
            raise ValueError(f"ANOVA for {cue} requires at least two non-empty regions")
        statistic, p_value = f_oneway(*groups)
        rows.append(
            {
                "cue": cue,
                "regions": len(groups),
                "f_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)
