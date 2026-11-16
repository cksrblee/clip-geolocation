"""Visual-cue coverage summaries and one-way ANOVA across regions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


CUE_CONDITIONS = {
    "vegetation_masked": "vegetation",
    "sky_masked": "sky",
    "text_masked": "text",
    "vehicle_masked": "vehicle",
}


def compute_cue_coverage_from_images(
    metadata: pd.DataFrame,
    image_dir: str | Path,
    interventions_dir: str | Path,
    diff_threshold: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pixel-difference mask coverage matching 26-3-Summer/compute_visual_cue_statistics.py."""
    orig_root = Path(image_dir)
    inv_root = Path(interventions_dir)

    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    for row in tqdm(
        metadata.itertuples(index=False),
        total=len(metadata),
        desc="Analyzing cue masks",
    ):
        filename = str(row.filename)
        region_id = str(row.region_id)
        orig_path = orig_root / filename
        if not orig_path.is_file():
            continue

        with Image.open(orig_path) as im:
            orig_img = np.asarray(im.convert("RGB"), dtype=np.int16)
        total_pixels = orig_img.shape[0] * orig_img.shape[1]

        wide_row: dict[str, object] = {"filename": filename, "region": region_id}
        for condition, cue_name in CUE_CONDITIONS.items():
            inv_path = inv_root / condition / filename
            if inv_path.is_file():
                with Image.open(inv_path) as im_inv:
                    inv_img = np.asarray(im_inv.convert("RGB"), dtype=np.int16)
                diff = np.abs(orig_img - inv_img).sum(axis=2) > diff_threshold
                coverage_pct = float((diff.sum() / total_pixels) * 100.0)
            else:
                coverage_pct = float("nan")

            wide_row[f"{cue_name}_coverage_pct"] = coverage_pct
            if not np.isnan(coverage_pct):
                long_rows.append(
                    {
                        "filename": filename,
                        "region_id": region_id,
                        "cue": cue_name,
                        "coverage_pct": coverage_pct,
                    }
                )
        wide_rows.append(wide_row)

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


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
