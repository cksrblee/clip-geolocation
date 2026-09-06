"""Area-normalized sampling and region-stratified dataset splitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


KM_PER_LATITUDE_DEGREE = 111.32


def bbox_area_km2(bbox: Mapping[str, Any]) -> float:
    """Approximate a small latitude/longitude bounding box area in square km."""
    corners = bbox.get("corners", [])
    if len(corners) < 2:
        raise ValueError("A region bbox must contain at least two corners")

    latitudes = [float(corner["latitude"]) for corner in corners]
    longitudes = [float(corner["longitude"]) for corner in corners]
    latitude_span = max(latitudes) - min(latitudes)
    longitude_span = max(longitudes) - min(longitudes)
    midpoint_latitude = math.radians((max(latitudes) + min(latitudes)) / 2.0)
    area = (
        latitude_span
        * KM_PER_LATITUDE_DEGREE
        * longitude_span
        * KM_PER_LATITUDE_DEGREE
        * math.cos(midpoint_latitude)
    )
    if area <= 0:
        raise ValueError("Region bbox area must be positive")
    return area


def infer_region_id(filename: str, region_ids: Sequence[str]) -> str | None:
    for region_id in sorted(region_ids, key=len, reverse=True):
        if filename.startswith(f"streetview_{region_id}_"):
            return region_id
    return None


def load_regions(config_path: str | Path) -> list[dict[str, Any]]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Region config must contain a non-empty 'regions' list")
    return regions


def add_region_ids(metadata: pd.DataFrame, region_ids: Sequence[str]) -> pd.DataFrame:
    if "filename" not in metadata:
        raise ValueError("Metadata is missing column: filename")

    result = metadata.copy()
    inferred = (
        result["filename"]
        .astype(str)
        .map(lambda filename: infer_region_id(filename, region_ids))
    )
    if "region_id" not in result:
        result["region_id"] = inferred
    else:
        result["region_id"] = result["region_id"].where(
            result["region_id"].isin(region_ids), inferred
        )
    return result[result["region_id"].isin(region_ids)].copy()


def area_normalized_sample(
    metadata: pd.DataFrame,
    regions: Sequence[Mapping[str, Any]],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample each region at the largest density supported by every region."""
    region_ids = [str(region["id"]) for region in regions]
    labeled = add_region_ids(metadata, region_ids)
    areas = {str(region["id"]): bbox_area_km2(region["bbox"]) for region in regions}
    counts = labeled["region_id"].value_counts().to_dict()

    empty_regions = [
        region_id for region_id in region_ids if counts.get(region_id, 0) == 0
    ]
    if empty_regions:
        raise ValueError(f"No images found for regions: {', '.join(empty_regions)}")

    target_density = min(
        counts[region_id] / areas[region_id] for region_id in region_ids
    )
    sampled_groups: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for region_id in region_ids:
        group = labeled[labeled["region_id"] == region_id]
        target_count = int(np.floor(target_density * areas[region_id]))
        sampled_groups.append(group.sample(n=target_count, random_state=seed))
        summary_rows.append(
            {
                "region_id": region_id,
                "area_km2": areas[region_id],
                "available_images": len(group),
                "available_density": len(group) / areas[region_id],
                "target_density": target_density,
                "sampled_images": target_count,
            }
        )

    sampled = pd.concat(sampled_groups, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled, pd.DataFrame(summary_rows)


def region_stratified_split(
    sampled: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and sum to less than 1")
    if "region_id" not in sampled:
        raise ValueError("Sampled metadata must contain region_id")

    from sklearn.model_selection import train_test_split

    train, remainder = train_test_split(
        sampled,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=sampled["region_id"],
    )
    test_ratio = 1.0 - train_ratio - val_ratio
    test_share = round(test_ratio / (val_ratio + test_ratio), 12)
    validation, test = train_test_split(
        remainder,
        test_size=test_share,
        random_state=seed,
        stratify=remainder["region_id"],
    )
    return {
        "train": train.reset_index(drop=True),
        "val": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def spatial_overlap_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Measure exact-coordinate overlap and nearest train distance for test images."""
    required = {"filename", "latitude", "longitude"}
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} metadata is missing: {', '.join(sorted(missing))}"
            )

    train_latitude = np.radians(train["latitude"].to_numpy(dtype=float))
    train_longitude = np.radians(train["longitude"].to_numpy(dtype=float))
    test_latitude = test["latitude"].to_numpy(dtype=float)
    test_longitude = test["longitude"].to_numpy(dtype=float)
    train_coordinates = set(
        zip(
            train["latitude"].to_numpy(dtype=float),
            train["longitude"].to_numpy(dtype=float),
        )
    )
    exact = np.asarray(
        [
            (latitude, longitude) in train_coordinates
            for latitude, longitude in zip(test_latitude, test_longitude)
        ],
        dtype=bool,
    )

    nearest_chunks: list[np.ndarray] = []
    radius_m = 6_371_000.0
    for start in range(0, len(test), chunk_size):
        latitude = np.radians(test_latitude[start : start + chunk_size])[:, None]
        longitude = np.radians(test_longitude[start : start + chunk_size])[:, None]
        delta_latitude = train_latitude[None, :] - latitude
        delta_longitude = train_longitude[None, :] - longitude
        haversine = (
            np.sin(delta_latitude / 2.0) ** 2
            + np.cos(latitude)
            * np.cos(train_latitude[None, :])
            * np.sin(delta_longitude / 2.0) ** 2
        )
        haversine = np.clip(haversine, 0.0, 1.0)
        distances = (
            radius_m * 2.0 * np.arctan2(np.sqrt(haversine), np.sqrt(1.0 - haversine))
        )
        nearest_chunks.append(distances.min(axis=1))

    return pd.DataFrame(
        {
            "filename": test["filename"].astype(str).to_numpy(),
            "exact_train_coordinate": exact,
            "nearest_train_distance_m": np.concatenate(nearest_chunks),
        }
    )


def build_splits(
    metadata_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sampled, summary = area_normalized_sample(
        pd.read_csv(metadata_path), load_regions(config_path), seed=seed
    )
    splits = region_stratified_split(sampled, train_ratio, val_ratio, seed)
    sampled.to_csv(output_path / "sampled_metadata.csv", index=False)
    summary.to_csv(output_path / "sampling_summary.csv", index=False)
    for split_name, split in splits.items():
        split.to_csv(output_path / f"{split_name}_metadata.csv", index=False)
    overlap = spatial_overlap_audit(splits["train"], splits["test"])
    overlap.to_csv(output_path / "test_spatial_overlap.csv", index=False)
    within_50m = overlap["nearest_train_distance_m"] <= 50.0
    pd.DataFrame(
        [
            {
                "test_images": len(overlap),
                "exact_train_coordinate": int(overlap["exact_train_coordinate"].sum()),
                "within_50m_of_train": int(within_50m.sum()),
                "beyond_50m_from_train": int((~within_50m).sum()),
            }
        ]
    ).to_csv(output_path / "spatial_overlap_summary.csv", index=False)
    return splits
