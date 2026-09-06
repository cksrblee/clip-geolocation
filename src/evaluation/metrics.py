"""Metrics used by the adaptation and controlled probing experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))
    lat1_rad = math.radians(float(lat1))
    lat2_rad = math.radians(float(lat2))
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def load_region_data(config_path: str | Path) -> list[dict[str, Any]]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Region config must contain a non-empty 'regions' list")
    return regions


def get_region_centers(
    region_data: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    centers: dict[str, tuple[float, float]] = {}
    for region in region_data:
        corners = region["bbox"]["corners"]
        latitudes = [float(corner["latitude"]) for corner in corners]
        longitudes = [float(corner["longitude"]) for corner in corners]
        centers[str(region["id"])] = (
            float(np.mean(latitudes)),
            float(np.mean(longitudes)),
        )
    return centers


def top1_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets)
    if prediction_array.shape != target_array.shape:
        raise ValueError("Predictions and targets must have the same shape")
    if target_array.size == 0:
        raise ValueError("Cannot calculate accuracy for an empty dataset")
    return float(np.mean(prediction_array == target_array) * 100.0)


def prediction_switch_rate(
    clean_predictions: Sequence[int],
    intervention_predictions: Sequence[int],
) -> float:
    """Percentage of samples whose predicted class changes after intervention."""
    clean = np.asarray(clean_predictions)
    intervened = np.asarray(intervention_predictions)
    if clean.shape != intervened.shape:
        raise ValueError("Clean and intervention predictions must have the same shape")
    if clean.size == 0:
        raise ValueError("Cannot calculate switch rate for an empty dataset")
    return float(np.mean(clean != intervened) * 100.0)


def accuracy_retention(intervention_accuracy: float, clean_accuracy: float) -> float:
    """Return SS = Acc(intervention) / Acc(clean) as a unitless ratio."""
    if clean_accuracy <= 0:
        raise ValueError("Clean accuracy must be positive")
    return float(intervention_accuracy / clean_accuracy)


def calculate_metrics(
    predictions: Sequence[int],
    ground_truth: Sequence[int],
    sample_coordinates: Sequence[tuple[float, float]],
    id_to_region: Mapping[int, str],
    region_centers: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    accuracy = top1_accuracy(predictions, ground_truth)
    prediction_array = np.asarray(predictions)
    coordinate_array = np.asarray(sample_coordinates, dtype=float)
    if coordinate_array.shape != (prediction_array.size, 2):
        raise ValueError(
            "Sample coordinates must contain one (latitude, longitude) pair per prediction"
        )
    errors = np.asarray(
        [
            haversine_distance(
                float(latitude),
                float(longitude),
                *region_centers[id_to_region[int(prediction)]],
            )
            for prediction, (latitude, longitude) in zip(
                prediction_array, coordinate_array
            )
        ],
        dtype=float,
    )
    return {
        "accuracy": accuracy,
        "mean_error_km": float(errors.mean()),
        "median_error_km": float(np.median(errors)),
        "raw_errors": errors,
    }


# Backward-compatible names retained for callers outside the paper pipeline.
calculate_shortcut_reliance = prediction_switch_rate
calculate_structure_sufficiency = accuracy_retention
