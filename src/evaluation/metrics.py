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


def geodesic_distance_km(
    coord1: tuple[float, float], coord2: tuple[float, float]
) -> float:
    """Compute geodesic distance in kilometers (uses geopy when installed)."""
    try:
        from geopy.distance import geodesic

        return float(geodesic(coord1, coord2).kilometers)
    except ImportError:
        return haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])


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


def calculate_shortcut_reliance(
    clean_accuracy: float, intervened_accuracy: float
) -> float:
    """Return SR = Acc(original) - Acc(intervention) in percentage points."""
    return float(clean_accuracy - intervened_accuracy)


def calculate_structure_sufficiency(
    intervened_accuracy: float, clean_accuracy: float
) -> float:
    """Return SS = Acc(intervention) / Acc(original) as a unitless ratio."""
    if clean_accuracy <= 0:
        return 0.0
    return float(intervened_accuracy / clean_accuracy)


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


def calc_centroid_error(
    predictions: Sequence[int],
    ground_truth: Sequence[int],
    id_to_region: Mapping[int, str],
    region_centers: Mapping[str, tuple[float, float]],
) -> tuple[float, np.ndarray]:
    """Compute region-centroid error matching 26-3-Summer (0 km when p == l)."""
    pred_array = np.asarray(predictions, dtype=int)
    gt_array = np.asarray(ground_truth, dtype=int)
    if pred_array.shape != gt_array.shape:
        raise ValueError("Predictions and ground_truth must have the same shape")
    if gt_array.size == 0:
        raise ValueError("Cannot calculate centroid error for an empty dataset")

    errors: list[float] = []
    for pred, gt in zip(pred_array.tolist(), gt_array.tolist()):
        if pred != gt:
            pred_region = id_to_region[pred]
            gt_region = id_to_region[gt]
            errors.append(
                geodesic_distance_km(
                    region_centers[pred_region], region_centers[gt_region]
                )
            )
        else:
            errors.append(0.0)
    error_array = np.asarray(errors, dtype=float)
    return float(error_array.mean()), error_array


def calculate_metrics(
    predictions: Sequence[int],
    ground_truth: Sequence[int],
    id_to_region: Mapping[int, str],
    region_centers: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    accuracy = top1_accuracy(predictions, ground_truth)
    mean_error_km, errors = calc_centroid_error(
        predictions, ground_truth, id_to_region, region_centers
    )
    return {
        "accuracy": accuracy,
        "mean_error_km": mean_error_km,
        "median_error_km": float(np.median(errors)),
        "raw_errors": errors,
    }


accuracy_retention = calculate_structure_sufficiency
